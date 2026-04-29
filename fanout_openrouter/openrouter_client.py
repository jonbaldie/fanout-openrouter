from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, AsyncIterator

import httpx

from .models import ChatMessage


class OpenRouterError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = True,
        upstream_message: str | None = None,
        upstream_code: int | str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.upstream_message = upstream_message
        self.upstream_code = upstream_code


@dataclass(frozen=True)
class CompletionResult:
    content: str
    model: str
    provider: str | None
    system_fingerprint: str | None
    choice: dict[str, Any]
    usage: dict[str, Any] | None
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class OpenRouterClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float,
        transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def create_chat_completion(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        extra_body: dict[str, Any] | None,
    ) -> CompletionResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [message.model_dump(exclude_none=True) for message in messages],
            "stream": False,
        }
        if extra_body:
            payload.update(extra_body)

        try:
            response = await self._client.post("chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            detail = exc.response.text.strip() or str(exc)
            upstream_message, upstream_code = _extract_upstream_error(exc.response)
            raise OpenRouterError(
                f"OpenRouter returned {status_code}: {detail}",
                status_code=status_code,
                retryable=_is_retryable_status(status_code),
                upstream_message=upstream_message,
                upstream_code=upstream_code,
            ) from exc
        except httpx.HTTPError as exc:
            raise OpenRouterError(str(exc)) from exc

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise OpenRouterError("OpenRouter returned no choices")

        message = choices[0].get("message") or {}
        tool_calls_raw = message.get("tool_calls")
        tool_calls = (
            tool_calls_raw
            if isinstance(tool_calls_raw, list) and tool_calls_raw
            else None
        )
        content = self._extract_content(message, allow_empty=tool_calls is not None)
        model_name = data.get("model") or model
        provider = data.get("provider")
        system_fingerprint = data.get("system_fingerprint")
        usage = data.get("usage")
        finish_reason_raw = choices[0].get("finish_reason")
        finish_reason = (
            finish_reason_raw if isinstance(finish_reason_raw, str) else None
        )
        return CompletionResult(
            content=content,
            model=model_name,
            provider=provider if isinstance(provider, str) else None,
            system_fingerprint=(
                system_fingerprint if isinstance(system_fingerprint, str) else None
            ),
            choice=choices[0],
            usage=usage if isinstance(usage, dict) else None,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )

    async def stream_chat_completion(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        extra_body: dict[str, Any] | None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Open a streaming chat completion against OpenRouter and yield each
        parsed data frame as a dict. The `[DONE]` sentinel is consumed and
        terminates the iterator. Upstream errors raised before the stream
        opens are surfaced as OpenRouterError with the same semantics as
        the non-streaming path, so the caller can retry or fall back.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": [message.model_dump(exclude_none=True) for message in messages],
            "stream": True,
        }
        if extra_body:
            payload.update(extra_body)

        request = self._client.build_request("POST", "chat/completions", json=payload)
        try:
            response = await self._client.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise OpenRouterError(str(exc)) from exc

        try:
            if response.status_code >= 400:
                body_bytes = await response.aread()
                detail = body_bytes.decode("utf-8", errors="replace").strip() or str(
                    response.status_code
                )
                upstream_message, upstream_code = _extract_upstream_error_from_body(
                    body_bytes
                )
                raise OpenRouterError(
                    f"OpenRouter returned {response.status_code}: {detail}",
                    status_code=response.status_code,
                    retryable=_is_retryable_status(response.status_code),
                    upstream_message=upstream_message,
                    upstream_code=upstream_code,
                )

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload_str = line[len("data: ") :]
                if payload_str == "[DONE]":
                    return
                try:
                    parsed = json.loads(payload_str)
                except ValueError:
                    # OpenRouter occasionally sends keep-alive comments or
                    # malformed fragments; skip them rather than abort.
                    continue
                if not isinstance(parsed, dict):
                    continue
                # Mid-stream error frames. OpenRouter reports these inline.
                error = parsed.get("error")
                if isinstance(error, dict):
                    message = error.get("message")
                    code = error.get("code")
                    status_code = code if isinstance(code, int) else None
                    raise OpenRouterError(
                        f"OpenRouter streamed error: "
                        f"{message if isinstance(message, str) else parsed}",
                        status_code=status_code,
                        retryable=(
                            _is_retryable_status(status_code)
                            if status_code is not None
                            else True
                        ),
                        upstream_message=message if isinstance(message, str) else None,
                        upstream_code=code if isinstance(code, (int, str)) else None,
                    )
                yield parsed
        finally:
            await response.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _extract_content(
        self,
        message: dict[str, Any],
        *,
        allow_empty: bool = False,
    ) -> str:
        """
        Extract assistant text from a completion message.

        `allow_empty=True` lets callers opt into tolerating a null/missing
        content payload, which is how OpenRouter signals a tool-call-only
        response (`message.content == null` with `tool_calls` populated).
        Retryable "empty response" behavior stays the default for the
        non-tool-call path so we still fail fast on genuinely empty
        answers.
        """
        content = message.get("content")
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
            if text_parts:
                return "".join(text_parts)

        if allow_empty and (content is None or content == ""):
            return ""

        raise OpenRouterError("OpenRouter returned a non-text message content payload")


def _is_retryable_status(status_code: int) -> bool:
    return status_code in {408, 409, 429} or status_code >= 500


def _extract_upstream_error(
    response: httpx.Response,
) -> tuple[str | None, int | str | None]:
    try:
        body = response.json()
    except (ValueError, json.JSONDecodeError):
        return None, None
    return _upstream_error_fields(body)


def _extract_upstream_error_from_body(
    body_bytes: bytes,
) -> tuple[str | None, int | str | None]:
    try:
        body = json.loads(body_bytes.decode("utf-8", errors="replace"))
    except (ValueError, UnicodeDecodeError):
        return None, None
    return _upstream_error_fields(body)


def _upstream_error_fields(
    body: Any,
) -> tuple[str | None, int | str | None]:
    if not isinstance(body, dict):
        return None, None

    error = body.get("error")
    if not isinstance(error, dict):
        return None, None

    message = error.get("message")
    code = error.get("code")
    if not isinstance(message, str):
        message = None
    if not isinstance(code, (int, str)):
        code = None
    return message, code
