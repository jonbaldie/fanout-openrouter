from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

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
        content = self._extract_content(message)
        model_name = data.get("model") or model
        provider = data.get("provider")
        system_fingerprint = data.get("system_fingerprint")
        usage = data.get("usage")
        return CompletionResult(
            content=content,
            model=model_name,
            provider=provider if isinstance(provider, str) else None,
            system_fingerprint=(
                system_fingerprint if isinstance(system_fingerprint, str) else None
            ),
            choice=choices[0],
            usage=usage if isinstance(usage, dict) else None,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _extract_content(self, message: dict[str, Any]) -> str:
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
