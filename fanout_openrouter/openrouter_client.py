from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .models import ChatMessage


class OpenRouterError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompletionResult:
    content: str
    model: str


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
        temperature: float | None,
    ) -> CompletionResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [message.model_dump(exclude_none=True) for message in messages],
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        try:
            response = await self._client.post("chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip() or str(exc)
            raise OpenRouterError(
                f"OpenRouter returned {exc.response.status_code}: {detail}"
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
        return CompletionResult(content=content, model=model_name)

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
