from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from typing import Awaitable, Callable, Protocol

from .models import ChatCompletionRequest, ChatMessage
from .openrouter_client import CompletionResult, OpenRouterError
from .policy import FanoutPolicy

logger = logging.getLogger(__name__)

SleepFunc = Callable[[float], Awaitable[None]]


class CompletionClient(Protocol):
    async def create_chat_completion(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        extra_body: dict[str, object] | None,
    ) -> CompletionResult: ...


class AllCandidatesFailedError(RuntimeError):
    pass


@dataclass(frozen=True)
class OrchestratedResponse:
    content: str
    model: str
    provider: str | None
    system_fingerprint: str | None
    choice: dict[str, object]
    usage: dict[str, object] | None
    synthesized: bool


class SynthesizerService:
    def __init__(
        self,
        client: CompletionClient,
        *,
        sleep_func: SleepFunc = asyncio.sleep,
        max_attempts: int = 10,
    ) -> None:
        self._client = client
        self._sleep = sleep_func
        self._max_attempts = max_attempts

    async def complete_chat(
        self,
        request: ChatCompletionRequest,
        policy: FanoutPolicy,
    ) -> OrchestratedResponse:
        distributed = distribute_models(
            list(policy.candidate_models), policy.fanout_count
        )
        tasks = [
            self._run_with_default_model_retry(
                request.messages,
                model,
                policy.default_fallback_model,
                request.candidate_request_body(),
            )
            for model in distributed
        ]
        results = await asyncio.gather(*tasks)

        candidates: list[CompletionResult] = []
        failures: list[str] = []
        for index, (result, error, final_model) in enumerate(results, start=1):
            if error:
                logger.warning(
                    "candidate_failed",
                    extra={"candidate_index": index, "error": error},
                )
                failures.append(f"candidate {index} ({final_model}): {error}")
                continue

            logger.info(
                "candidate_completed",
                extra={"candidate_index": index, "model": final_model},
            )
            candidates.append(result)

        if not candidates:
            raise AllCandidatesFailedError(
                "all candidates failed: " + "; ".join(failures)
            )

        if len(candidates) == 1:
            candidate = candidates[0]
            return OrchestratedResponse(
                content=candidate.content,
                model=candidate.model,
                provider=candidate.provider,
                system_fingerprint=candidate.system_fingerprint,
                choice=candidate.choice,
                usage=candidate.usage,
                synthesized=False,
            )

        synthesis_prompt = build_synthesis_prompt(
            serialize_messages(request.messages),
            [candidate.content for candidate in candidates],
        )
        synthesis_messages = [ChatMessage(role="user", content=synthesis_prompt)]
        final_result, error, _ = await self._run_with_default_model_retry(
            synthesis_messages,
            policy.synthesis_model,
            policy.default_fallback_model,
            request.synthesis_request_body(),
        )

        if error:
            logger.warning("synthesis_failed", extra={"error": error})
            candidate = candidates[0]
            return OrchestratedResponse(
                content=candidate.content,
                model=candidate.model,
                provider=candidate.provider,
                system_fingerprint=candidate.system_fingerprint,
                choice=candidate.choice,
                usage=candidate.usage,
                synthesized=False,
            )

        return OrchestratedResponse(
            content=final_result.content,
            model=final_result.model,
            provider=final_result.provider,
            system_fingerprint=final_result.system_fingerprint,
            choice=final_result.choice,
            usage=final_result.usage,
            synthesized=True,
        )

    async def _run_with_default_model_retry(
        self,
        messages: list[ChatMessage],
        model: str,
        default_model: str,
        extra_body: dict[str, object] | None,
    ) -> tuple[CompletionResult | None, str | None, str]:
        result, error = await self._run_with_retry(messages, model, extra_body)
        if (
            error or result is None or not result.content.strip()
        ) and model != default_model:
            logger.warning(
                "primary_model_failed_retrying_default",
                extra={"model": model, "default_model": default_model, "error": error},
            )
            result, error = await self._run_with_retry(
                messages,
                default_model,
                extra_body,
            )
            return result, error, default_model
        return result, error, model

    async def _run_with_retry(
        self,
        messages: list[ChatMessage],
        model: str,
        extra_body: dict[str, object] | None,
    ) -> tuple[CompletionResult | None, str | None]:
        last_error: str | None = None
        for attempt in range(1, self._max_attempts + 1):
            if attempt > 1:
                delay = min(2 ** (attempt - 2), 60)
                logger.warning(
                    "candidate_retry",
                    extra={
                        "attempt": attempt,
                        "max_attempts": self._max_attempts,
                        "delay": delay,
                        "last_error": last_error,
                    },
                )
                await self._sleep(delay)

            try:
                result = await self._client.create_chat_completion(
                    model=model,
                    messages=messages,
                    extra_body=extra_body,
                )
            except OpenRouterError as exc:
                last_error = str(exc)
                if attempt < self._max_attempts and exc.retryable:
                    continue
                return None, last_error

            if not result.content.strip():
                last_error = "empty response"
                if attempt < self._max_attempts:
                    continue
                return None, last_error

            return result, None

        return (
            None,
            f"all {self._max_attempts} attempts failed (last error: {last_error})",
        )


def distribute_models(model_pool: list[str], n: int) -> list[str]:
    if not model_pool:
        return ["anthropic/claude-sonnet-4"] * n
    return [model_pool[index % len(model_pool)] for index in range(n)]


def serialize_messages(messages: list[ChatMessage]) -> str:
    blocks = []
    for message in messages:
        content = _message_content_to_text(message)
        blocks.append(f"<{message.role}>\n{content}\n</{message.role}>")
    return "\n\n".join(blocks)


def build_synthesis_prompt(original_prompt: str, candidates: list[str]) -> str:
    parts = [original_prompt, "\n\n---\n\n"]
    parts.append(
        f"Below are {len(candidates)} candidate responses. "
        "Synthesize them into a single final answer and respond only with your synthesis in the same format.\n\n"
    )
    for index, candidate in enumerate(candidates, start=1):
        parts.append(
            f"<candidate_{index}>\n{candidate.strip()}\n</candidate_{index}>\n\n"
        )
    return "".join(parts)


def _message_content_to_text(message: ChatMessage) -> str:
    if isinstance(message.content, str):
        return message.content.strip()

    if not isinstance(message.content, list):
        return ""

    parts: list[str] = []
    for item in message.content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str):
            stripped = text.strip()
            if stripped:
                parts.append(stripped)
    return "\n".join(parts)
