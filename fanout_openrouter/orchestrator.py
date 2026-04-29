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
        temperature: float | None,
    ) -> CompletionResult: ...


class AllCandidatesFailedError(RuntimeError):
    pass


@dataclass(frozen=True)
class OrchestratedResponse:
    content: str
    model: str
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
                request.temperature,
            )
            for model in distributed
        ]
        results = await asyncio.gather(*tasks)

        candidates: list[str] = []
        for index, (content, error, final_model) in enumerate(results, start=1):
            if error:
                logger.warning(
                    "candidate_failed",
                    extra={"candidate_index": index, "error": error},
                )
                continue

            logger.info(
                "candidate_completed",
                extra={"candidate_index": index, "model": final_model},
            )
            candidates.append(content)

        if not candidates:
            raise AllCandidatesFailedError("all candidates failed")

        if len(candidates) == 1:
            return OrchestratedResponse(
                content=candidates[0],
                model=policy.virtual_model,
                synthesized=False,
            )

        synthesis_prompt = build_synthesis_prompt(
            serialize_messages(request.messages),
            candidates,
        )
        synthesis_messages = [ChatMessage(role="user", content=synthesis_prompt)]
        final_output, error, _ = await self._run_with_default_model_retry(
            synthesis_messages,
            policy.synthesis_model,
            policy.default_fallback_model,
            request.temperature,
        )

        if error:
            logger.warning("synthesis_failed", extra={"error": error})
            return OrchestratedResponse(
                content=candidates[0],
                model=policy.virtual_model,
                synthesized=False,
            )

        return OrchestratedResponse(
            content=final_output,
            model=policy.virtual_model,
            synthesized=True,
        )

    async def _run_with_default_model_retry(
        self,
        messages: list[ChatMessage],
        model: str,
        default_model: str,
        temperature: float | None,
    ) -> tuple[str, str | None, str]:
        output, error = await self._run_with_retry(messages, model, temperature)
        if (error or not output.strip()) and model != default_model:
            logger.warning(
                "primary_model_failed_retrying_default",
                extra={"model": model, "default_model": default_model, "error": error},
            )
            output, error = await self._run_with_retry(
                messages,
                default_model,
                temperature,
            )
            return output, error, default_model
        return output, error, model

    async def _run_with_retry(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float | None,
    ) -> tuple[str, str | None]:
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
                    temperature=temperature,
                )
            except OpenRouterError as exc:
                last_error = str(exc)
                if attempt < self._max_attempts:
                    continue
                return "", last_error

            if not result.content.strip():
                last_error = "empty response"
                if attempt < self._max_attempts:
                    continue
                return "", last_error

            return result.content, None

        return (
            "",
            f"all {self._max_attempts} attempts failed (last error: {last_error})",
        )


def distribute_models(model_pool: list[str], n: int) -> list[str]:
    if not model_pool:
        return ["anthropic/claude-sonnet-4"] * n
    return [model_pool[index % len(model_pool)] for index in range(n)]


def serialize_messages(messages: list[ChatMessage]) -> str:
    blocks = []
    for message in messages:
        blocks.append(f"<{message.role}>\n{message.content.strip()}\n</{message.role}>")
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
