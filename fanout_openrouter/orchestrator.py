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


class UpstreamClientError(RuntimeError):
    """All candidates failed with the same non-retryable upstream 4xx."""

    def __init__(
        self,
        *,
        status_code: int,
        message: str,
        code: int | str | None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code


@dataclass(frozen=True)
class OrchestratedResponse:
    content: str
    model: str
    provider: str | None
    system_fingerprint: str | None
    choice: dict[str, object]
    usage: dict[str, object] | None
    synthesized: bool


@dataclass(frozen=True)
class CandidateFailure:
    message: str
    last_exception: OpenRouterError | None


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
        failure_exceptions: list[OpenRouterError | None] = []
        for index, (result, failure, final_model) in enumerate(results, start=1):
            if failure is not None:
                logger.warning(
                    "candidate_failed",
                    extra={"candidate_index": index, "error": failure.message},
                )
                failures.append(f"candidate {index} ({final_model}): {failure.message}")
                failure_exceptions.append(failure.last_exception)
                continue

            logger.info(
                "candidate_completed",
                extra={"candidate_index": index, "model": final_model},
            )
            candidates.append(result)

        if not candidates:
            shared = _shared_upstream_client_error(failure_exceptions)
            if shared is not None:
                raise shared
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
        final_result, synth_failure, _ = await self._run_with_default_model_retry(
            synthesis_messages,
            policy.synthesis_model,
            policy.default_fallback_model,
            request.synthesis_request_body(),
        )

        if synth_failure is not None:
            logger.warning("synthesis_failed", extra={"error": synth_failure.message})
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
    ) -> tuple[CompletionResult | None, CandidateFailure | None, str]:
        result, failure = await self._run_with_retry(messages, model, extra_body)
        if (
            failure is not None or result is None or not result.content.strip()
        ) and model != default_model:
            logger.warning(
                "primary_model_failed_retrying_default",
                extra={
                    "model": model,
                    "default_model": default_model,
                    "error": failure.message if failure else None,
                },
            )
            result, failure = await self._run_with_retry(
                messages,
                default_model,
                extra_body,
            )
            return result, failure, default_model
        return result, failure, model

    async def _run_with_retry(
        self,
        messages: list[ChatMessage],
        model: str,
        extra_body: dict[str, object] | None,
    ) -> tuple[CompletionResult | None, CandidateFailure | None]:
        last_error: str | None = None
        last_exception: OpenRouterError | None = None
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
                last_exception = exc
                if attempt < self._max_attempts and exc.retryable:
                    continue
                return None, CandidateFailure(
                    message=last_error,
                    last_exception=last_exception,
                )

            if not result.content.strip():
                last_error = "empty response"
                last_exception = None
                if attempt < self._max_attempts:
                    continue
                return None, CandidateFailure(
                    message=last_error,
                    last_exception=last_exception,
                )

            return result, None

        return (
            None,
            CandidateFailure(
                message=(
                    f"all {self._max_attempts} attempts failed "
                    f"(last error: {last_error})"
                ),
                last_exception=last_exception,
            ),
        )


def _shared_upstream_client_error(
    exceptions: list[OpenRouterError | None],
) -> UpstreamClientError | None:
    if not exceptions:
        return None

    status_codes: set[int] = set()
    messages: set[str] = set()
    codes: set[int | str] = set()
    for exc in exceptions:
        if exc is None or exc.status_code is None:
            return None
        if exc.retryable:
            return None
        if exc.status_code < 400 or exc.status_code >= 500:
            return None
        status_codes.add(exc.status_code)
        if exc.upstream_message is not None:
            messages.add(exc.upstream_message)
        if exc.upstream_code is not None:
            codes.add(exc.upstream_code)

    if len(status_codes) != 1:
        return None

    # Need at least a consistent upstream message to echo; otherwise bail.
    if len(messages) != 1:
        return None

    (status_code,) = tuple(status_codes)
    (message,) = tuple(messages)
    code: int | str | None = None
    if len(codes) == 1:
        (code,) = tuple(codes)

    return UpstreamClientError(
        status_code=status_code,
        message=message,
        code=code,
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
