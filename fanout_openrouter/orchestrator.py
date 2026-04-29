from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from typing import Any, AsyncIterator, Awaitable, Callable, Protocol

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

    def stream_chat_completion(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        extra_body: dict[str, object] | None,
    ) -> AsyncIterator[dict[str, Any]]: ...


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


@dataclass(frozen=True)
class StreamPreamble:
    """Identity chosen for the final streamed response, emitted before any
    upstream deltas so the caller can shape the first SSE chunk."""

    model: str
    provider: str | None
    system_fingerprint: str | None
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
            [_format_candidate_response(candidate) for candidate in candidates],
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

    async def stream_chat(
        self,
        request: ChatCompletionRequest,
        policy: FanoutPolicy,
    ) -> AsyncIterator[tuple[StreamPreamble | None, dict[str, Any] | None]]:
        """
        Run candidates to completion, then stream the synthesis call's
        deltas as they arrive upstream. The iterator yields a sequence of
        `(preamble, chunk)` pairs:

          - exactly one leading pair where `preamble` is set and `chunk` is
            None: tells the caller which identity (model/provider) to
            stamp on outgoing SSE frames.
          - zero or more pairs where `preamble` is None and `chunk` is a
            raw upstream delta dict.

        Upstream errors raised before any delta is yielded propagate as
        UpstreamClientError / AllCandidatesFailedError, matching the
        non-streaming path so the HTTP layer can turn them into a JSON
        error response rather than an SSE body.

        If there's only one viable candidate (or synthesis itself fails
        before emitting any content), we fall back to a buffered stream
        built from the first candidate so the wire contract stays
        consistent from the client's perspective.
        """
        candidates = await self._run_candidates(request, policy)

        if len(candidates) == 1:
            candidate = candidates[0]
            preamble = StreamPreamble(
                model=candidate.model,
                provider=candidate.provider,
                system_fingerprint=candidate.system_fingerprint,
                synthesized=False,
            )
            buffered = _buffered_deltas(candidate)
            if buffered:
                yield preamble, buffered[0]
                for delta in buffered[1:]:
                    await asyncio.sleep(0.02)
                    yield None, delta
            else:
                yield preamble, None
            return

        synthesis_prompt = build_synthesis_prompt(
            serialize_messages(request.messages),
            [_format_candidate_response(candidate) for candidate in candidates],
        )
        synthesis_messages = [ChatMessage(role="user", content=synthesis_prompt)]
        extra_body = request.synthesis_request_body()

        # Attempt the primary synthesis model first. If it fails *before*
        # emitting any content, transparently fall back to the policy's
        # default fallback model. Once any delta has been yielded we're
        # committed - a mid-stream upstream error propagates as-is.
        primary_model = policy.synthesis_model
        fallback_model = policy.default_fallback_model

        stream_started = False
        try:
            async for item in self._stream_one_synthesis_attempt(
                synthesis_messages,
                primary_model,
                extra_body,
                synthesized=True,
            ):
                stream_started = True
                yield item
            return
        except OpenRouterError as exc:
            if stream_started:
                raise
            logger.warning(
                "synthesis_stream_primary_failed",
                extra={"model": primary_model, "error": str(exc)},
            )

        if primary_model != fallback_model:
            try:
                async for item in self._stream_one_synthesis_attempt(
                    synthesis_messages,
                    fallback_model,
                    extra_body,
                    synthesized=True,
                ):
                    stream_started = True
                    yield item
                return
            except OpenRouterError as exc:
                if stream_started:
                    raise
                logger.warning(
                    "synthesis_stream_fallback_failed",
                    extra={"model": fallback_model, "error": str(exc)},
                )

        # Both streamed attempts failed before producing content; emit the
        # first candidate as a buffered stream so the client still gets a
        # valid response.
        logger.warning("synthesis_stream_falling_back_to_candidate")
        candidate = candidates[0]
        preamble = StreamPreamble(
            model=candidate.model,
            provider=candidate.provider,
            system_fingerprint=candidate.system_fingerprint,
            synthesized=False,
        )
        buffered = _buffered_deltas(candidate)
        if buffered:
            yield preamble, buffered[0]
            for delta in buffered[1:]:
                await asyncio.sleep(0.02)
                yield None, delta
        else:
            yield preamble, None

    async def _stream_one_synthesis_attempt(
        self,
        synthesis_messages: list[ChatMessage],
        model: str,
        extra_body: dict[str, object] | None,
        *,
        synthesized: bool,
    ) -> AsyncIterator[tuple[StreamPreamble | None, dict[str, Any] | None]]:
        preamble_sent = False
        saw_content = False
        async for chunk in self._client.stream_chat_completion(
            model=model,
            messages=synthesis_messages,
            extra_body=extra_body,
        ):
            if not preamble_sent:
                preamble = _preamble_from_chunk(
                    chunk,
                    default_model=model,
                    synthesized=synthesized,
                )
                yield preamble, chunk
                preamble_sent = True
            else:
                yield None, chunk

            if _chunk_has_content(chunk) or _chunk_has_tool_calls(chunk):
                saw_content = True

        if not saw_content:
            raise OpenRouterError(
                f"synthesis stream from {model} produced no content",
                retryable=True,
            )

    async def _run_candidates(
        self,
        request: ChatCompletionRequest,
        policy: FanoutPolicy,
    ) -> list[CompletionResult]:
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
        return candidates

    async def _run_with_default_model_retry(
        self,
        messages: list[ChatMessage],
        model: str,
        default_model: str,
        extra_body: dict[str, object] | None,
    ) -> tuple[CompletionResult | None, CandidateFailure | None, str]:
        result, failure = await self._run_with_retry(messages, model, extra_body)
        content_missing = (
            result is not None and not result.content.strip() and not result.tool_calls
        )
        if (
            failure is not None or result is None or content_missing
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

            if not result.content.strip() and not result.tool_calls:
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


import random


def distribute_models(model_pool: list[str], n: int) -> list[str]:
    if not model_pool:
        return ["anthropic/claude-sonnet-4"] * n

    # If we need exactly the pool size, just shuffle it
    if n == len(model_pool):
        return random.sample(model_pool, k=n)

    # If we need fewer, pick a random subset
    if n < len(model_pool):
        return random.sample(model_pool, k=n)

    # If we need more than the pool has, round-robin but with shuffled bases
    # to ensure load is spread evenly across all upstream models over time
    result = []
    while len(result) < n:
        chunk = random.sample(model_pool, k=min(len(model_pool), n - len(result)))
        result.extend(chunk)
    return result


def serialize_messages(messages: list[ChatMessage]) -> str:
    blocks = []
    for message in messages:
        content = _message_content_to_text(message)
        blocks.append(f"<{message.role}>\n{content}\n</{message.role}>")
    return "\n\n".join(blocks)


def _format_candidate_response(candidate: CompletionResult) -> str:
    parts = []
    if candidate.content:
        parts.append(candidate.content.strip())
    if candidate.tool_calls:
        for tc in candidate.tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "unknown")
            args = func.get("arguments", "{}")
            parts.append(f"[Tool Call: {name}({args})]")
    if not parts:
        return "[Empty response]"
    return "\n".join(parts)


def build_synthesis_prompt(original_prompt: str, candidates: list[str]) -> str:
    parts = [original_prompt, "\n\n---\n\n"]
    parts.append(
        f"Below are {len(candidates)} candidate responses. "
        "Synthesize them into a single final answer. If the candidates indicate they wanted to call a tool (shown as [Tool Call: ...]), you MUST execute the corresponding tool using the native tool calling mechanism, rather than outputting text.\n\n"
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


def _chunk_has_content(chunk: dict[str, Any]) -> bool:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first = choices[0]
    if not isinstance(first, dict):
        return False
    delta = first.get("delta")
    if not isinstance(delta, dict):
        return False
    content = delta.get("content")
    return isinstance(content, str) and len(content) > 0


def _chunk_has_tool_calls(chunk: dict[str, Any]) -> bool:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first = choices[0]
    if not isinstance(first, dict):
        return False
    delta = first.get("delta")
    if not isinstance(delta, dict):
        return False
    tool_calls = delta.get("tool_calls")
    return isinstance(tool_calls, list) and len(tool_calls) > 0


def _preamble_from_chunk(
    chunk: dict[str, Any],
    *,
    default_model: str,
    synthesized: bool,
) -> StreamPreamble:
    model = chunk.get("model") if isinstance(chunk.get("model"), str) else None
    provider = chunk.get("provider") if isinstance(chunk.get("provider"), str) else None
    system_fingerprint = (
        chunk.get("system_fingerprint")
        if isinstance(chunk.get("system_fingerprint"), str)
        else None
    )
    return StreamPreamble(
        model=model or default_model,
        provider=provider,
        system_fingerprint=system_fingerprint,
        synthesized=synthesized,
    )


def _buffered_deltas(candidate: CompletionResult) -> list[dict[str, Any]]:
    """
    Build a synthetic delta stream for a fully-buffered candidate
    response. Used as a fallback when we can't stream the synthesis
    call for real (e.g. single-candidate policy, synthesis failure).
    This mirrors OpenRouter's chunk shape so the caller can treat it
    uniformly with a real stream.
    """
    message = (
        candidate.choice.get("message") if isinstance(candidate.choice, dict) else None
    )
    if not isinstance(message, dict):
        message = {}

    role_raw = message.get("role")
    role = role_raw if isinstance(role_raw, str) and role_raw else "assistant"

    finish_raw = (
        candidate.choice.get("finish_reason")
        if isinstance(candidate.choice, dict)
        else None
    )
    finish_reason = finish_raw if isinstance(finish_raw, str) and finish_raw else "stop"

    native_finish_raw = (
        candidate.choice.get("native_finish_reason")
        if isinstance(candidate.choice, dict)
        else None
    )
    native_finish_reason = (
        native_finish_raw if isinstance(native_finish_raw, str) else None
    )

    chunks: list[dict[str, Any]] = []

    # Initial role-only delta
    chunks.append(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": role, "content": ""},
                    "finish_reason": None,
                    "native_finish_reason": None,
                }
            ]
        }
    )

    # Content, sliced into smallish pieces. This is a fallback path only.
    content = candidate.content or ""
    if content:
        piece_size = 64
        for start in range(0, len(content), piece_size):
            piece = content[start : start + piece_size]
            chunks.append(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": role, "content": piece},
                            "finish_reason": None,
                            "native_finish_reason": None,
                        }
                    ]
                }
            )

    # If the candidate produced tool calls, emit them in a dedicated delta.
    if candidate.tool_calls:
        chunks.append(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": role,
                            "content": None,
                            "tool_calls": candidate.tool_calls,
                        },
                        "finish_reason": None,
                        "native_finish_reason": None,
                    }
                ]
            }
        )

    # Terminal delta with finish_reason and (optionally) usage.
    terminal: dict[str, Any] = {
        "choices": [
            {
                "index": 0,
                "delta": {"role": role, "content": ""},
                "finish_reason": finish_reason,
                "native_finish_reason": native_finish_reason,
            }
        ]
    }
    if candidate.usage:
        terminal["usage"] = candidate.usage
    chunks.append(terminal)

    return chunks
