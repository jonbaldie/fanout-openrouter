from __future__ import annotations

import asyncio

from fanout_openrouter.models import ChatCompletionRequest, ChatMessage
from fanout_openrouter.openrouter_client import CompletionResult, OpenRouterError
from fanout_openrouter.orchestrator import SynthesizerService
from fanout_openrouter.policy import FanoutPolicy


async def _no_sleep(_: float) -> None:
    return None


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self._candidate_index = 0
        self._failures_remaining: dict[str, int] = {}
        self._candidate_outputs = ["candidate one", "candidate two"]
        self._synthesis_output = "synthesized answer"
        self._synthesis_error: str | None = None

    async def create_chat_completion(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        temperature: float | None,
    ) -> CompletionResult:
        del temperature
        prompt = messages[0].content
        self.calls.append((model, prompt))

        remaining_failures = self._failures_remaining.get(model, 0)
        if remaining_failures > 0:
            self._failures_remaining[model] = remaining_failures - 1
            raise OpenRouterError(f"forced failure for {model}")

        if "Below are 2 candidate responses." in prompt:
            if self._synthesis_error:
                raise OpenRouterError(self._synthesis_error)
            return CompletionResult(content=self._synthesis_output, model=model)

        output = self._candidate_outputs[self._candidate_index]
        self._candidate_index += 1
        return CompletionResult(content=output, model=model)


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="fanout/minimal",
        messages=[ChatMessage(role="user", content="Explain tracer bullets.")],
    )


def test_complete_chat_synthesizes_candidates() -> None:
    client = FakeClient()
    service = SynthesizerService(client, sleep_func=_no_sleep)
    policy = FanoutPolicy(
        virtual_model="fanout/minimal",
        candidate_models=("model-a", "model-b"),
        synthesis_model="synthesis-model",
        fanout_count=2,
        default_fallback_model="model-a",
    )

    result = asyncio.run(service.complete_chat(_request(), policy))

    assert result.content == "synthesized answer"
    assert result.synthesized is True
    assert [model for model, _ in client.calls] == [
        "model-a",
        "model-b",
        "synthesis-model",
    ]


def test_complete_chat_falls_back_to_first_candidate_when_synthesis_fails() -> None:
    client = FakeClient()
    client._synthesis_error = "upstream synthesis failed"
    service = SynthesizerService(client, sleep_func=_no_sleep)
    policy = FanoutPolicy(
        virtual_model="fanout/minimal",
        candidate_models=("model-a", "model-b"),
        synthesis_model="synthesis-model",
        fanout_count=2,
        default_fallback_model="model-a",
    )

    result = asyncio.run(service.complete_chat(_request(), policy))

    assert result.content == "candidate one"
    assert result.synthesized is False


def test_primary_model_retries_then_uses_default_model() -> None:
    client = FakeClient()
    client._failures_remaining["model-bad"] = 10
    service = SynthesizerService(client, sleep_func=_no_sleep)
    policy = FanoutPolicy(
        virtual_model="fanout/minimal",
        candidate_models=("model-bad",),
        synthesis_model="model-a",
        fanout_count=1,
        default_fallback_model="model-a",
    )

    result = asyncio.run(service.complete_chat(_request(), policy))

    assert result.content == "candidate one"
    assert result.synthesized is False
    assert client.calls.count(("model-bad", "Explain tracer bullets.")) == 10
    assert client.calls.count(("model-a", "Explain tracer bullets.")) == 1
