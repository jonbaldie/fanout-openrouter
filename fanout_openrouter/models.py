from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_serializer


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None

    model_config = ConfigDict(extra="allow")


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    # Optional per-request candidate list. When provided, the proxy fans out to
    # these models rather than looking up a named virtual-model policy.
    models: list[str] | None = None
    temperature: float | None = None
    stream: bool = False

    model_config = ConfigDict(extra="allow")

    def candidate_request_body(self) -> dict[str, Any]:
        # Exclude routing fields that must not be forwarded to upstream models.
        body = self.model_dump(
            exclude_none=True,
            exclude={"model", "models", "messages", "stream"},
        )
        body.pop("n", None)
        return body

    def synthesis_request_body(self) -> dict[str, Any]:
        return self.candidate_request_body()

    def passthrough_request_body(self) -> dict[str, Any]:
        # For direct pass-through, forward everything except fields we manage
        # ourselves. `models` is intentionally excluded: if we're in pass-through
        # the list was empty, so there is nothing meaningful to forward.
        body = self.model_dump(
            exclude_none=True,
            exclude={"model", "models", "messages", "stream"},
        )
        body.pop("n", None)
        return body


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    # Content may be absent when the model returns tool_calls instead; keep
    # the wire shape honest rather than coercing null -> "".
    content: str | None = None
    refusal: str | None = None
    reasoning: str | None = None
    reasoning_details: list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None

    model_config = ConfigDict(extra="allow")

    @model_serializer(mode="wrap")
    def serialize_model(self, handler) -> dict[str, Any]:
        result = handler(self)
        # Strip fields that OpenRouter omits entirely when not applicable.
        # - tool_calls: omitted unless the model issued a function call.
        # - reasoning_details: omitted by OpenRouter for non-extended-thinking
        #   models; present only when the model produced structured reasoning.
        # NOTE: do NOT strip `reasoning`; OpenRouter always emits it as null
        # for models that don't produce chain-of-thought, so we must match.
        for field in ("tool_calls", "reasoning_details"):
            if result.get(field) is None:
                result.pop(field, None)
        return result


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    # OpenRouter emits a range of finish reasons ("stop", "length",
    # "tool_calls", "content_filter", etc.). Stay permissive so every
    # upstream value flows through unchanged.
    finish_reason: str = "stop"
    logprobs: dict[str, Any] | None = None
    native_finish_reason: str | None = None

    model_config = ConfigDict(extra="allow")


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    provider: str | None = None
    system_fingerprint: str | None = None
    choices: list[Choice]
    usage: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow")


class ErrorDetail(BaseModel):
    message: str
    code: int | str | None = None
    type: str | None = None
    param: str | None = None

    model_config = ConfigDict(extra="allow")


class ErrorResponse(BaseModel):
    error: ErrorDetail
    user_id: str | None = None

    model_config = ConfigDict(extra="allow")


class ModelCard(BaseModel):
    """
    OpenRouter-shaped model catalog item. Values for fields we don't track
    internally are set to sensible null-ish defaults to preserve the wire
    shape of /models.
    """

    id: str
    canonical_slug: str
    hugging_face_id: str | None = ""
    name: str
    created: int
    description: str = ""
    context_length: int | None = None
    architecture: dict[str, Any] = {}
    pricing: dict[str, Any] = {}
    top_provider: dict[str, Any] = {}
    per_request_limits: dict[str, Any] | None = None
    supported_parameters: list[str] = []
    default_parameters: dict[str, Any] = {}
    knowledge_cutoff: str | int | None = None
    expiration_date: str | None = None
    links: dict[str, Any] = {}

    model_config = ConfigDict(extra="allow")


class ModelsResponse(BaseModel):
    data: list[ModelCard]
