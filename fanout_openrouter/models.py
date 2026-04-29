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
    temperature: float | None = None
    stream: bool = False

    model_config = ConfigDict(extra="allow")

    def candidate_request_body(self) -> dict[str, Any]:
        return self.model_dump(
            exclude_none=True,
            exclude={"model", "messages", "stream"},
        )

    def synthesis_request_body(self) -> dict[str, Any]:
        body = self.candidate_request_body()
        for key in (
            "n",
            "response_format",
            "tool_choice",
            "tools",
            "parallel_tool_calls",
        ):
            body.pop(key, None)
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
        if result.get("tool_calls") is None:
            result.pop("tool_calls", None)
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
