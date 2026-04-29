from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


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
    content: str
    refusal: str | None = None
    reasoning: str | None = None
    reasoning_details: list[dict[str, Any]] | None = None

    model_config = ConfigDict(extra="allow")


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: Literal["stop"] = "stop"
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
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "fan-out-openrouter"


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
