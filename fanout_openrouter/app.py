from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ErrorDetail,
    ErrorResponse,
    ModelCard,
    ModelsResponse,
    ResponseMessage,
)
from .openrouter_client import OpenRouterClient
from .orchestrator import (
    AllCandidatesFailedError,
    SynthesizerService,
    UpstreamClientError,
)
from .policy import PolicyRegistry
from .settings import Settings


class FanoutAPIError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        code: int | str | None = None,
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code
        self.param = param


def create_app(
    *,
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None,
    sleep_func=asyncio.sleep,
) -> FastAPI:
    app = FastAPI(title="fan-out-openrouter", version="0.1.0")
    resolved_settings = settings or Settings.from_env()
    app.state.settings = resolved_settings
    app.state.transport = transport
    app.state.sleep_func = sleep_func
    app.state.policy_registry = PolicyRegistry.from_file(resolved_settings.policy_file)

    @app.exception_handler(FanoutAPIError)
    async def handle_fanout_api_error(
        _: Request,
        exc: FanoutAPIError,
    ) -> JSONResponse:
        return _error_response(
            status_code=exc.status_code,
            message=exc.message,
            code=exc.code,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        _: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return _error_response(
            status_code=400,
            message=_validation_error_message(exc),
            code=400,
        )

    @app.get("/api/v1/models", response_model=ModelsResponse)
    async def list_models() -> ModelsResponse:
        registry = app.state.policy_registry
        return ModelsResponse(
            data=[
                ModelCard(id=policy.virtual_model, created=policy.created)
                for policy in registry.list()
            ]
        )

    @app.post("/api/v1/chat/completions", response_model=ChatCompletionResponse)
    async def create_chat_completion(
        request: ChatCompletionRequest,
        authorization: str | None = Header(default=None),
    ) -> ChatCompletionResponse | StreamingResponse:
        policy = app.state.policy_registry.get(request.model)
        if policy is None:
            raise FanoutAPIError(
                400,
                f"{request.model} is not a valid model ID",
                code=400,
            )

        settings = app.state.settings
        api_key = settings.openrouter_api_key or _extract_bearer_token(authorization)
        if not api_key:
            raise FanoutAPIError(
                401,
                "No cookie auth credentials found",
                code=401,
            )

        client = OpenRouterClient(
            api_key=api_key,
            base_url=settings.openrouter_base_url,
            timeout=settings.request_timeout_seconds,
            transport=app.state.transport,
        )
        service = SynthesizerService(client, sleep_func=app.state.sleep_func)

        try:
            result = await service.complete_chat(request, policy)
        except UpstreamClientError as exc:
            raise FanoutAPIError(
                exc.status_code,
                exc.message,
                code=exc.code,
            ) from exc
        except AllCandidatesFailedError as exc:
            raise FanoutAPIError(
                502,
                str(exc),
                code="all_candidates_failed",
            ) from exc
        finally:
            await client.aclose()

        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        if request.stream:
            return StreamingResponse(
                _stream_chat_completion(
                    response_id=response_id,
                    created=created,
                    model=result.model,
                    provider=result.provider,
                    system_fingerprint=result.system_fingerprint,
                    choice=result.choice,
                    content=result.content,
                    usage=result.usage,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        return ChatCompletionResponse(
            id=response_id,
            created=created,
            model=result.model,
            provider=result.provider,
            system_fingerprint=result.system_fingerprint,
            choices=[Choice.model_validate(result.choice)],
            usage=result.usage,
        )

    return app


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


async def _stream_chat_completion(
    *,
    response_id: str,
    created: int,
    model: str,
    provider: str | None,
    system_fingerprint: str | None,
    choice: dict[str, object],
    content: str,
    usage: dict[str, object] | None,
) -> AsyncIterator[str]:
    message = choice.get("message")
    if not isinstance(message, dict):
        message = {}

    assistant_role = message.get("role")
    if not isinstance(assistant_role, str) or not assistant_role:
        assistant_role = "assistant"

    initial_delta = {"content": "", "role": assistant_role}
    if "reasoning" in message:
        initial_delta["reasoning"] = message.get("reasoning")
    if "reasoning_details" in message:
        initial_delta["reasoning_details"] = message.get("reasoning_details")

    yield _sse_data(
        _stream_chunk_payload(
            response_id=response_id,
            created=created,
            model=model,
            provider=provider,
            system_fingerprint=system_fingerprint,
            delta=initial_delta,
            finish_reason=None,
            native_finish_reason=None,
        )
    )

    for piece in _stream_content_pieces(content):
        yield _sse_data(
            _stream_chunk_payload(
                response_id=response_id,
                created=created,
                model=model,
                provider=provider,
                system_fingerprint=system_fingerprint,
                delta={"content": piece, "role": assistant_role},
                finish_reason=None,
                native_finish_reason=None,
            )
        )

    final_delta = {"content": "", "role": assistant_role}
    if "reasoning" in message:
        final_delta["reasoning"] = message.get("reasoning")

    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason:
        finish_reason = "stop"

    native_finish_reason = choice.get("native_finish_reason")
    if native_finish_reason is not None and not isinstance(native_finish_reason, str):
        native_finish_reason = None

    terminal_chunk = _stream_chunk_payload(
        response_id=response_id,
        created=created,
        model=model,
        provider=provider,
        system_fingerprint=system_fingerprint,
        delta=final_delta,
        finish_reason=finish_reason,
        native_finish_reason=native_finish_reason,
    )
    yield _sse_data(terminal_chunk)
    if usage:
        yield _sse_data(
            {
                **_stream_chunk_payload(
                    response_id=response_id,
                    created=created,
                    model=model,
                    provider=provider,
                    system_fingerprint=system_fingerprint,
                    delta={"content": "", "role": assistant_role},
                    finish_reason=finish_reason,
                    native_finish_reason=native_finish_reason,
                ),
                "usage": usage,
            }
        )
    yield _sse_data("[DONE]")


def _stream_content_pieces(content: str, chunk_size: int = 64) -> list[str]:
    if not content:
        return []
    return [
        content[index : index + chunk_size]
        for index in range(0, len(content), chunk_size)
    ]


def _stream_chunk_payload(
    *,
    response_id: str,
    created: int,
    model: str,
    provider: str | None,
    system_fingerprint: str | None,
    delta: dict[str, object],
    finish_reason: str | None,
    native_finish_reason: str | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
                "native_finish_reason": native_finish_reason,
            }
        ],
    }
    if provider is not None:
        payload["provider"] = provider
    if system_fingerprint is not None:
        payload["system_fingerprint"] = system_fingerprint
    return payload


def _sse_data(payload: object) -> str:
    if isinstance(payload, str):
        data = payload
    else:
        data = json.dumps(payload, separators=(",", ":"))
    return f"data: {data}\n\n"


def _error_response(
    *,
    status_code: int,
    message: str,
    code: int | str | None = None,
    param: str | None = None,
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(
            message=message,
            param=param,
            code=code,
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(exclude_none=True),
    )


def _validation_error_message(exc: RequestValidationError) -> str:
    param = _validation_error_param(exc)
    if param == "messages":
        return 'Input required: specify "prompt" or "messages"'

    first_error = exc.errors()[0] if exc.errors() else None
    message = (
        first_error.get("msg", "invalid request") if first_error else "invalid request"
    )
    if param:
        return f"{param}: {message}"
    return message


def _validation_error_param(exc: RequestValidationError) -> str | None:
    if not exc.errors():
        return None

    location = exc.errors()[0].get("loc") or ()
    filtered = [str(part) for part in location if part != "body"]
    if not filtered:
        return None
    return ".".join(filtered)


app = create_app()


def main() -> None:
    uvicorn.run("fanout_openrouter.app:app", host="0.0.0.0", port=8000)


__all__ = ["app", "create_app", "main"]
