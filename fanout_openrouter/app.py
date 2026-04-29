from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import AsyncIterator
from collections import defaultdict

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
)
from .openrouter_client import OpenRouterClient
from .orchestrator import (
    AllCandidatesFailedError,
    SynthesizerService,
    UpstreamClientError,
)
from .policy import PolicyRegistry
from .settings import Settings
from .logging import configure_logging

configure_logging(structured=False)  # Can be made configurable via env later


class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.window_size = 60.0
        self.users: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, user_id: str) -> bool:
        if self.requests_per_minute <= 0:
            return True

        now = time.monotonic()
        user_timestamps = self.users[user_id]

        # Keep only timestamps within the window
        valid_idx = 0
        for i, ts in enumerate(user_timestamps):
            if now - ts <= self.window_size:
                valid_idx = i
                break
        else:
            valid_idx = len(user_timestamps)

        del user_timestamps[:valid_idx]

        if len(user_timestamps) >= self.requests_per_minute:
            return False

        user_timestamps.append(now)
        return True


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
    app.state.rate_limiter = RateLimiter(resolved_settings.rate_limit_rpm)

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
            data=[_virtual_model_card(policy) for policy in registry.list()]
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

        if not app.state.rate_limiter.is_allowed(api_key):
            raise FanoutAPIError(
                429,
                "Too many requests",
                code=429,
            )

        client = OpenRouterClient(
            api_key=api_key,
            base_url=settings.openrouter_base_url,
            timeout=settings.request_timeout_seconds,
            transport=app.state.transport,
        )
        service = SynthesizerService(client, sleep_func=app.state.sleep_func)
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        if request.stream:
            return await _handle_streaming(
                service=service,
                client=client,
                request=request,
                policy=policy,
                response_id=response_id,
                created=created,
            )

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


async def _handle_streaming(
    *,
    service,
    client,
    request: ChatCompletionRequest,
    policy,
    response_id: str,
    created: int,
) -> StreamingResponse:
    """
    Consume the synthesizer's streaming iterator. Pre-roll the first chunk
    (which comes bundled with the preamble) before returning the response
    so that upstream auth / routing errors surface as JSON, not SSE.
    OpenRouter does the same: stream=true with bad auth yields a plain JSON
    error, not an SSE body.
    """
    stream_iter = service.stream_chat(request, policy).__aiter__()

    early_comments: list[str] = []
    try:
        while True:
            preamble, first_chunk = await stream_iter.__anext__()
            if preamble is not None:
                break
            if isinstance(first_chunk, str):
                early_comments.append(first_chunk)
    except UpstreamClientError as exc:
        await client.aclose()
        raise FanoutAPIError(
            exc.status_code,
            exc.message,
            code=exc.code,
        ) from exc
    except AllCandidatesFailedError as exc:
        await client.aclose()
        raise FanoutAPIError(
            502,
            str(exc),
            code="all_candidates_failed",
        ) from exc
    except StopAsyncIteration:
        await client.aclose()
        raise FanoutAPIError(
            502,
            "synthesizer produced no output",
            code="empty_stream",
        ) from None

    if preamble is None:
        await client.aclose()
        raise FanoutAPIError(
            500,
            "synthesizer stream missing preamble",
            code="internal_error",
        )

    async def body() -> AsyncIterator[str]:
        try:
            for comment in early_comments:
                yield f"{comment}\n\n"

            if first_chunk is None:
                # Upstream yielded no content before closing; emit a synthetic
                # empty reply so the client sees a valid role/content payload.
                yield _sse_data(
                    _stream_chunk_payload(
                        response_id=response_id,
                        created=created,
                        model=preamble.model,
                        provider=preamble.provider,
                        system_fingerprint=preamble.system_fingerprint,
                        delta={"role": "assistant", "content": ""},
                        finish_reason="stop",
                        native_finish_reason=None,
                    )
                )
            else:
                # If there's a first chunk, just stream it (and all subsequent
                # chunks) exactly as they arrived from upstream, merely
                # restamping the top-level envelope fields. We do not inject
                # synthetic chunks because that shifts SSE boundaries and breaks
                # strict wire parity, especially for tool_calls.
                yield _sse_data(
                    _restamp_upstream_chunk(
                        first_chunk,
                        response_id=response_id,
                        created=created,
                        preamble=preamble,
                    )
                )

            async for preamble_next, chunk in stream_iter:
                if chunk is None:
                    continue
                if isinstance(chunk, str):
                    yield f"{chunk}\n\n"
                    continue
                yield _sse_data(
                    _restamp_upstream_chunk(
                        chunk,
                        response_id=response_id,
                        created=created,
                        preamble=preamble,
                    )
                )

            yield _sse_data("[DONE]")
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning("mid_stream_error", exc_info=exc)

            error_data = {
                "error": {
                    "message": "upstream error during streaming synthesis",
                    "code": 502,
                }
            }
            if isinstance(exc, UpstreamClientError):
                error_data["error"]["message"] = exc.message
                if exc.code is not None:
                    error_data["error"]["code"] = exc.code  # type: ignore
            elif isinstance(exc, AllCandidatesFailedError):
                error_data["error"]["message"] = str(exc)
                error_data["error"]["code"] = "all_candidates_failed"  # type: ignore
            elif hasattr(exc, "upstream_message") and exc.upstream_message:
                error_data["error"]["message"] = exc.upstream_message
                if hasattr(exc, "upstream_code") and exc.upstream_code is not None:
                    error_data["error"]["code"] = exc.upstream_code  # type: ignore

            yield _sse_data(error_data)
            # Do NOT yield [DONE] after an error block per typical SSE conventions
        finally:
            await client.aclose()

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _role_from_chunk(chunk: dict[str, object] | None) -> str | None:
    if not isinstance(chunk, dict):
        return None
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    delta = first.get("delta")
    if not isinstance(delta, dict):
        return None
    role = delta.get("role")
    return role if isinstance(role, str) and role else None


def _restamp_upstream_chunk(
    chunk: dict[str, object],
    *,
    response_id: str,
    created: int,
    preamble,
) -> dict[str, object]:
    """
    Rewrite id / created / model / provider / system_fingerprint so the
    client sees our facade's identity, not the upstream's. Everything else
    (choices, finish_reason, usage, reasoning, etc.) is forwarded as-is
    so we preserve the upstream's wire shape intact.
    """
    copy = dict(chunk)
    copy["id"] = response_id
    copy["object"] = "chat.completion.chunk"
    copy["created"] = created
    copy["model"] = preamble.model
    if preamble.provider is not None:
        copy["provider"] = preamble.provider
    else:
        copy.pop("provider", None)
    if preamble.system_fingerprint is not None:
        copy["system_fingerprint"] = preamble.system_fingerprint
    else:
        copy.pop("system_fingerprint", None)
    return copy


def _virtual_model_card(policy) -> ModelCard:  # type: ignore[no-untyped-def]
    return ModelCard(
        id=policy.virtual_model,
        canonical_slug=policy.virtual_model,
        hugging_face_id="",
        name=policy.virtual_model,
        created=policy.created,
        description=(
            f"Virtual fan-out model '{policy.virtual_model}' that fans out to "
            f"candidate models and synthesizes their responses."
        ),
        context_length=None,
        architecture={
            "modality": "text->text",
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "tokenizer": "Unknown",
            "instruct_type": None,
        },
        pricing={"prompt": "0", "completion": "0"},
        top_provider={
            "context_length": None,
            "max_completion_tokens": None,
            "is_moderated": False,
        },
        per_request_limits=None,
        supported_parameters=[
            "max_tokens",
            "messages",
            "stream",
            "temperature",
        ],
        default_parameters={
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "frequency_penalty": None,
            "presence_penalty": None,
            "repetition_penalty": None,
        },
        knowledge_cutoff=None,
        expiration_date=None,
        links={},
    )


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


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
