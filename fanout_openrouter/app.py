from __future__ import annotations

import asyncio
import time
import uuid

import httpx
from fastapi import FastAPI, Header, HTTPException
import uvicorn

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ResponseMessage,
)
from .openrouter_client import OpenRouterClient
from .orchestrator import AllCandidatesFailedError, SynthesizerService
from .policy import get_policy
from .settings import Settings


def create_app(
    *,
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None,
    sleep_func=asyncio.sleep,
) -> FastAPI:
    app = FastAPI(title="fan-out-openrouter", version="0.1.0")
    app.state.settings = settings or Settings.from_env()
    app.state.transport = transport
    app.state.sleep_func = sleep_func

    @app.post("/api/v1/chat/completions", response_model=ChatCompletionResponse)
    async def create_chat_completion(
        request: ChatCompletionRequest,
        authorization: str | None = Header(default=None),
    ) -> ChatCompletionResponse:
        if request.stream:
            raise HTTPException(
                status_code=400, detail="stream=true is not supported yet"
            )

        policy = get_policy(request.model)
        if policy is None:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported virtual model: {request.model}",
            )

        settings = app.state.settings
        api_key = settings.openrouter_api_key or _extract_bearer_token(authorization)
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="OPENROUTER_API_KEY or inbound bearer token is required",
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
        except AllCandidatesFailedError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            await client.aclose()

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=result.model,
            choices=[
                Choice(
                    message=ResponseMessage(content=result.content),
                )
            ],
        )

    return app


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


app = create_app()


def main() -> None:
    uvicorn.run("fanout_openrouter.app:app", host="0.0.0.0", port=8000)


__all__ = ["app", "create_app", "main"]
