# fan-out-openrouter

Tracer bullet for an OpenRouter-compatible chat completion API that fans out to OpenRouter, synthesizes candidate answers, and returns a standard chat completion response.

## Run

Set an upstream OpenRouter key and start the server:

```bash
OPENROUTER_API_KEY=... uv run uvicorn fanout_openrouter.app:app --reload
```

The server exposes `POST /api/v1/chat/completions`.

## First Virtual Model

The tracer bullet currently supports one hardcoded virtual model:

- `fanout/minimal`

That policy performs two candidate calls and one synthesis call against OpenRouter.

## Test

```bash
uv run pytest
```
