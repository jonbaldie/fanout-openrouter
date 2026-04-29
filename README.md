# fan-out-openrouter

Tracer bullet for an OpenRouter-compatible chat completion API that fans out to OpenRouter, synthesizes candidate answers, and returns a standard chat completion response.

## Run

Set an upstream OpenRouter key and start the server:

```bash
OPENROUTER_API_KEY=... uv run uvicorn fanout_openrouter.app:app --reload
```

The server also loads `.env` automatically, so `OPENROUTER_API_KEY` can live there.

The server exposes:

- `POST /api/v1/chat/completions`
- `GET /api/v1/models`

## First Virtual Model

The tracer bullet loads virtual models from `fanout_policies.json` by default.

The bundled config currently includes:

- `fanout/minimal`

That policy performs two candidate calls and one synthesis call against OpenRouter.

You can override the policy file with `FANOUT_POLICY_FILE=/path/to/policies.json`.

## Test

```bash
uv run pytest
```

The default test suite includes live smoke coverage against OpenRouter, so `OPENROUTER_API_KEY` must be available via the environment or `.env`.
