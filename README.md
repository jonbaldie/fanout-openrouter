# fan-out-openrouter

Tracer bullet for an OpenRouter-compatible chat completion API that fans out to OpenRouter, synthesizes candidate answers, and returns a standard chat completion response.

## Run

Set an upstream OpenRouter key and start the server:

```bash
OPENROUTER_API_KEY=... uv run uvicorn fanout_openrouter.app:app --reload
```

The server exposes:

- `POST /api/v1/chat/completions`
- `GET /api/v1/models`

## Configuration

The server automatically loads environment variables from a `.env` file in the root directory.

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | *(None)* | Your OpenRouter API key. If not set, clients must pass it in the `Authorization: Bearer` header. |
| `FANOUT_POLICY_FILE` | `fanout_policies.json` | Path to the JSON configuration file defining the virtual models. |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | The base URL for upstream OpenRouter calls. |
| `OPENROUTER_TIMEOUT_SECONDS` | `60.0` | Upstream request timeout. |
| `FANOUT_DOTENV_PATH` | `.env` | Path to the dotenv file to load. |

## Model Policies

The tracer bullet loads virtual models from a JSON policy file (default: `fanout_policies.json`). This defines how requests to a specific virtual model are routed, fanned out, and synthesized.

Example `fanout_policies.json`:

```json
{
  "policies": [
    {
      "virtual_model": "fanout/minimal",
      "candidate_models": [
        "google/gemini-3-flash-preview",
        "anthropic/claude-3-haiku"
      ],
      "fanout_count": 3,
      "synthesis_model": "anthropic/claude-3-5-sonnet",
      "default_fallback_model": "openai/gpt-4o-mini",
      "created": 1710000000
    }
  ]
}
```

### Policy Schema
- `virtual_model`: The model ID clients will request (e.g. `fanout/minimal`).
- `candidate_models`: List of underlying OpenRouter model IDs to invoke.
- `fanout_count`: The total number of candidate requests to make. If this exceeds the number of `candidate_models`, the pool is round-robined.
- `synthesis_model` *(optional)*: The model to use for synthesizing the final response. Defaults to the first candidate model.
- `default_fallback_model` *(optional)*: A reliable model to fall back to if primary candidates or synthesis fail. Defaults to the first candidate model.
- `created`: Unix timestamp for when the virtual model was created (surfaced in `/models`).

## Tool Calling Synthesis

Unlike naive wrappers that bypass fan-out logic when tools are provided, `fanout-openrouter` provides **true tool calling synthesis**:

1. **Fanned Out Tool Choice:** The user's prompt and tool schema are fanned out to the candidate models.
2. **Tool Intent Extraction:** If candidates decide to invoke a tool, their tool calls are captured and formatted into text blocks (e.g., `[Tool Call: get_weather({"city": "Paris"})]`).
3. **True Synthesis:** The `synthesis_model` receives the candidate responses (including their intended tool calls) and is explicitly instructed to evaluate them. If the synthesis model agrees with the candidates, it invokes the final tool call natively, ensuring strict schema adherence for the downstream client.

## Test

```bash
uv run pytest
```

The default test suite includes live smoke coverage against OpenRouter, so `OPENROUTER_API_KEY` must be available via the environment or `.env`.
