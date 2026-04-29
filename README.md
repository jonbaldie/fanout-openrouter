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
        "openai/gpt-4o-mini"
      ],
      "fanout_count": 2,
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

When a request includes `tools`, the fan-out and synthesis process still applies:

1. **Candidate Evaluation:** The prompt and tool schemas are sent to all candidate models.
2. **Serialization:** If a candidate decides to invoke a tool, its tool call is serialized into a text block (e.g., `[Tool Call: get_weather({"city": "Paris"})]`) so the synthesizer can evaluate the intent.
3. **Synthesis & Execution:** The synthesis model receives the candidate responses (including the serialized tool calls) along with the original tool schemas. It is instructed to evaluate the candidates' intents and, if appropriate, execute the final tool call natively. This ensures strict schema adherence for the downstream client.

## Test

```bash
uv run pytest
```

The default test suite includes live smoke coverage against OpenRouter, so `OPENROUTER_API_KEY` must be available via the environment or `.env`.
