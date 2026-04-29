# fan-out-openrouter

An OpenRouter-compatible chat completion proxy that fans out requests to multiple models via OpenRouter, synthesizes their candidate answers, and returns a standard chat completion response. This enables higher quality, consensus-driven outputs while maintaining strict adherence to the OpenAI wire protocol, including streaming and tool calling.

## Features

- **Drop-in replacement:** Implements the OpenAI/OpenRouter chat completion API contract perfectly. Works seamlessly with existing clients (e.g. standard OpenAI python/node clients).
- **Fan-out architecture:** Runs multiple inference paths concurrently to generate diverse candidate answers.
- **Synthesis engine:** Analyzes the candidates and generates a single, superior synthesized response.
- **Robust tool calling:** Candidate models suggest tool usage, which the synthesis model evaluates and accurately invokes.
- **Resilient routing:** Gracefully falls back to configured default models if candidate generation or synthesis encounters upstream errors.
- **Full streaming support:** Perfectly preserves OpenAI-style SSE streaming behavior, including mid-stream error propagation.

## Setup and Installation

Requirements:
- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) (Fast Python package and project manager)

```bash
# Clone the repository
git clone https://github.com/yourusername/fan-out-openrouter.git
cd fan-out-openrouter

# Install dependencies using uv
uv sync --all-extras
```

## Running the Server

Set an upstream OpenRouter key and start the server:

```bash
OPENROUTER_API_KEY=your_key_here uv run uvicorn fanout_openrouter.app:app --reload
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

The proxy loads virtual models from a JSON policy file (default: `fanout_policies.json`). This defines how requests to a specific virtual model are routed, fanned out, and synthesized.

Example `fanout_policies.json`:

```json
{
  "policies": [
    {
      "virtual_model": "fanout/minimal",
      "candidate_models": [
        "anthropic/claude-haiku-4.5",
        "openai/gpt-5.4-nano"
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

## Development and Testing

The project uses `pytest` for testing, `ruff` for linting/formatting, and relies heavily on live smoke testing against OpenRouter to ensure absolute contract parity.

```bash
# Run formatting and linting
uv run ruff format .
uv run ruff check .

# Run the test suite
# Note: The default test suite includes live smoke coverage against OpenRouter.
# OPENROUTER_API_KEY must be available via the environment or .env
uv run pytest
```

## Contributing

Contributions are welcome! Please ensure that:
1. You run `uv run ruff check .` and `uv run ruff format .` before submitting.
2. The entire test suite (`uv run pytest`) passes. If you are changing the API surface, pay special attention to the parity tests (`test_parity.py`).
3. You follow the existing code structure and formatting conventions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
