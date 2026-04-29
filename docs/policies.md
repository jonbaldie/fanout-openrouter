# Policy Configuration Guide

The `fan-out-openrouter` proxy routes incoming requests based on policies defined in a JSON file. By default, this is `fanout_policies.json` in the root of the project, though you can override it using the `FANOUT_POLICY_FILE` environment variable.

This file defines "virtual models" that clients can request. When a client requests a virtual model, the proxy looks up its policy to determine how to fan out the request to actual upstream models.

## Structure of `fanout_policies.json`

The file must contain a single JSON object with a `policies` array. Each object in that array defines one virtual model.

```json
{
  "policies": [
    {
      "virtual_model": "fanout/minimal",
      "candidate_models": [
        "anthropic/claude-3.5-haiku",
        "openai/gpt-4o-mini"
      ],
      "fanout_count": 2,
      "synthesis_model": "anthropic/claude-3.5-haiku",
      "default_fallback_model": "anthropic/claude-3.5-haiku",
      "created": 1710000000
    },
    {
      "virtual_model": "fanout/high-quality",
      "candidate_models": [
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o",
        "google/gemini-1.5-pro"
      ],
      "fanout_count": 5,
      "created": 1710000001
    }
  ]
}
```

## Schema Reference

### `virtual_model` (required)
The name of the model that clients will request. It's recommended to namespace this (e.g., `fanout/name` or `your-company/name`) to clearly distinguish it from upstream OpenRouter models. This name is surfaced when clients call the `GET /api/v1/models` endpoint.

### `candidate_models` (required)
A list of actual OpenRouter model IDs (e.g., `"openai/gpt-4o"`, `"anthropic/claude-3.5-sonnet"`) that will be used to generate the initial candidate answers.

### `fanout_count` (required)
The total number of concurrent upstream requests to make. 
- If this matches the length of `candidate_models`, each model is queried exactly once.
- If it exceeds the length of `candidate_models` (e.g., 5 candidates for 3 models), the proxy will round-robin through the `candidate_models` pool to reach the target count, giving you multiple distinct generations from the same models.

### `synthesis_model` (optional)
The OpenRouter model ID used to review all the candidate answers and synthesize the final output. If omitted, the proxy defaults to the first model listed in `candidate_models`. It is highly recommended to use a capable model (like Sonnet 3.5 or GPT-4o) for synthesis, even if the candidates are cheaper models.

### `default_fallback_model` (optional)
The OpenRouter model ID used as a fallback. If the candidate generation completely fails, or if the synthesis step crashes, the proxy will transparently issue a standard, non-fan-out request to this model to ensure the client still gets a response. If omitted, defaults to the first model in `candidate_models`.

### `created` (required)
A Unix timestamp integer. This is strictly used for the `GET /api/v1/models` API response to satisfy the OpenAI wire format.

## Reloading Policies

The proxy reads the policy file at startup. If you modify `fanout_policies.json`, you must restart the `uvicorn` server for the changes to take effect. If you are running in development mode via `just dev` (which uses `--reload`), saving the JSON file will automatically trigger a server restart.
