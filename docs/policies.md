# Policy Configuration Guide

The `fan-out-openrouter` proxy routes incoming requests based on policies defined in a JSON file. By default, this is `fanout_policies.json` in the root of the project, though you can override it using the `FANOUT_POLICY_FILE` environment variable.

This file defines "virtual models" that clients can request. When a client requests a virtual model, the proxy looks up its policy to determine how to fan out the request to actual upstream models.

## Common Use-Case Configurations

Based on the top ranked models on OpenRouter, here are some highly recommended policy presets spanning different tiers:

```json
{
  "policies": [
    {
      "virtual_model": "fanout/auto",
      "candidate_models": [
        "openrouter/auto"
      ],
      "fanout_count": 3,
      "synthesis_model": "openrouter/auto",
      "created": 1710000004
    },
    {
      "virtual_model": "fanout/frontier",
      "candidate_models": [
        "anthropic/claude-opus-4.7",
        "openai/gpt-5.5",
        "google/gemini-3.1-pro-preview"
      ],
      "fanout_count": 3,
      "synthesis_model": "anthropic/claude-opus-4.7",
      "default_fallback_model": "openai/gpt-5.5",
      "created": 1710000005
    },
    {
      "virtual_model": "fanout/minimal",
      "candidate_models": [
        "anthropic/claude-haiku-4.5",
        "openai/gpt-5.4-nano"
      ],
      "fanout_count": 2,
      "synthesis_model": "anthropic/claude-haiku-4.5",
      "created": 1710000000
    },
    {
      "virtual_model": "fanout/max-intelligence",
      "candidate_models": [
        "anthropic/claude-4.7-opus-20260416",
        "anthropic/claude-4.6-sonnet-20260217",
        "deepseek/deepseek-v3.2-20251201"
      ],
      "fanout_count": 3,
      "synthesis_model": "anthropic/claude-4.7-opus-20260416",
      "default_fallback_model": "anthropic/claude-4.6-sonnet-20260217",
      "created": 1710000001
    },
    {
      "virtual_model": "fanout/fast-consensus",
      "candidate_models": [
        "google/gemini-3-flash-preview-20251217",
        "x-ai/grok-4.1-fast",
        "stepfun/step-3.5-flash"
      ],
      "fanout_count": 5,
      "synthesis_model": "google/gemini-3-flash-preview-20251217",
      "created": 1710000002
    },
    {
      "virtual_model": "fanout/free-tier",
      "candidate_models": [
        "tencent/hy3-preview-20260421:free",
        "nvidia/nemotron-3-super-120b-a12b-20230311:free"
      ],
      "fanout_count": 4,
      "synthesis_model": "tencent/hy3-preview-20260421:free",
      "created": 1710000003
    }
  ]
}
```

### 1. `fanout/auto` (Dynamic Routing)
- **Goal:** Let OpenRouter dynamically pick the best models based on cost/performance while still leveraging consensus.
- **Strategy:** Queries `openrouter/auto` 3 times. OpenRouter will route each request dynamically, and then synthesis is performed by `openrouter/auto`.
- **Use Cases:** General purpose queries where you want cost-efficiency without giving up the benefits of multi-agent synthesis.

### 2. `fanout/frontier` (Absolute Vanguard)
- **Goal:** Absolute cutting-edge capabilities from the three major AI labs.
- **Strategy:** Queries the very latest flagship models (`Claude 4.7 Opus`, `GPT-5.5`, and `Gemini 3.1 Pro`) concurrently. Opus is then used to synthesize the absolute best response from all three industry titans.
- **Use Cases:** The hardest logical problems, novel mathematical research, extreme edge cases in coding.

### 3. `fanout/max-intelligence` (Tier 1)
- **Goal:** Best possible reasoning and coding answers for complex problems, regardless of cost.
- **Strategy:** Ask the 3 smartest models on the planet (Claude 4.7 Opus, Claude 4.6 Sonnet, and Deepseek v3.2) to draft solutions, and have Opus evaluate them and synthesize the absolute best response.
- **Use Cases:** Software architecture, advanced mathematics, refactoring massive codebases.

### 4. `fanout/fast-consensus` (Tier 2)
- **Goal:** Fast, cheap, and capable consensus logic.
- **Strategy:** Fanning out 5 times across ultra-fast models like Gemini 3 Flash, Grok 4.1 Fast, and Step 3.5 Flash. It round-robins them, gathering 5 responses in a matter of seconds, before asking Gemini 3 Flash to synthesize the consensus.
- **Use Cases:** Document summarization, routing classifiers, extracting data from large messy logs.

### 5. `fanout/free-tier` (Tier 3)
- **Goal:** Zero-cost inference.
- **Strategy:** Leverages models that are currently offered for free on OpenRouter, giving you a powerful multi-agent ensemble setup at $0.00 cost.
- **Use Cases:** Academic experimentation, personal hobby projects, high-volume repetitive tasks.

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
