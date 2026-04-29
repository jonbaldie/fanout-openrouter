# AGENTS.md

This repo is being built as an OpenRouter-compatible facade, not a vaguely similar API.

## Working Style

- Build in tracer bullets. Start with the smallest real end-to-end slice, prove it works, then extend it.
- When discussing next steps, capture the slices in `TodoWrite` and then immediately start executing the top one.
- Keep changes small and direct. Don't sprawl unless there's a concrete need.

## API Contract

- CRITICAL: the external API surface needs to match OpenRouter.ai as closely as possible.
- That means request shape, response shape, error shape, and streaming behavior all matter.
- `stream=true` is not optional polish. It is part of the contract.
- If a behavior differs from OpenRouter/OpenAI, treat that as a real gap, not a nice-to-have.

## Testing Expectations

- CRITICAL: don't hide behind mocks for contract-critical behavior.
- Use the real OpenRouter API for smoke coverage.
- Live smoke tests should run by default, not behind an opt-in flag.
- Use the cheap configured default models for live verification unless there's a strong reason not to.
- Look for genuine failure modes and edge cases, then test those paths too.
- CRITICAL: use parity tests against real OpenRouter as the oracle for API-surface work.
- Start parity work in tracer bullets: one live case, fully end to end, verbose enough that progress and failure points are obvious.
- The parity loop is: hit real OpenRouter, hit our local service, normalize only truly non-deterministic fields, then diff the wire shape.

## Common Commands

- Install/update deps with `uv sync --all-extras`.
- Run the whole test suite with `uv run pytest`. This is the default verification path and it should hit real OpenRouter smoke coverage.
- Run the current tracer-bullet parity case with `uv run pytest tests/test_parity.py::test_parity_chat_non_stream_happy -v -s --tb=short`.
- Run the app locally with `uv run uvicorn fanout_openrouter.app:app --reload`.
- For a quick one-off manual check, it's fine to use `uv run python -c "..."` with `fastapi.testclient.TestClient` against `/api/v1/chat/completions`.
- If you're changing streaming or wire-shape behavior, don't stop at unit-ish confidence. Re-run `uv run pytest` and make sure the live contract checks still pass.

## Practical Guardrails

- Prefer real proof over simulated confidence.
- If something is only tested with mocks, assume it still isn't proven.
- Keep the app honest: validate the actual wire contract, not just internal helpers.
- When in doubt, bias toward the next smallest slice that increases real-world compatibility.
