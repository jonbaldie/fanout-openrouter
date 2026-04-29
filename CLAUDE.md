# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->


## Build & Test

```bash
# Install / update deps
uv sync --all-extras

# Run the full test suite (hits real OpenRouter by default)
uv run pytest

# Run parity tests only
uv run pytest tests/test_parity.py -v -s --tb=short

# Lint and format
uv run ruff check .
uv run ruff format .

# Run the server locally
uv run uvicorn fanout_openrouter.app:app --reload
```

`OPENROUTER_API_KEY` must be set in env or `.env` for live tests to pass.

## Architecture Overview

FastAPI facade that fans out a single `/api/v1/chat/completions` request to multiple upstream OpenRouter models concurrently, then synthesizes their responses into one. Implements the OpenAI/OpenRouter wire contract exactly (request shape, response shape, error shape, SSE streaming).

Key modules:
- `fanout_openrouter/app.py` — FastAPI app, SSE streaming assembly
- `fanout_openrouter/orchestrator.py` — SynthesizerService: fan-out, retry, fallback, synthesis
- `fanout_openrouter/openrouter_client.py` — httpx client against real OpenRouter
- `fanout_openrouter/policy.py` — Virtual model → candidate-model registry (loaded from `fanout_policies.json`)
- `fanout_openrouter/models.py` — Pydantic v2 wire models
- `fanout_openrouter/settings.py` — Env/.env driven settings

## Conventions & Patterns

- Match OpenRouter/OpenAI wire shapes exactly — any divergence is a bug, not a choice.
- Live smoke tests run by default; do not hide contract-critical behavior behind mocks.
- Parity tests (`tests/test_parity.py`) are the oracle: hit real OpenRouter, hit local facade, diff the wire shape.
- Keep changes small. Extend only when there is a concrete, tested need.
- `uv run pytest` must be fully green before committing a slice.
