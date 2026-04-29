# List all available commands
default:
    @just --list

# Install/update dependencies
sync:
    uv sync --all-extras

# Format all code
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# Run the linter
lint:
    uv run ruff check .

# Run the test suite
test *args:
    uv run pytest {{args}}

# Run the development server
dev:
    uv run uvicorn fanout_openrouter.app:app --reload

# Find available work via beads
ready:
    bd ready

# Safely push all beads state and git state
push:
    git pull --rebase
    bd dolt push
    git push
    git status
