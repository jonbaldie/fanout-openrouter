#!/bin/bash
docker run --rm -v $(pwd):/workspace -w /workspace -e OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" ubuntu:latest /bin/bash -c "
  set -ex
  apt-get update && apt-get install -y curl git
  
  # Install uv (standard installation script defaults to \$HOME/.local/bin)
  curl -LsSf https://astral.sh/uv/install.sh | sh
  
  export PATH=\"\$HOME/.local/bin:\$PATH\"
  
  # Run the CI steps
  uv sync --all-extras
  uv run ruff check .
  uv run ruff format --check .
  uv run pytest
"
