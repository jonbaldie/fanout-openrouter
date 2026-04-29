"""
pytest configuration shared across the test suite.

Live tests (marked with ``@pytest.mark.live``) hit the real OpenRouter API.
They are skipped automatically when the ``CI`` environment variable is set to
``"true"`` (which GitHub Actions sets on every runner) so that the public CI
workflow never incurs API costs or requires secrets.

To run *only* live tests locally::

    uv run pytest -m live

To explicitly exclude them (mirrors CI behaviour)::

    uv run pytest -m "not live"

The default (no ``-m`` flag) runs everything, which is the intended local
developer experience.
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if os.environ.get("CI") != "true":
        return

    skip_live = pytest.mark.skip(
        reason=(
            "Live API tests are skipped in CI (CI=true) to avoid OpenRouter costs. "
            "Run locally with: uv run pytest -m live"
        )
    )
    for item in items:
        if item.get_closest_marker("live") is not None:
            item.add_marker(skip_live)
