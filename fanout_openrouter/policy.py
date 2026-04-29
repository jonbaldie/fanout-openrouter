from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FanoutPolicy:
    virtual_model: str
    candidate_models: tuple[str, ...]
    synthesis_model: str
    fanout_count: int
    default_fallback_model: str


_POLICIES: dict[str, FanoutPolicy] = {
    "fanout/minimal": FanoutPolicy(
        virtual_model="fanout/minimal",
        candidate_models=(
            "anthropic/claude-sonnet-4",
            "anthropic/claude-sonnet-4",
        ),
        synthesis_model="anthropic/claude-sonnet-4",
        fanout_count=2,
        default_fallback_model="anthropic/claude-sonnet-4",
    )
}


def get_policy(model: str) -> FanoutPolicy | None:
    return _POLICIES.get(model)
