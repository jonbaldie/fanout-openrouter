from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time


@dataclass(frozen=True)
class FanoutPolicy:
    virtual_model: str
    candidate_models: tuple[str, ...]
    synthesis_model: str
    fanout_count: int
    default_fallback_model: str
    created: int


class PolicyRegistry:
    def __init__(self, policies: dict[str, FanoutPolicy]) -> None:
        self._policies = policies

    @classmethod
    def from_file(cls, file_path: str) -> "PolicyRegistry":
        raw = json.loads(Path(file_path).read_text(encoding="utf-8"))
        entries = raw.get("policies") if isinstance(raw, dict) else None
        if not isinstance(entries, list) or not entries:
            raise ValueError("policy file must contain a non-empty 'policies' array")

        policies: dict[str, FanoutPolicy] = {}
        for entry in entries:
            policy = _parse_policy(entry)
            policies[policy.virtual_model] = policy
        return cls(policies)

    def get(self, model: str) -> FanoutPolicy | None:
        return self._policies.get(model)

    def list(self) -> list[FanoutPolicy]:
        return list(self._policies.values())


def _parse_policy(entry: object) -> FanoutPolicy:
    if not isinstance(entry, dict):
        raise ValueError("each policy entry must be an object")

    virtual_model = _require_str(entry, "virtual_model")
    candidate_models_raw = entry.get("candidate_models")
    if not isinstance(candidate_models_raw, list) or not candidate_models_raw:
        raise ValueError(
            f"policy {virtual_model} must define a non-empty candidate_models list"
        )
    candidate_models = tuple(
        _require_str_value(value, f"candidate_models[{index}]")
        for index, value in enumerate(candidate_models_raw)
    )

    fanout_count_raw = entry.get("fanout_count", len(candidate_models))
    if not isinstance(fanout_count_raw, int) or fanout_count_raw < 1:
        raise ValueError(
            f"policy {virtual_model} fanout_count must be a positive integer"
        )

    synthesis_model = _optional_str(entry.get("synthesis_model")) or candidate_models[0]
    default_fallback_model = (
        _optional_str(entry.get("default_fallback_model")) or candidate_models[0]
    )
    created_raw = entry.get("created", int(time.time()))
    if not isinstance(created_raw, int):
        raise ValueError(f"policy {virtual_model} created must be an integer")

    return FanoutPolicy(
        virtual_model=virtual_model,
        candidate_models=candidate_models,
        synthesis_model=synthesis_model,
        fanout_count=fanout_count_raw,
        default_fallback_model=default_fallback_model,
        created=created_raw,
    )


def _require_str(entry: dict[str, object], key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"policy field {key} must be a non-empty string")
    return value


def _require_str_value(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"policy field {field_name} must be a non-empty string")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("optional string policy fields must be non-empty strings")
    return value
