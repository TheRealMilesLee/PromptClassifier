from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TrainingExample:
    """One manually labelled task request used by the baseline model."""

    text: str
    work_type: str
    intent: str
    complexity: str
    effort: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrainingExample":
        required = ("text", "work_type", "intent", "complexity", "effort")
        missing = [key for key in required if not str(value.get(key, "")).strip()]
        if missing:
            raise ValueError(f"training example is missing fields: {', '.join(missing)}")

        return cls(
            text=str(value["text"]).strip(),
            work_type=str(value["work_type"]).strip(),
            intent=str(value["intent"]).strip(),
            complexity=str(value["complexity"]).strip(),
            effort=str(value["effort"]).strip(),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "text": self.text,
            "work_type": self.work_type,
            "intent": self.intent,
            "complexity": self.complexity,
            "effort": self.effort,
        }
