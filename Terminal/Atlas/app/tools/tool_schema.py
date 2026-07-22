from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ToolSchema:
    name: str
    description: str
    arguments: dict[str, str] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                self.description,
                tuple(sorted(self.arguments.items())),
            )
        )