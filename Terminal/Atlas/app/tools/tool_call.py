from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                tuple(sorted(self.kwargs.items())),
            )
        )