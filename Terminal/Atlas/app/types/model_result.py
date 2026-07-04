"""
Standard result returned by Atlas AI models.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelResult:
    """Represents the result of an AI model generation."""

    success: bool
    content: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(
            cls,
            content: str,
            metadata: dict[str, Any] | None = None,
    ) -> "ModelResult":
        """Create a successful model result."""

        return cls(
            success=True,
            content=content,
            metadata=metadata or {},
        )

    @classmethod
    def fail(
            cls,
            error: str,
            metadata: dict[str, Any] | None = None,
    ) -> "ModelResult":
        """Create a failed model result."""

        return cls(
            success=False,
            error=error,
            metadata=metadata or {},
        )