"""
Atlas Model Manager.

Executes registered AI models and publishes model lifecycle events.
"""

from typing import Any

from app.events.event_bus import EventBus
from app.models.model_registry import ModelRegistry
from app.types.model_result import ModelResult


class ModelManager:
    """Generates responses through registered Atlas AI models."""

    def __init__(
            self,
            model_registry: ModelRegistry,
            event_bus: EventBus,
    ) -> None:
        self._model_registry = model_registry
        self._event_bus = event_bus

    async def generate(
            self,
            model_name: str,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        """Generate a response using a registered model."""

        model = self._model_registry.get(model_name)

        self._event_bus.publish(
            "model.started",
            {
                "model_name": model.name,
                "provider": model.provider,
            },
        )

        try:
            result = await model.generate(
                prompt,
                **kwargs,
            )
        except Exception as exc:
            self._event_bus.publish(
                "model.failed",
                {
                    "model_name": model.name,
                    "provider": model.provider,
                    "error": str(exc),
                },
            )

            return ModelResult.fail(error=str(exc))

        event_name = (
            "model.completed"
            if result.success
            else "model.failed"
        )

        self._event_bus.publish(
            event_name,
            {
                "model_name": model.name,
                "provider": model.provider,
                "success": result.success,
            },
        )

        return result