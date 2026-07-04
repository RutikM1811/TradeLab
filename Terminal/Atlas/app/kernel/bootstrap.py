"""
Atlas Kernel Bootstrap.

Responsible for starting and coordinating the Atlas framework.
"""

from typing import Any

from app.config.settings import get_settings
from app.core.logger import configure_logger, logger
from app.events.event_bus import EventBus
from app.kernel.container import Container
from app.kernel.registry import Registry
from app.models.echo_model import EchoModel
from app.models.model_manager import ModelManager
from app.models.model_registry import ModelRegistry
from app.tools.system_info_tool import SystemInfoTool
from app.tools.tool_executor import ToolExecutor
from app.tools.tool_registry import ToolRegistry
from app.types.model_result import ModelResult
from app.types.tool_result import ToolResult


class Kernel:
    """Atlas framework kernel."""

    def __init__(self) -> None:
        """Initialize the Atlas kernel."""

        self.settings = get_settings()
        self.container = Container()

    def boot(self) -> None:
        """Boot the Atlas framework."""

        configure_logger()

        logger.info("=" * 60)
        logger.info(f"Starting {self.settings.APP_NAME}")
        logger.info(f"Version : {self.settings.APP_VERSION}")
        logger.info(f"Debug   : {self.settings.DEBUG}")
        logger.info("=" * 60)

        self._register_core_services()

        event_bus = self.container.resolve(EventBus)

        event_bus.publish(
            "kernel.started",
            {
                "app_name": self.settings.APP_NAME,
                "version": self.settings.APP_VERSION,
            },
        )

        logger.success("Kernel initialized successfully.")

    def _register_core_services(self) -> None:
        """Register Atlas core framework services."""

        # Shared core services
        event_bus = EventBus()
        registry = Registry()

        # Tool runtime
        tool_registry = ToolRegistry()

        tool_executor = ToolExecutor(
            tool_registry=tool_registry,
            event_bus=event_bus,
        )

        # Model runtime
        model_registry = ModelRegistry()

        model_manager = ModelManager(
            model_registry=model_registry,
            event_bus=event_bus,
        )

        # Register built-in Atlas models
        model_registry.register(EchoModel())

        # Register shared services
        self.container.register(EventBus, event_bus)
        self.container.register(Registry, registry)

        # Register tool runtime
        self.container.register(ToolRegistry, tool_registry)
        self.container.register(ToolExecutor, tool_executor)

        # Register model runtime
        self.container.register(ModelRegistry, model_registry)
        self.container.register(ModelManager, model_manager)

        # Register built-in Atlas tools
        tool_registry.register(SystemInfoTool())

        logger.debug("Core services registered successfully.")

    async def execute_tool(
            self,
            tool_name: str,
            **kwargs: Any,
    ) -> ToolResult:
        """Execute a registered Atlas tool."""

        tool_executor = self.container.resolve(ToolExecutor)

        return await tool_executor.execute(
            tool_name,
            **kwargs,
        )

    async def generate(
            self,
            model_name: str,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        """Generate a response using a registered Atlas model."""

        model_manager = self.container.resolve(ModelManager)

        return await model_manager.generate(
            model_name,
            prompt,
            **kwargs,
        )