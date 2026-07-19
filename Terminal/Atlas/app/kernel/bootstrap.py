"""
Atlas Kernel Bootstrap.

Responsible for starting and coordinating the Atlas framework.
"""

from typing import Any
from uuid import UUID

from app.config.settings import get_settings
from app.core.logger import configure_logger, logger
from app.events.event_bus import EventBus
from app.kernel.container import Container
from app.kernel.registry import Registry
from app.memory.conversation import Conversation
from app.memory.conversation_manager import ConversationManager
from app.models.atlas.atlas_model import AtlasModel
from app.models.atlas.development_backend import DevelopmentBackend
from app.models.atlas.fallback_backend import FallbackBackend
from app.models.atlas.groq_backend import GroqBackend
from app.models.atlas.openrouter_backend import OpenRouterBackend
from app.models.echo_model import EchoModel
from app.models.model_manager import ModelManager
from app.models.model_registry import ModelRegistry
from app.services.chat_runtime import ChatRuntime
from app.storage.local.json_conversation_storage import (
    JsonConversationStorage,
)
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

        logger.success(
            "Kernel initialized successfully."
        )

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

        if self.settings.ATLAS_BACKEND == "openrouter":
            atlas_backend = OpenRouterBackend(
                api_key=self.settings.OPENROUTER_API_KEY,
                model=self.settings.OPENROUTER_MODEL,
                base_url=self.settings.OPENROUTER_BASE_URL,
            )

            logger.info(
                "Using OpenRouter inference backend: {}",
                self.settings.OPENROUTER_MODEL,
            )

        elif self.settings.ATLAS_BACKEND == "groq":
            atlas_backend = GroqBackend(
                api_key=self.settings.GROQ_API_KEY,
                model=self.settings.GROQ_MODEL,
                base_url=self.settings.GROQ_BASE_URL,
            )

            logger.info(
                "Using Groq inference backend: {}",
                self.settings.GROQ_MODEL,
            )

        elif self.settings.ATLAS_BACKEND == "fallback":
            atlas_backend = FallbackBackend(
                [
                    GroqBackend(
                        api_key=self.settings.GROQ_API_KEY,
                        model=self.settings.GROQ_MODEL,
                        base_url=self.settings.GROQ_BASE_URL,
                    ),
                    OpenRouterBackend(
                        api_key=self.settings.OPENROUTER_API_KEY,
                        model=self.settings.OPENROUTER_MODEL,
                        base_url=self.settings.OPENROUTER_BASE_URL,
                    ),
                    DevelopmentBackend(),
                ]
            )

            logger.info(
                "Using fallback inference backend "
                "(Groq -> OpenRouter -> Development)."
            )

        elif self.settings.ATLAS_BACKEND == "development":
            atlas_backend = DevelopmentBackend()

            logger.info(
                "Using development inference backend."
            )

        else:
            raise ValueError(
                "Unsupported Atlas backend: "
                f"{self.settings.ATLAS_BACKEND}"
            )

        atlas_model = AtlasModel(
            backend=atlas_backend,
        )

        model_registry.register(
            EchoModel()
        )
        model_registry.register(
            atlas_model
        )

        model_manager = ModelManager(
            model_registry=model_registry,
            event_bus=event_bus,
        )

        # Conversation persistence
        conversation_storage = JsonConversationStorage(
            directory=self.settings.CONVERSATION_STORAGE_PATH,
        )

        # Conversation runtime
        conversation_manager = ConversationManager(
            storage=conversation_storage,
        )

        chat_runtime = ChatRuntime(
            model=atlas_model,
            conversation_manager=conversation_manager,
        )

        # Register shared services
        self.container.register(
            EventBus,
            event_bus,
        )
        self.container.register(
            Registry,
            registry,
        )

        # Register tool runtime
        self.container.register(
            ToolRegistry,
            tool_registry,
        )
        self.container.register(
            ToolExecutor,
            tool_executor,
        )

        # Register model runtime
        self.container.register(
            ModelRegistry,
            model_registry,
        )
        self.container.register(
            ModelManager,
            model_manager,
        )
        self.container.register(
            AtlasModel,
            atlas_model,
        )

        # Register conversation persistence
        self.container.register(
            JsonConversationStorage,
            conversation_storage,
        )

        # Register conversation runtime
        self.container.register(
            ConversationManager,
            conversation_manager,
        )
        self.container.register(
            ChatRuntime,
            chat_runtime,
        )

        # Register built-in Atlas tools
        tool_registry.register(
            SystemInfoTool()
        )

        logger.debug(
            "Core services registered successfully."
        )

    async def execute_tool(
            self,
            tool_name: str,
            **kwargs: Any,
    ) -> ToolResult:
        """Execute a registered Atlas tool."""

        tool_executor = self.container.resolve(
            ToolExecutor
        )

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

        model_manager = self.container.resolve(
            ModelManager
        )

        return await model_manager.generate(
            model_name,
            prompt,
            **kwargs,
        )

    def create_conversation(
            self,
    ) -> tuple[UUID, Conversation]:
        """Create a new managed Atlas conversation."""

        chat_runtime = self.container.resolve(
            ChatRuntime
        )

        return chat_runtime.create_conversation()

    async def chat(
            self,
            conversation_id: UUID,
            message: str,
            **kwargs: Any,
    ) -> ModelResult:
        """Send a message to a managed Atlas conversation."""

        chat_runtime = self.container.resolve(
            ChatRuntime
        )

        return await chat_runtime.send_to(
            conversation_id,
            message,
            **kwargs,
        )

    def get_conversation(
            self,
            conversation_id: UUID,
    ) -> Conversation:
        """Return a managed conversation by ID."""

        conversation_manager = self.container.resolve(
            ConversationManager
        )

        return conversation_manager.get(
            conversation_id
        )

    def list_conversations(
            self,
    ) -> tuple[tuple[UUID, Conversation], ...]:
        """Return all managed conversations."""

        conversation_manager = self.container.resolve(
            ConversationManager
        )

        return conversation_manager.all()

    def delete_conversation(
            self,
            conversation_id: UUID,
    ) -> None:
        """Delete a managed conversation."""

        conversation_manager = self.container.resolve(
            ConversationManager
        )

        conversation_manager.delete(
            conversation_id
        )

    def clear_conversations(self) -> None:
        """Delete all managed conversations."""

        conversation_manager = self.container.resolve(
            ConversationManager
        )

        conversation_manager.clear()

    def conversation_count(self) -> int:
        """Return the number of managed conversations."""

        conversation_manager = self.container.resolve(
            ConversationManager
        )

        return len(conversation_manager)

    def save_conversation(
            self,
            conversation_id: UUID,
    ) -> None:
        """Persist a managed conversation."""

        conversation_manager = self.container.resolve(
            ConversationManager
        )

        conversation_manager.save(
            conversation_id
        )

    def restore_conversations(self) -> int:
        """Restore all persisted conversations into memory."""

        conversation_manager = self.container.resolve(
            ConversationManager
        )

        return conversation_manager.restore_all()