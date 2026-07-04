from app.kernel.registry import Registry


class DummyTool:
    pass


def test_registry_registers_and_gets_component() -> None:
    registry = Registry()
    tool = DummyTool()

    registry.register("dummy_tool", tool)

    resolved_tool = registry.get("dummy_tool")

    assert resolved_tool is tool
    assert registry.contains("dummy_tool")


def test_registry_rejects_duplicate_name() -> None:
    registry = Registry()

    registry.register("dummy_tool", DummyTool())

    try:
        registry.register("dummy_tool", DummyTool())
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_registry_rejects_empty_name() -> None:
    registry = Registry()

    try:
        registry.register("   ", DummyTool())
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_registry_raises_for_missing_component() -> None:
    registry = Registry()

    try:
        registry.get("missing_tool")
        assert False, "Expected KeyError"
    except KeyError:
        pass


def test_registry_returns_copy_of_components() -> None:
    registry = Registry()
    tool = DummyTool()

    registry.register("dummy_tool", tool)

    components = registry.all()
    components.clear()

    assert registry.contains("dummy_tool")