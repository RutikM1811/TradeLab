from app.services.tool_router import ToolRouter


def test_returns_none_for_normal_message() -> None:
    router = ToolRouter()

    assert router.route("Hello Atlas") is None


def test_routes_system_information_request() -> None:
    router = ToolRouter()

    assert router.route("Show my system information") == "system_info"


def test_routes_cpu_request() -> None:
    router = ToolRouter()

    assert router.route("What CPU do I have?") == "system_info"


def test_routes_ram_request() -> None:
    router = ToolRouter()

    assert router.route("How much RAM do I have?") == "system_info"


def test_routes_hardware_specs_request() -> None:
    router = ToolRouter()

    assert router.route("Show my hardware specs") == "system_info"


def test_is_case_insensitive() -> None:
    router = ToolRouter()

    assert router.route("SHOW MY CPU") == "system_info"


def test_routes_memory_request() -> None:
    router = ToolRouter()

    assert router.route("How much memory is installed?") == "system_info"


def test_returns_none_for_general_question() -> None:
    router = ToolRouter()

    assert router.route("Tell me a joke") is None


def test_returns_none_for_empty_message() -> None:
    router = ToolRouter()

    assert router.route("") is None


def test_returns_none_for_whitespace_message() -> None:
    router = ToolRouter()

    assert router.route("     ") is None