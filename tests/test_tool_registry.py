"""
Unit tests for the Tool Registry (tools/tool_registry.py).
Validates dynamic tool registration, introspection, keyword dispatching,
and structured error reporting for unknown tools or bad arguments.
"""

import pytest
from config import Config
from tools.tool_registry import ToolRegistry, create_default_registry


@pytest.fixture
def test_config(tmp_path):
    """Provides an isolated configuration pointing to a temporary workspace."""
    workspace = tmp_path / "registry_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return Config(
        groq_api_key="mock_key",
        model_name="llama-3.3-70b-versatile",
        max_iterations=10,
        execution_timeout_seconds=5,
        require_user_approval=True,
        workspace_dir=workspace,
        project_root=tmp_path,
        max_tool_output_chars=4000,
        max_history_messages=30
    )


class TestToolRegistry:
    """Tests for ToolRegistry registration and dispatch behavior."""

    def test_register_and_list_tools(self):
        registry = ToolRegistry()

        def custom_adder(a: int, b: int) -> dict:
            return {"success": True, "sum": a + b}

        registry.register(
            name="adder",
            description="Adds two integers together.",
            func=custom_adder
        )

        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "adder"
        assert tools[0]["description"] == "Adds two integers together."
        assert "a" in tools[0]["parameters"]
        assert "b" in tools[0]["parameters"]

    def test_dispatch_valid_tool(self):
        registry = ToolRegistry()

        def multiplier(x: int, y: int) -> dict:
            return {"success": True, "product": x * y}

        registry.register("multiply", "Multiplies two numbers", multiplier)

        result = registry.dispatch("multiply", x=6, y=7)
        assert result["success"] is True
        assert result["product"] == 42

    def test_dispatch_unknown_tool(self):
        registry = ToolRegistry()
        result = registry.dispatch("non_existent_tool", arg="val")

        assert result["success"] is False
        assert result["tool_name"] == "non_existent_tool"
        assert "not found" in result["error"]

    def test_dispatch_invalid_arguments(self):
        registry = ToolRegistry()

        def strict_tool(required_arg: str) -> dict:
            return {"success": True, "value": required_arg}

        registry.register("strict", "A strict tool", strict_tool)

        # Call missing required argument
        result_missing = registry.dispatch("strict")
        assert result_missing["success"] is False
        assert "Invalid arguments" in result_missing["error"]

        # Call with unexpected keyword argument
        result_unexpected = registry.dispatch("strict", required_arg="ok", extra_junk=123)
        assert result_unexpected["success"] is False
        assert "Invalid arguments" in result_unexpected["error"]

    def test_empty_tool_name_rejected(self):
        registry = ToolRegistry()
        with pytest.raises(ValueError):
            registry.register("", "Empty name", lambda: {})

    def test_non_callable_rejected(self):
        registry = ToolRegistry()
        with pytest.raises(TypeError):
            registry.register("bad_tool", "Not a func", "string_instead_of_callable")


class TestDefaultRegistryIntegration:
    """Tests verifying the pre-configured default tool registry."""

    def test_default_registry_contains_all_tools(self, test_config):
        registry = create_default_registry(config=test_config)
        tool_names = [t["name"] for t in registry.list_tools()]

        expected_tools = ["create_file", "read_file", "list_files", "delete_file", "execute_python"]
        for expected in expected_tools:
            assert expected in tool_names

    def test_dispatch_create_and_execute_through_default_registry(self, test_config):
        registry = create_default_registry(config=test_config)

        # 1. Create a script via registry dispatch
        create_res = registry.dispatch(
            "create_file",
            filename="main.py",
            content="print('Dispatched via Registry!')"
        )
        assert create_res["success"] is True

        # 2. Execute the script via registry dispatch
        exec_res = registry.dispatch(
            "execute_python",
            filename="main.py",
            approved=True
        )
        assert exec_res["success"] is True
        assert exec_res["exit_code"] == 0
        assert "Dispatched via Registry!" in exec_res["stdout"]

    def test_get_groq_schemas_format(self, test_config):
        registry = create_default_registry(config=test_config)
        schemas = registry.get_groq_schemas()

        assert isinstance(schemas, list)
        assert len(schemas) >= 5

        for schema in schemas:
            assert schema["type"] == "function"
            fn = schema["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn
            assert fn["parameters"]["type"] == "object"
            assert "properties" in fn["parameters"]
            assert isinstance(fn["parameters"]["required"], list)

        # Check create_file schema specifically
        create_schema = next(s for s in schemas if s["function"]["name"] == "create_file")
        props = create_schema["function"]["parameters"]["properties"]
        assert "filename" in props
        assert props["filename"]["type"] == "string"
        assert "content" in props
        assert props["content"]["type"] == "string"
        assert "filename" in create_schema["function"]["parameters"]["required"]
        assert "content" in create_schema["function"]["parameters"]["required"]
