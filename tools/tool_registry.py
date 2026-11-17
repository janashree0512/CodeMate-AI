"""
CodeMate AI - Central Tool Registry Module (tools/tool_registry.py)
===================================================================

Centralized registry for managing, validating, inspecting, and dispatching
all agent tools.

Key Responsibilities:
- Central Registration: Store tool functions, metadata, and signatures.
- Groq Schema Generation: Automatically convert tools into standard OpenAI/Groq
  tool-calling JSON schemas with property types and required fields.
- Argument Validation: Inspect incoming arguments against function signatures,
  detect missing required parameters, and catch type mismatches before invocation.
- Safe Dispatching: Catch runtime errors and return structured response dictionaries.
"""

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from config import Config
from tools.file_tools import create_file, delete_file, list_files, read_file
from tools.execution_tool import execute_python


@dataclass
class ToolDefinition:
    """Stores metadata, signature, and callable handler for a registered tool."""
    name: str
    description: str
    func: Callable[..., Dict[str, Any]]
    signature: inspect.Signature


class ToolRegistry:
    """
    Central repository for registering, validating, introspecting, and dispatching agent tools.
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, name: str, description: str, func: Callable[..., Dict[str, Any]]) -> None:
        """
        Register a tool callable with a unique name and description.

        Args:
            name: Unique identifier for the tool.
            description: Human-readable purpose of the tool (exposed to the LLM).
            func: Python callable executing the tool logic.
        """
        if not name or not name.strip():
            raise ValueError("Tool name cannot be empty.")
        if not callable(func):
            raise TypeError(f"Tool handler for '{name}' must be callable.")

        clean_name = name.strip()
        sig = inspect.signature(func)
        self._tools[clean_name] = ToolDefinition(
            name=clean_name,
            description=description.strip(),
            func=func,
            signature=sig
        )

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Retrieve tool metadata by name, or None if not found."""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return a summary of all registered tools and their accepted parameters."""
        tool_list = []
        for name, tool_def in self._tools.items():
            params = [
                param_name for param_name in tool_def.signature.parameters.keys()
                if param_name != "config"
            ]
            tool_list.append({
                "name": name,
                "description": tool_def.description,
                "parameters": params
            })
        return tool_list

    def get_groq_schemas(self) -> List[Dict[str, Any]]:
        """
        Generate standard JSON function-calling schemas compatible with Groq / Llama 3 models.
        """
        schemas: List[Dict[str, Any]] = []
        for name, tool_def in self._tools.items():
            properties: Dict[str, Any] = {}
            required: List[str] = []

            for param_name, param in tool_def.signature.parameters.items():
                if param_name in {"config"}:
                    continue

                param_type = "string"
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == float:
                    param_type = "number"

                properties[param_name] = {
                    "type": param_type,
                    "description": f"Parameter: {param_name}"
                }
                if param.default is inspect.Parameter.empty:
                    required.append(param_name)

            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool_def.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            })
        return schemas

    def validate_arguments(self, tool_def: ToolDefinition, kwargs: Dict[str, Any]) -> Optional[str]:
        """
        Validate incoming arguments against the tool's signature.
        Returns an error string if validation fails, or None if valid.
        """
        sig = tool_def.signature
        # Check for missing required arguments
        for param_name, param in sig.parameters.items():
            if param_name == "config":
                continue
            if param.default is inspect.Parameter.empty and param_name not in kwargs:
                return f"Missing required argument '{param_name}' for tool '{tool_def.name}'."

        # Check for unexpected arguments
        accepted_params = set(sig.parameters.keys())
        for arg_name in kwargs.keys():
            if arg_name not in accepted_params:
                return f"Unexpected argument '{arg_name}' passed to tool '{tool_def.name}'."

        return None

    def dispatch(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Safely validate and execute a tool by name with keyword arguments.

        Returns:
            Structured dictionary with execution result or explicit error message.
        """
        tool_def = self.get_tool(tool_name)
        if not tool_def:
            available = list(self._tools.keys())
            return {
                "success": False,
                "tool_name": tool_name,
                "error": f"Tool '{tool_name}' not found. Available tools: {available}"
            }

        # Validate arguments before execution
        val_error = self.validate_arguments(tool_def, kwargs)
        if val_error:
            return {
                "success": False,
                "tool_name": tool_name,
                "error": f"Invalid arguments supplied for tool '{tool_name}': {val_error}"
            }

        try:
            # Execute the tool callable
            result = tool_def.func(**kwargs)
            return result
        except TypeError as err:
            return {
                "success": False,
                "tool_name": tool_name,
                "error": f"Argument type error in tool '{tool_name}': {err}"
            }
        except Exception as err:
            return {
                "success": False,
                "tool_name": tool_name,
                "error": f"Unexpected error while executing tool '{tool_name}': {err}"
            }


def create_default_registry(config: Optional[Config] = None) -> ToolRegistry:
    """
    Factory function initializing the default registry pre-loaded with
    CodeMate AI's 4 core tools:
    1. create_file
    2. read_file
    3. list_files
    4. execute_python
    (plus delete_file for workspace management).
    """
    registry = ToolRegistry()

    # Bind config to tools if provided
    def _create_file(filename: str, content: str, overwrite: bool = False, approved: bool = False) -> Dict[str, Any]:
        return create_file(filename=filename, content=content, overwrite=overwrite, approved=approved, config=config)

    def _read_file(filename: str) -> Dict[str, Any]:
        return read_file(filename=filename, config=config)

    def _list_files(directory: str = "") -> Dict[str, Any]:
        return list_files(directory=directory, config=config)

    def _delete_file(filename: str, approved: bool = False) -> Dict[str, Any]:
        return delete_file(filename=filename, approved=approved, config=config)

    def _execute_python(filename: str, approved: bool = False, timeout: Optional[int] = None) -> Dict[str, Any]:
        return execute_python(filename=filename, approved=approved, timeout=timeout, config=config)

    registry.register(
        name="create_file",
        description="Create a new text file inside the sandboxed workspace. Requires approved=True if overwrite=True on an existing file.",
        func=_create_file
    )
    registry.register(
        name="read_file",
        description="Read the text content of a file located within the sandboxed workspace.",
        func=_read_file
    )
    registry.register(
        name="list_files",
        description="List files and subfolders located inside the sandboxed workspace.",
        func=_list_files
    )
    registry.register(
        name="delete_file",
        description="Delete a file from the sandboxed workspace. Mandatory approval required (approved=True).",
        func=_delete_file
    )
    registry.register(
        name="execute_python",
        description="Execute a Python script located within the sandboxed workspace with strict timeout and sanitized environment. Mandatory approval required (approved=True).",
        func=_execute_python
    )

    return registry
