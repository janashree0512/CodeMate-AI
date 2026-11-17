"""
CodeMate AI - Tool Registry Module
==================================

Provides a centralized registry pattern to manage, inspect, and dispatch
available tools to the agent.

Responsibilities:
- Register tool functions along with their human-readable descriptions.
- Store tool metadata and function callables.
- Safely dispatch tool calls by name and keyword arguments.
- Handle unknown tools and invalid arguments gracefully.
- Return structured error dictionaries on failure.
"""

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from config import Config
from tools.file_tools import create_file, delete_file, list_files, read_file
from tools.code_executor import execute_python


@dataclass
class ToolDefinition:
    """Stores metadata and execution handler for a single registered tool."""
    name: str
    description: str
    func: Callable[..., Dict[str, Any]]
    signature: inspect.Signature


class ToolRegistry:
    """
    Central repository for registering, introspecting, and dispatching agent tools.
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, name: str, description: str, func: Callable[..., Dict[str, Any]]) -> None:
        """
        Register a new tool callable with a unique name and description.
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

    def dispatch(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Dispatch execution to a registered tool by name with keyword arguments.

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

        try:
            # Execute the tool function
            result = tool_def.func(**kwargs)
            return result
        except TypeError as err:
            return {
                "success": False,
                "tool_name": tool_name,
                "error": f"Invalid arguments supplied for tool '{tool_name}': {err}"
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
    CodeMate AI's standard file management and code execution tools.
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
        description="Create a new file in the sandboxed workspace. Requires approved=True if overwrite=True on an existing file.",
        func=_create_file
    )
    registry.register(
        name="read_file",
        description="Read the text content of a file located within the sandboxed workspace.",
        func=_read_file
    )
    registry.register(
        name="list_files",
        description="List files and directories located inside the sandboxed workspace.",
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
