"""
CodeMate AI - LLM Gateway Client (agent/llm_client.py)
======================================================

Manages communication with the Groq API (using Llama 3 models with function calling).
Converts internal tool definitions into OpenAI/Groq compatible JSON tool schemas,
manages message payloads, and handles API errors.

Note:
Full client round-trip with live network calls will be activated during the LLM Integration phase.
"""

from typing import Any, Dict, List, Optional
from config import Config, load_config
from tools.tool_registry import ToolRegistry


class LLMClientError(Exception):
    """Raised when communication with the LLM API fails."""
    pass


class GroqClient:
    """
    Client interface for Groq API with function/tool calling support.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or load_config(strict_api_key=False)
        self.model_name = self.config.model_name
        self._client = None

    def get_client(self):
        """
        Lazily initialize the Groq client instance.
        """
        if self._client is None:
            try:
                import groq
            except ImportError as exc:
                raise LLMClientError(
                    "The 'groq' package is not installed. Run: pip install -r requirements.txt"
                ) from exc

            self.config.validate_api_key()
            self._client = groq.Groq(api_key=self.config.groq_api_key)

        return self._client

    def format_tool_schema(self, tool_registry: ToolRegistry) -> List[Dict[str, Any]]:
        """
        Convert registered tools into the standard JSON function-calling schema
        expected by Groq and Llama 3 tool-calling models.
        """
        schemas = []
        for tool_info in tool_registry.list_tools():
            tool_def = tool_registry.get_tool(tool_info["name"])
            if not tool_def:
                continue

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

                properties[param_name] = {
                    "type": param_type,
                    "description": f"Parameter: {param_name}"
                }
                if param.default is inspect.Parameter.empty:
                    required.append(param_name)

            schemas.append({
                "type": "function",
                "function": {
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            })
        return schemas


import inspect
