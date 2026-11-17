"""
CodeMate AI - Groq LLM Gateway Client (agent/llm_client.py)
===========================================================

Production-grade client for Groq API using Llama 3 models with function/tool calling.
Supports multi-turn conversation history, robust error and rate limit handling,
structured tool-call schema formatting, and secure credential handling.

Key Features:
- Secure: Never logs or exposes GROQ_API_KEY.
- Resilient: Explicitly handles AuthenticationError, RateLimitError, APIConnectionError.
- Observable: Structured logging for token usage, latency, and tool calls.
- Flexible: Configurable model (defaults to llama-3.3-70b-versatile).
"""

import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import groq
from groq import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    Groq,
    GroqError,
    RateLimitError,
)

from config import Config, ConfigurationError, load_config
from tools.tool_registry import ToolRegistry

# Configure module-level logger
logger = logging.getLogger("CodeMateAI.LLMClient")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [LLMClient]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class LLMClientError(Exception):
    """Base exception for LLM client failures."""
    pass


class LLMAuthenticationError(LLMClientError):
    """Raised when the Groq API key is invalid or unauthorized."""
    pass


class LLMRateLimitError(LLMClientError):
    """Raised when Groq API rate limits (TPM / RPM) are exceeded."""
    pass


class LLMConnectionError(LLMClientError):
    """Raised when network connectivity to Groq fails."""
    pass


@dataclass
class ToolCallRequest:
    """Represents a structured tool invocation requested by the model."""
    id: str
    tool_name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    """Normalized response container from Groq LLM completions."""
    content: Optional[str]
    tool_calls: List[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def has_tool_calls(self) -> bool:
        """True if the model chose to invoke one or more tools."""
        return len(self.tool_calls) > 0


class GroqClient:
    """
    Reusable Groq API Client for CodeMate AI.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or load_config(strict_api_key=False)
        self.model_name = self.config.model_name
        self._client: Optional[Groq] = None

    def get_client(self) -> Groq:
        """
        Lazily initializes the Groq client instance.
        Validates API key existence before connecting.
        """
        if self._client is None:
            # Check and validate API key
            try:
                self.config.validate_api_key()
            except ConfigurationError as exc:
                logger.error("Groq API key validation failed.")
                raise LLMAuthenticationError(str(exc)) from exc

            # Initialize Groq SDK client
            self._client = Groq(api_key=self.config.groq_api_key)
            logger.info("Groq client initialized with model: %s", self.model_name)

        return self._client

    def format_tool_schema(self, tool_registry: ToolRegistry) -> List[Dict[str, Any]]:
        """
        Convert registered tools into standard JSON schemas for Groq / Llama 3 tool-calling.
        """
        schemas: List[Dict[str, Any]] = []
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

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Send a conversation thread (with optional tools) to Groq.

        Args:
            messages: List of message dictionaries [{'role': 'user', 'content': '...'}, ...]
            tools: Optional JSON schema list for tool calling.
            temperature: Sampling temperature (0.0 to 1.0; low value preferred for deterministic code).
            max_tokens: Maximum completion tokens to generate.

        Returns:
            LLMResponse: Structured response with parsed content and tool calls.
        """
        client = self.get_client()

        # Build payload parameters
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        logger.info(
            "Sending chat request to Groq (%s) with %d message(s) and %d tool(s).",
            self.model_name,
            len(messages),
            len(tools) if tools else 0
        )

        try:
            raw_response = client.chat.completions.create(**payload)
        except AuthenticationError as exc:
            logger.error("Groq Authentication Error: Invalid or expired API key.")
            raise LLMAuthenticationError(
                "Invalid Groq API Key. Please verify GROQ_API_KEY in your .env file."
            ) from exc
        except RateLimitError as exc:
            logger.error("Groq Rate Limit Exceeded: %s", exc)
            raise LLMRateLimitError(
                "Groq API rate limit reached. Please wait before retrying."
            ) from exc
        except APIConnectionError as exc:
            logger.error("Groq API Connection Failed: %s", exc)
            raise LLMConnectionError(
                "Failed to connect to Groq API. Please check your internet connection."
            ) from exc
        except (BadRequestError, APIStatusError, GroqError) as exc:
            logger.error("Groq API Error: %s", exc)
            raise LLMClientError(f"Groq API error occurred: {exc}") from exc

        # Parse assistant choice
        choice = raw_response.choices[0]
        assistant_message = choice.message
        finish_reason = choice.finish_reason or "stop"

        # Parse tool calls if present
        parsed_tool_calls: List[ToolCallRequest] = []
        if assistant_message.tool_calls:
            for tc in assistant_message.tool_calls:
                call_id = tc.id
                func_name = tc.function.name
                raw_args = tc.function.arguments

                try:
                    arguments_dict = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    logger.warning("Failed to parse tool call arguments as JSON: %s", raw_args)
                    arguments_dict = {"raw_arguments": raw_args}

                parsed_tool_calls.append(
                    ToolCallRequest(
                        id=call_id,
                        tool_name=func_name,
                        arguments=arguments_dict
                    )
                )

        # Extract token usage metrics
        usage = raw_response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        logger.info(
            "Response received. Tokens: prompt=%d, completion=%d, total=%d. Finish: %s. Tool calls: %d",
            prompt_tokens,
            completion_tokens,
            total_tokens,
            finish_reason,
            len(parsed_tool_calls)
        )

        return LLMResponse(
            content=assistant_message.content,
            tool_calls=parsed_tool_calls,
            finish_reason=finish_reason,
            model=raw_response.model or self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )

    def ask(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.2) -> str:
        """
        Convenience method to query the LLM with a single prompt and return text.

        Args:
            prompt: User question or instruction.
            system_prompt: Optional system role instruction.
            temperature: Sampling temperature.

        Returns:
            str: Model response text.
        """
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.chat(messages=messages, tools=None, temperature=temperature)
        return response.content or ""
