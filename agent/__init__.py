"""
Agent package for CodeMate AI.
Contains LLM client interface, prompt templates, and the autonomous agent controller.
"""

from agent.prompts import SYSTEM_PROMPT, TASK_PLANNING_TEMPLATE, ERROR_CORRECTION_PROMPT
from agent.llm_client import (
    GroqClient,
    LLMResponse,
    ToolCallRequest,
    LLMClientError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMConnectionError,
)

__all__ = [
    "SYSTEM_PROMPT",
    "TASK_PLANNING_TEMPLATE",
    "ERROR_CORRECTION_PROMPT",
    "GroqClient",
    "LLMResponse",
    "ToolCallRequest",
    "LLMClientError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMConnectionError",
]
