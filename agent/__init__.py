"""
Agent package for CodeMate AI.
Contains LLM client interface, prompt templates, diagnostics engine, and autonomous agent controller.
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
from agent.diagnostics import DiagnosticReport, parse_execution_error
from agent.agent import CodeMateAgent, run_agent, AgentResult, ToolExecutionRecord

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
    "DiagnosticReport",
    "parse_execution_error",
    "CodeMateAgent",
    "run_agent",
    "AgentResult",
    "ToolExecutionRecord",
]
