"""
Agent package for CodeMate AI.
Contains LLM client interface, prompt templates, and the autonomous agent controller.
"""

from agent.prompts import SYSTEM_PROMPT, TASK_PLANNING_TEMPLATE, ERROR_CORRECTION_PROMPT

__all__ = [
    "SYSTEM_PROMPT",
    "TASK_PLANNING_TEMPLATE",
    "ERROR_CORRECTION_PROMPT",
]
