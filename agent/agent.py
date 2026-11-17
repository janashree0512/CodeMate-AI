"""
CodeMate AI - Autonomous Agent Controller (agent/agent.py)
==========================================================

Orchestrates the ReAct (Reason + Act + Observe) execution loop.
Coordinates between user input, LLM reasoning, tool invocation via ToolRegistry,
and structured diagnostic feedback for error self-correction.

Termination Conditions:
1. Success: Exit code == 0 and verification passes.
2. Max Iterations Exceeded: Halts runaway execution loops.
3. User Rejection: Execution or destructive operation rejected by human operator.
4. Unrecoverable Error: API or system failure.
"""

from typing import Any, Dict, List, Optional
from config import Config, load_config
from tools.tool_registry import ToolRegistry, create_default_registry
from agent.prompts import SYSTEM_PROMPT


class AgentExecutionState:
    """Tracks state and telemetry across iterations of a task."""
    def __init__(self, task: str, max_iterations: int):
        self.task = task
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.completed = False
        self.success = False
        self.error_count = 0
        self.action_logs: List[Dict[str, Any]] = []

    def log_action(self, action_type: str, details: Dict[str, Any]) -> None:
        """Record an action summary for the user interface."""
        self.action_logs.append({
            "iteration": self.current_iteration,
            "action": action_type,
            "details": details
        })


class CodeMateAgent:
    """
    Main Autonomous Agent Controller.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        registry: Optional[ToolRegistry] = None
    ):
        self.config = config or load_config(strict_api_key=False)
        self.registry = registry or create_default_registry(config=self.config)
        self.messages: List[Dict[str, Any]] = []
        self.state: Optional[AgentExecutionState] = None

    def initialize_session(self, user_task: str) -> None:
        """
        Initialize the message history and execution state for a new programming task.
        """
        self.state = AgentExecutionState(
            task=user_task,
            max_iterations=self.config.max_iterations
        )
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_task}
        ]

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Return the conversation message thread."""
        return list(self.messages)

    def execute_step(self) -> Dict[str, Any]:
        """
        Execute a single step in the ReAct cycle.
        (Full loop orchestration with LLM tool calling is completed in the Agent Loop phase).
        """
        if self.state is None:
            raise RuntimeError("Agent session must be initialized before executing steps.")

        if self.state.current_iteration >= self.state.max_iterations:
            return {
                "terminated": True,
                "reason": "max_iterations_reached",
                "message": f"Execution halted: Reached limit of {self.state.max_iterations} iterations."
            }

        self.state.current_iteration += 1
        return {
            "terminated": False,
            "iteration": self.state.current_iteration,
            "status": "ready_for_llm_dispatch"
        }
