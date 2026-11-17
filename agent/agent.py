"""
CodeMate AI - Core Autonomous Agent Controller (agent/agent.py)
==============================================================

Implements the ReAct (Reason + Act + Observe) autonomous execution loop.
Directly communicates with the Groq API using standard function/tool-calling schemas,
dispatches tool executions through the Python tool registry, captures runtime feedback,
analyzes tracebacks with the diagnostics engine, and autonomously attempts error self-correction.

Self-Correction Workflow:
1. Generate & save code to workspace/
2. Execute in restricted subprocess sandbox
3. Capture stdout, stderr, and exit_code
4. If failed, analyze via diagnostics engine (SyntaxError vs RuntimeError vs Timeout vs AssertionError)
5. Feed structured diagnostic feedback into conversation history
6. Model inspects diagnostics, patches file (approved=True, overwrite=True)
7. Re-execute and observe until verified success or max retries
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import Config, load_config
from agent.prompts import SYSTEM_PROMPT
from agent.llm_client import GroqClient, LLMResponse, ToolCallRequest
from agent.diagnostics import parse_execution_error, DiagnosticReport
from tools.tool_registry import ToolRegistry, create_default_registry

# Configure module-level logger
logger = logging.getLogger("CodeMateAI.Agent")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [AgentController]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class ToolExecutionRecord:
    """Telemetry record for a single tool execution action."""
    iteration: int
    tool_name: str
    tool_id: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    success: bool


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


@dataclass
class AgentResult:
    """Final output and execution report returned by the agent controller."""
    task: str
    success: bool
    final_response: str
    iterations: int
    max_iterations: int
    termination_reason: str
    correction_attempts: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    action_history: List[ToolExecutionRecord] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)


class CodeMateAgent:
    """
    Autonomous Coding Agent Controller implementing the ReAct loop
    with automated error detection, diagnosis, and self-correction.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        registry: Optional[ToolRegistry] = None,
        llm_client: Optional[GroqClient] = None,
        auto_approve: bool = True
    ):
        """
        Initialize the CodeMate AI agent.

        Args:
            config: Application configuration instance.
            registry: Tool registry with available functions.
            llm_client: Groq API client instance.
            auto_approve: If True, automatically approves safe workspace file operations
                          and execution during autonomous runs.
        """
        self.config = config or load_config(strict_api_key=False)
        self.registry = registry or create_default_registry(config=self.config)
        self.llm_client = llm_client or GroqClient(config=self.config)
        self.auto_approve = auto_approve
        self.messages: List[Dict[str, Any]] = []
        self.state: Optional[AgentExecutionState] = None

    def initialize_session(self, user_task: str) -> None:
        """
        Initialize message history and execution state for a new programming task.
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
        """Return the current conversation message thread."""
        return list(self.messages)

    def execute_step(self) -> Dict[str, Any]:
        """
        Execute or step forward one iteration in the ReAct lifecycle.
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

    def _execute_tool_call(
        self,
        tool_call: ToolCallRequest,
        iteration: int
    ) -> ToolExecutionRecord:
        """
        Dispatch a single tool call through the registry.
        Applies diagnostics on execution failures and maintains error state.
        """
        args = dict(tool_call.arguments)

        # Inject approval flag if tool accepts it and auto_approve is active
        tool_def = self.registry.get_tool(tool_call.tool_name)
        if tool_def and "approved" in tool_def.signature.parameters:
            if "approved" not in args:
                args["approved"] = self.auto_approve

        logger.info(
            "[Iter %d] Dispatching tool: '%s' with arguments: %s",
            iteration,
            tool_call.tool_name,
            args
        )

        # Execute through Python tool registry
        result = self.registry.dispatch(tool_call.tool_name, **args)
        is_success = bool(result.get("success", False))

        # Perform error diagnostics for Python code execution failures
        if tool_call.tool_name == "execute_python" and not is_success:
            report: DiagnosticReport = parse_execution_error(
                stdout=result.get("stdout", ""),
                stderr=result.get("stderr", ""),
                exit_code=result.get("exit_code"),
                timed_out=result.get("timed_out", False)
            )
            result["diagnostics"] = report.to_dict()
            logger.warning(
                "[Iter %d] Execution failure diagnosed: Category='%s', Type='%s', Line=%s: %s",
                iteration,
                report.category,
                report.exception_name,
                report.failed_line,
                report.actionable_suggestion
            )
        elif is_success:
            logger.info("[Iter %d] Tool '%s' executed successfully.", iteration, tool_call.tool_name)
        else:
            logger.warning(
                "[Iter %d] Tool '%s' reported failure/error: %s",
                iteration,
                tool_call.tool_name,
                result.get("error", "Unknown error")
            )

        return ToolExecutionRecord(
            iteration=iteration,
            tool_name=tool_call.tool_name,
            tool_id=tool_call.id,
            arguments=args,
            result=result,
            success=is_success
        )

    def run(self, task: str, max_iterations: Optional[int] = None) -> AgentResult:
        """
        Run the autonomous agent loop with error diagnosis and self-correction.

        Args:
            task: Natural language programming instruction.
            max_iterations: Maximum ReAct cycles (defaults to config.max_iterations).

        Returns:
            AgentResult: Execution report with verified answer, attempts, and history.
        """
        limit = max_iterations if max_iterations is not None else self.config.max_iterations
        if limit <= 0:
            limit = 10

        logger.info("Starting CodeMate AI agent task: %s (Max iterations: %d)", task, limit)

        # 1. Initialize conversation history with system prompt and user task
        self.initialize_session(task)
        if self.state:
            self.state.max_iterations = limit

        action_history: List[ToolExecutionRecord] = []
        execution_history: List[Dict[str, Any]] = []
        correction_attempts = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        current_iteration = 0
        final_response_text = ""
        termination_reason = "max_iterations_reached"
        success = False

        # Format tool schemas for the LLM
        tool_schemas = self.llm_client.format_tool_schema(self.registry)

        # 2. Main Agent Execution Loop
        while current_iteration < limit:
            current_iteration += 1
            if self.state:
                self.state.current_iteration = current_iteration
            logger.info("=== Agent Loop Iteration %d / %d ===", current_iteration, limit)

            # Step A: Query LLM with current conversation history and available tools
            llm_response: LLMResponse = self.llm_client.chat(
                messages=self.messages,
                tools=tool_schemas,
                temperature=0.2
            )

            # Track token usage metrics
            total_prompt_tokens += llm_response.prompt_tokens
            total_completion_tokens += llm_response.completion_tokens
            total_tokens += llm_response.total_tokens

            # Step B: Check if model emitted any tool calls
            if llm_response.has_tool_calls:
                logger.info(
                    "[Iter %d] Model requested %d tool call(s).",
                    current_iteration,
                    len(llm_response.tool_calls)
                )

                # Append assistant message with tool calls to history
                assistant_tool_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": llm_response.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.tool_name,
                                "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else str(tc.arguments)
                            }
                        }
                        for tc in llm_response.tool_calls
                    ]
                }
                self.messages.append(assistant_tool_msg)

                # Step C: Execute each tool call and append 'tool' response messages
                for tool_call in llm_response.tool_calls:
                    record = self._execute_tool_call(tool_call, current_iteration)
                    action_history.append(record)
                    if self.state:
                        self.state.log_action(tool_call.tool_name, record.result)

                    # Maintain execution-specific history and correction counter
                    if tool_call.tool_name == "execute_python":
                        execution_history.append({
                            "iteration": current_iteration,
                            "filename": record.arguments.get("filename"),
                            "success": record.success,
                            "exit_code": record.result.get("exit_code"),
                            "stdout": record.result.get("stdout"),
                            "stderr": record.result.get("stderr"),
                            "diagnostics": record.result.get("diagnostics")
                        })
                        if not record.success:
                            correction_attempts += 1

                    # Provide tool output back to LLM context
                    tool_response_msg: Dict[str, Any] = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.tool_name,
                        "content": json.dumps(record.result)
                    }
                    self.messages.append(tool_response_msg)

                # Continue to next iteration so LLM can inspect tool output
                continue

            else:
                # Step D: No tool calls emitted -> Model completed its task
                logger.info("[Iter %d] Model completed task without additional tool calls.", current_iteration)
                final_response_text = llm_response.content or "Task completed."

                self.messages.append({
                    "role": "assistant",
                    "content": final_response_text
                })

                # Verify if any executions occurred and whether the last execution passed
                if execution_history:
                    last_exec = execution_history[-1]
                    success = bool(last_exec.get("success", False))
                    termination_reason = "completed" if success else "unresolved_execution_failure"
                else:
                    success = True
                    termination_reason = "completed"

                if self.state:
                    self.state.completed = True
                    self.state.success = success
                break

        if not success and termination_reason == "max_iterations_reached":
            logger.warning(
                "Agent stopped: Reached limit of %d iterations without full completion.",
                limit
            )
            final_response_text = (
                f"Agent reached the maximum iteration limit ({limit}) without concluding. "
                f"Correction attempts made: {correction_attempts}. "
                "Review the action history and tool execution logs for partial progress."
            )

        logger.info(
            "Task finished. Reason: %s. Iterations: %d. Correction Attempts: %d. Total tokens: %d",
            termination_reason,
            current_iteration,
            correction_attempts,
            total_tokens
        )

        return AgentResult(
            task=task,
            success=success,
            final_response=final_response_text,
            iterations=current_iteration,
            max_iterations=limit,
            termination_reason=termination_reason,
            correction_attempts=correction_attempts,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
            execution_history=execution_history,
            action_history=action_history,
            messages=self.messages
        )


def run_agent(
    task: str,
    config: Optional[Config] = None,
    registry: Optional[ToolRegistry] = None,
    llm_client: Optional[GroqClient] = None,
    max_iterations: Optional[int] = None,
    auto_approve: bool = True
) -> AgentResult:
    """
    Convenience function to initialize and run the CodeMate AI agent.
    """
    agent = CodeMateAgent(
        config=config,
        registry=registry,
        llm_client=llm_client,
        auto_approve=auto_approve
    )
    return agent.run(task=task, max_iterations=max_iterations)
