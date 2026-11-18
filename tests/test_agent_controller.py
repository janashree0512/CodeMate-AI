"""
Unit tests for the Core AI Agent Controller (agent/agent.py).
Tests multi-turn ReAct loops, tool execution dispatching, tool failure handling,
and maximum iteration limits using a controlled mock LLM client with real tool execution.
"""

from unittest.mock import MagicMock
import pytest

from config import Config, DEFAULT_MODEL
from agent.agent import CodeMateAgent, run_agent, AgentResult
from agent.llm_client import LLMResponse, ToolCallRequest
from tools.tool_registry import create_default_registry


@pytest.fixture
def agent_config(tmp_path):
    """Provides an isolated configuration pointing to a temporary test workspace."""
    workspace = tmp_path / "agent_sandbox"
    workspace.mkdir(parents=True, exist_ok=True)
    return Config(
        groq_api_key="gsk_mock_valid_key",
        model_name=DEFAULT_MODEL,
        max_iterations=4,
        execution_timeout_seconds=5,
        require_user_approval=True,
        workspace_dir=workspace,
        project_root=tmp_path,
        max_tool_output_chars=4000,
        max_history_messages=30
    )


class TestAgentController:
    """Test suite for CodeMateAgent and run_agent."""

    def test_single_turn_direct_completion(self, agent_config):
        mock_llm = MagicMock()
        mock_llm.format_tool_schema.return_value = []
        mock_llm.chat.return_value = LLMResponse(
            content="Direct explanation without needing any tools.",
            tool_calls=[],
            finish_reason="stop",
            model=DEFAULT_MODEL,
            prompt_tokens=10,
            completion_tokens=8,
            total_tokens=18
        )

        agent = CodeMateAgent(config=agent_config, llm_client=mock_llm)
        result = agent.run("What is Python?")

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.iterations == 1
        assert result.termination_reason == "completed"
        assert result.final_response == "Direct explanation without needing any tools."
        assert len(result.action_history) == 0

    def test_multi_turn_create_file_and_complete(self, agent_config):
        registry = create_default_registry(config=agent_config)
        mock_llm = MagicMock()
        mock_llm.format_tool_schema.return_value = [{"type": "function"}]

        # Turn 1: LLM requests create_file tool
        turn_1_response = LLMResponse(
            content="I will create the script file.",
            tool_calls=[
                ToolCallRequest(
                    id="call_create_1",
                    tool_name="create_file",
                    arguments={"filename": "greet.py", "content": "print('Greetings from CodeMate!')"}
                )
            ],
            finish_reason="tool_calls",
            model=DEFAULT_MODEL
        )

        # Turn 2: LLM receives tool result and concludes
        turn_2_response = LLMResponse(
            content="Successfully created greet.py with greeting message.",
            tool_calls=[],
            finish_reason="stop",
            model=DEFAULT_MODEL
        )

        mock_llm.chat.side_effect = [turn_1_response, turn_2_response]

        agent = CodeMateAgent(config=agent_config, registry=registry, llm_client=mock_llm)
        result = agent.run("Create a greeting script")

        assert result.success is True
        assert result.iterations == 2
        assert result.termination_reason == "completed"
        assert len(result.action_history) == 1

        # Verify real file creation occurred on disk in test workspace
        created_file = agent_config.workspace_dir / "greet.py"
        assert created_file.exists()
        assert "Greetings from CodeMate!" in created_file.read_text()

    def test_create_and_execute_python_workflow(self, agent_config):
        registry = create_default_registry(config=agent_config)
        mock_llm = MagicMock()
        mock_llm.format_tool_schema.return_value = []

        # Turn 1: Create script
        resp_1 = LLMResponse(
            content="Writing math program.",
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    tool_name="create_file",
                    arguments={"filename": "calc.py", "content": "print(10 * 10)"}
                )
            ]
        )

        # Turn 2: Execute script
        resp_2 = LLMResponse(
            content="Running script to verify.",
            tool_calls=[
                ToolCallRequest(
                    id="call_2",
                    tool_name="execute_python",
                    arguments={"filename": "calc.py"}
                )
            ]
        )

        # Turn 3: Observe output 100 and finalize
        resp_3 = LLMResponse(
            content="The script was executed and output 100 as expected.",
            tool_calls=[]
        )

        mock_llm.chat.side_effect = [resp_1, resp_2, resp_3]

        agent = CodeMateAgent(config=agent_config, registry=registry, llm_client=mock_llm)
        result = agent.run("Write and run a script to multiply 10 by 10")

        assert result.success is True
        assert result.iterations == 3
        assert len(result.action_history) == 2

        # Check action 1: create_file
        assert result.action_history[0].tool_name == "create_file"
        assert result.action_history[0].success is True

        # Check action 2: execute_python
        assert result.action_history[1].tool_name == "execute_python"
        assert result.action_history[1].success is True
        assert "100" in result.action_history[1].result["stdout"]

    def test_tool_failure_recorded_and_passed_to_llm(self, agent_config):
        registry = create_default_registry(config=agent_config)
        mock_llm = MagicMock()
        mock_llm.format_tool_schema.return_value = []

        # Turn 1: Try to run a file that doesn't exist
        resp_1 = LLMResponse(
            content="Running missing file.",
            tool_calls=[
                ToolCallRequest(
                    id="call_fail",
                    tool_name="execute_python",
                    arguments={"filename": "does_not_exist.py"}
                )
            ]
        )

        # Turn 2: LLM acknowledges failure
        resp_2 = LLMResponse(
            content="The script does not exist, so I cannot run it.",
            tool_calls=[]
        )

        mock_llm.chat.side_effect = [resp_1, resp_2]

        agent = CodeMateAgent(config=agent_config, registry=registry, llm_client=mock_llm)
        result = agent.run("Run nonexistent script")

        assert result.success is False
        assert result.termination_reason == "unresolved_execution_failure"
        assert len(result.action_history) == 1
        assert result.action_history[0].success is False
        assert "Script not found" in result.action_history[0].result["error"]

    def test_max_iterations_ceiling_prevents_infinite_loop(self, agent_config):
        registry = create_default_registry(config=agent_config)
        mock_llm = MagicMock()
        mock_llm.format_tool_schema.return_value = []

        # LLM that stubbornly requests list_files on every single turn
        endless_tool_response = LLMResponse(
            content="Listing files again.",
            tool_calls=[
                ToolCallRequest(
                    id="call_loop",
                    tool_name="list_files",
                    arguments={}
                )
            ]
        )
        mock_llm.chat.return_value = endless_tool_response

        agent = CodeMateAgent(config=agent_config, registry=registry, llm_client=mock_llm)
        # Cap at 3 iterations
        result = agent.run("Endless task", max_iterations=3)

        assert result.success is False
        assert result.iterations == 3
        assert result.termination_reason == "max_iterations_reached"

    def test_agent_self_correction_cycle(self, agent_config):
        """
        Verify end-to-end self-correction:
        1. Agent writes buggy code (ZeroDivisionError).
        2. Execution fails; diagnostics attach structured report.
        3. Agent analyzes error, updates code with fix (overwrite=True, approved=True).
        4. Re-execution succeeds.
        5. Agent observes success and concludes.
        """
        registry = create_default_registry(config=agent_config)
        mock_llm = MagicMock()
        mock_llm.format_tool_schema.return_value = []

        # Turn 1: Write buggy code
        turn_1 = LLMResponse(
            content="Writing division script.",
            tool_calls=[
                ToolCallRequest(
                    id="call_c1",
                    tool_name="create_file",
                    arguments={"filename": "divide.py", "content": "res = 10 / 0\nprint('Result:', res)\n"}
                )
            ]
        )

        # Turn 2: Run code (will fail with ZeroDivisionError)
        turn_2 = LLMResponse(
            content="Running script.",
            tool_calls=[
                ToolCallRequest(
                    id="call_e1",
                    tool_name="execute_python",
                    arguments={"filename": "divide.py"}
                )
            ]
        )

        # Turn 3: Fix code after receiving ZeroDivisionError diagnostics
        turn_3 = LLMResponse(
            content="Detected ZeroDivisionError. Fixing divide.py to divide by 2.",
            tool_calls=[
                ToolCallRequest(
                    id="call_c2",
                    tool_name="create_file",
                    arguments={"filename": "divide.py", "content": "res = 10 / 2\nprint('Result:', res)\n", "overwrite": True, "approved": True}
                )
            ]
        )

        # Turn 4: Re-execute fixed code
        turn_4 = LLMResponse(
            content="Re-running fixed divide.py.",
            tool_calls=[
                ToolCallRequest(
                    id="call_e2",
                    tool_name="execute_python",
                    arguments={"filename": "divide.py"}
                )
            ]
        )

        # Turn 5: Observe success and conclude
        turn_5 = LLMResponse(
            content="The calculation succeeded with Result: 5.0 after fixing the ZeroDivisionError.",
            tool_calls=[]
        )

        mock_llm.chat.side_effect = [turn_1, turn_2, turn_3, turn_4, turn_5]

        agent = CodeMateAgent(config=agent_config, registry=registry, llm_client=mock_llm)
        result = agent.run("Calculate 10 divided by something", max_iterations=6)

        assert result.success is True
        assert result.iterations == 5
        assert result.history_attempts == 1 if hasattr(result, "history_attempts") else result.correction_attempts == 1
        assert len(result.execution_history) == 2
        assert result.execution_history[0]["success"] is False
        assert result.execution_history[0]["diagnostics"]["exception_name"] == "ZeroDivisionError"
        assert result.execution_history[1]["success"] is True
        assert "Result: 5.0" in result.execution_history[1]["stdout"]

    def test_run_agent_convenience_function(self, agent_config):
        mock_llm = MagicMock()
        mock_llm.format_tool_schema.return_value = []
        mock_llm.chat.return_value = LLMResponse(
            content="Answer from run_agent.",
            tool_calls=[]
        )

        result = run_agent(
            task="Simple task",
            config=agent_config,
            llm_client=mock_llm,
            max_iterations=2
        )

        assert result.success is True
        assert result.final_response == "Answer from run_agent."
