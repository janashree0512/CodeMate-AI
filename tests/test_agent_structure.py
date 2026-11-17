"""
Verification tests for the initial agent module structure and imports.
"""

import config
import agent
from agent.prompts import SYSTEM_PROMPT, TASK_PLANNING_TEMPLATE, ERROR_CORRECTION_PROMPT
from agent.llm_client import GroqClient
from agent.agent import CodeMateAgent, AgentExecutionState
import tools
from tools.file_tools import create_file, read_file, list_files, delete_file
from tools.execution_tool import execute_python, LocalRestrictedExecutor
from tools.tool_registry import ToolRegistry, create_default_registry


def test_agent_package_imports():
    assert SYSTEM_PROMPT is not None
    assert TASK_PLANNING_TEMPLATE is not None
    assert ERROR_CORRECTION_PROMPT is not None


def test_tools_package_imports():
    assert callable(create_file)
    assert callable(read_file)
    assert callable(list_files)
    assert callable(delete_file)
    assert callable(execute_python)
    assert LocalRestrictedExecutor is not None


def test_llm_client_tool_schema_generation(tmp_path):
    test_cfg = config.Config(
        groq_api_key="mock",
        model_name="llama-3.3-70b-versatile",
        max_iterations=5,
        execution_timeout_seconds=5,
        require_user_approval=True,
        workspace_dir=tmp_path,
        project_root=tmp_path
    )
    reg = create_default_registry(config=test_cfg)
    client = GroqClient(config=test_cfg)
    schemas = client.format_tool_schema(reg)

    assert len(schemas) == 5
    schema_names = [s["function"]["name"] for s in schemas]
    assert "create_file" in schema_names
    assert "execute_python" in schema_names


def test_agent_controller_initialization(tmp_path):
    test_cfg = config.Config(
        groq_api_key="mock",
        model_name="llama-3.3-70b-versatile",
        max_iterations=3,
        execution_timeout_seconds=5,
        require_user_approval=True,
        workspace_dir=tmp_path,
        project_root=tmp_path
    )
    agent_inst = CodeMateAgent(config=test_cfg)
    agent_inst.initialize_session("Write a bubble sort algorithm")

    history = agent_inst.get_conversation_history()
    assert len(history) == 2
    assert history[0]["role"] == "system"
    assert history[1]["role"] == "user"
    assert history[1]["content"] == "Write a bubble sort algorithm"

    # Step lifecycle
    step_1 = agent_inst.execute_step()
    assert step_1["terminated"] is False
    assert step_1["iteration"] == 1

    step_2 = agent_inst.execute_step()
    step_3 = agent_inst.execute_step()
    assert step_3["iteration"] == 3

    # Max iteration termination check
    step_4 = agent_inst.execute_step()
    assert step_4["terminated"] is True
    assert step_4["reason"] == "max_iterations_reached"
