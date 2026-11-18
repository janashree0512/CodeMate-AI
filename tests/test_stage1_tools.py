"""
Unit tests for Stage 1 Agent Tools:
1. edit_file
2. run_shell
3. search_files
4. update_plan & show_plan
5. run_tests
"""

import sys
from pathlib import Path
import pytest

from config import Config
from tools.file_tools import (
    create_file,
    read_file,
    edit_file,
    search_files,
    resolve_safe_path
)
from tools.execution_tool import run_shell, run_tests
from tools.tool_registry import create_default_registry
from agent.agent import CodeMateAgent, AgentExecutionState


@pytest.fixture
def stage1_config(tmp_path):
    """Provides an isolated workspace configuration for Stage 1 tool tests."""
    ws = tmp_path / "stage1_workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return Config(
        groq_api_key="mock_key",
        model_name="llama-3.3-70b-versatile",
        max_iterations=10,
        execution_timeout_seconds=5,
        require_user_approval=True,
        workspace_dir=ws,
        project_root=tmp_path,
        max_tool_output_chars=4000,
        max_history_messages=30
    )


class TestEditFileTool:
    """Tests for edit_file: replacement, unique match enforcement, path safety, approval."""

    def test_edit_file_success(self, stage1_config):
        create_file("module.py", "def greet():\n    return 'Hello'\n", config=stage1_config)
        res = edit_file("module.py", "'Hello'", "'Hi there'", approved=True, config=stage1_config)

        assert res["success"] is True
        assert res["occurrences_replaced"] == 1
        read_res = read_file("module.py", config=stage1_config)
        assert "return 'Hi there'" in read_res["content"]

    def test_edit_file_requires_approval(self, stage1_config):
        create_file("demo.py", "x = 10\n", config=stage1_config)
        res = edit_file("demo.py", "10", "20", approved=False, config=stage1_config)

        assert res["success"] is False
        assert res["approval_required"] is True
        assert "requires explicit user approval" in res["error"]
        # Ensure file was not modified
        assert read_file("demo.py", config=stage1_config)["content"] == "x = 10\n"

    def test_edit_file_not_found_pattern(self, stage1_config):
        create_file("demo.py", "x = 10\n", config=stage1_config)
        res = edit_file("demo.py", "nonexistent_string", "20", approved=True, config=stage1_config)

        assert res["success"] is False
        assert "not found" in res["error"]

    def test_edit_file_multiple_occurrences(self, stage1_config):
        create_file("demo.py", "foo = 1\nbar = 1\n", config=stage1_config)
        res = edit_file("demo.py", "= 1", "= 2", approved=True, config=stage1_config)

        assert res["success"] is False
        assert "found 2 times" in res["error"]
        assert "requires exactly one match" in res["error"]

    def test_edit_file_missing_file(self, stage1_config):
        res = edit_file("nonexistent.py", "a", "b", approved=True, config=stage1_config)
        assert res["success"] is False
        assert "File not found" in res["error"]

    def test_edit_file_path_traversal(self, stage1_config):
        res = edit_file("../../outside.py", "a", "b", approved=True, config=stage1_config)
        assert res["success"] is False
        assert "Path traversal detected" in res["error"]


class TestSearchFilesTool:
    """Tests for search_files: pattern matching, capping, and path jail."""

    def test_search_files_success(self, stage1_config):
        create_file("a.py", "def target_func():\n    pass\n", config=stage1_config)
        create_file("b.py", "x = 1\n# calls target_func\ntarget_func()\n", config=stage1_config)

        res = search_files("target_func", config=stage1_config)
        assert res["success"] is True
        assert res["total_matches"] == 3
        filenames = [m["filename"] for m in res["matches"]]
        assert "a.py" in filenames
        assert "b.py" in filenames

    def test_search_files_empty_pattern(self, stage1_config):
        res = search_files("", config=stage1_config)
        assert res["success"] is False
        assert "cannot be empty" in res["error"]

    def test_search_files_path_traversal(self, stage1_config):
        res = search_files("test", directory="../..", config=stage1_config)
        assert res["success"] is False
        assert "Path traversal detected" in res["error"]

    def test_search_files_capped_results(self, stage1_config):
        lines = "\n".join([f"KEYWORD_{i} matched" for i in range(100)])
        create_file("big.txt", lines, config=stage1_config)

        res = search_files("KEYWORD", config=stage1_config)
        assert res["success"] is True
        assert res["total_matches"] == 50
        assert res["capped"] is True


class TestRunShellTool:
    """Tests for run_shell: allowlist, dangerous token blocking, timeout, and approval."""

    def test_run_shell_success(self, stage1_config):
        res = run_shell("python -c \"print('Hello from shell')\"", approved=True, config=stage1_config)
        assert res["success"] is True
        assert res["exit_code"] == 0
        assert "Hello from shell" in res["stdout"]

    def test_run_shell_requires_approval(self, stage1_config):
        res = run_shell("python -c \"print(1)\"", approved=False, config=stage1_config)
        assert res["success"] is False
        assert res["approval_required"] is True
        assert "requires explicit user approval" in res["error"]

    def test_run_shell_blocked_commands(self, stage1_config):
        # Non-allowlisted command
        res = run_shell("git status", approved=True, config=stage1_config)
        assert res["success"] is False
        assert "is not in the allowlist" in res["error"]

    @pytest.mark.parametrize("dangerous_cmd", [
        "python -c '1' ; ls",
        "python -c '1' && ls",
        "cat file | python",
        "python -c '1' > out.txt",
        "rm -rf foo",
        "sudo ls",
        "curl http://example.com",
        "wget http://example.com",
    ])
    def test_run_shell_blocked_dangerous_tokens(self, stage1_config, dangerous_cmd):
        res = run_shell(dangerous_cmd, approved=True, config=stage1_config)
        assert res["success"] is False
        assert "Security violation" in res["error"] or "is not in the allowlist" in res["error"]

    def test_run_shell_path_traversal_blocked(self, stage1_config):
        res = run_shell("cat ../secret.txt", approved=True, config=stage1_config)
        assert res["success"] is False
        assert "Path traversal" in res["error"] or "Security violation" in res["error"]

    def test_run_shell_pip_install_only(self, stage1_config):
        # pip uninstall blocked
        res_bad = run_shell("pip uninstall pytest", approved=True, config=stage1_config)
        assert res_bad["success"] is False
        assert "only 'pip install' is permitted" in res_bad["error"]

    def test_run_shell_timeout(self, stage1_config):
        create_file("sleep_loop.py", "import time\ntime.sleep(10)\n", config=stage1_config)
        res = run_shell("python sleep_loop.py", approved=True, timeout=1, config=stage1_config)
        assert res["success"] is False
        assert res["timed_out"] is True
        assert "timed out" in res["error"]


class TestRunTestsTool:
    """Tests for run_tests: pytest wrapper, summary, and failure extraction."""

    def test_run_tests_success(self, stage1_config):
        test_content = (
            "def test_ok():\n"
            "    assert 1 + 1 == 2\n"
        )
        create_file("test_sample.py", test_content, config=stage1_config)

        res = run_tests("test_sample.py", approved=True, config=stage1_config)
        assert res["success"] is True
        assert res["exit_code"] == 0
        assert res["summary"] == "Tests passed successfully"
        assert res["failed_tests"] == []

    def test_run_tests_failure_detection(self, stage1_config):
        test_content = (
            "def test_pass():\n"
            "    assert True\n\n"
            "def test_failing_case():\n"
            "    assert 1 == 2\n"
        )
        create_file("test_fail.py", test_content, config=stage1_config)

        res = run_tests("test_fail.py", approved=True, config=stage1_config)
        assert res["success"] is False
        assert res["exit_code"] != 0
        assert "failed" in res["summary"]
        assert any("test_failing_case" in f for f in res["failed_tests"])

    def test_run_tests_requires_approval(self, stage1_config):
        create_file("test_app.py", "def test_a(): pass\n", config=stage1_config)
        res = run_tests("test_app.py", approved=False, config=stage1_config)
        assert res["success"] is False
        assert res["approval_required"] is True

    def test_run_tests_nonexistent_path(self, stage1_config):
        res = run_tests("does_not_exist.py", approved=True, config=stage1_config)
        assert res["success"] is False
        assert "Test path not found" in res["error"]


class TestPlanningTools:
    """Tests for update_plan and show_plan in AgentExecutionState & CodeMateAgent."""

    def test_agent_execution_state_plan(self):
        state = AgentExecutionState("task description", max_iterations=5)
        assert state.get_plan() == []

        state.set_plan(["1. Read requirements", "2. Create script", "3. Test"])
        assert len(state.get_plan()) == 3
        assert state.plan[0] == "1. Read requirements"

        # Check action logs
        assert len(state.action_logs) == 1
        assert state.action_logs[0]["action"] == "update_plan"
        assert state.action_logs[0]["details"]["step_count"] == 3

    def test_agent_plan_tools_integration(self, stage1_config):
        agent = CodeMateAgent(config=stage1_config)
        agent.initialize_session("Implement binary search")

        up_res = agent.update_plan(["Step 1: Write algo", "Step 2: Run pytest"])
        assert up_res["success"] is True
        assert len(up_res["plan"]) == 2

        show_res = agent.show_plan()
        assert show_res["success"] is True
        assert show_res["total_steps"] == 2
        assert show_res["plan"][1] == "Step 2: Run pytest"


class TestDefaultRegistryStage1:
    """Tests that all 5 new/updated tools are registered in default registry."""

    def test_all_stage1_tools_registered(self, stage1_config):
        registry = create_default_registry(config=stage1_config)
        tools = {t["name"] for t in registry.list_tools()}

        expected = {
            "create_file",
            "read_file",
            "edit_file",
            "delete_file",
            "list_files",
            "search_files",
            "execute_python",
            "run_shell",
            "run_tests",
            "update_plan",
            "show_plan"
        }
        for tool_name in expected:
            assert tool_name in tools, f"Missing tool in registry: {tool_name}"
