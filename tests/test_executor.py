"""
Unit tests for the Controlled Python Code Execution Engine (tools/code_executor.py).
Verifies subprocess isolation, stdout/stderr capture, exit codes, timeout handling,
human-in-the-loop approval, and credential scrubbing.
"""

import os
from pathlib import Path
import pytest

from config import Config
from tools.file_tools import create_file
from tools.code_executor import execute_python, LocalRestrictedExecutor


@pytest.fixture
def exec_config(tmp_path):
    """Provides an isolated configuration pointing to a temporary workspace."""
    workspace = tmp_path / "exec_sandbox"
    workspace.mkdir(parents=True, exist_ok=True)
    return Config(
        groq_api_key="mock_key",
        model_name="llama-3.3-70b-versatile",
        max_iterations=10,
        execution_timeout_seconds=3,
        require_user_approval=True,
        workspace_dir=workspace,
        project_root=tmp_path
    )


class TestCodeExecutor:
    """Test suite for LocalRestrictedExecutor."""

    def test_execute_success_stdout(self, exec_config):
        script_content = (
            "def calculate():\n"
            "    return [x**2 for x in range(5)]\n"
            "print('RESULT:', calculate())\n"
        )
        create_file("test_math.py", script_content, config=exec_config)

        result = execute_python("test_math.py", approved=True, config=exec_config)

        assert result["success"] is True
        assert result["exit_code"] == 0
        assert "RESULT: [0, 1, 4, 9, 16]" in result["stdout"]
        assert result["stderr"] == ""
        assert result["timed_out"] is False
        assert result["mode"] == "restricted_local"
        assert result["execution_time_seconds"] >= 0.0

    def test_execute_runtime_error_stderr(self, exec_config):
        script_content = (
            "items = [1, 2, 3]\n"
            "print(items[10])\n"  # IndexError
        )
        create_file("buggy.py", script_content, config=exec_config)

        result = execute_python("buggy.py", approved=True, config=exec_config)

        assert result["success"] is False
        assert result["exit_code"] != 0
        assert "IndexError: list index out of range" in result["stderr"]
        assert result["timed_out"] is False

    def test_execute_syntax_error(self, exec_config):
        script_content = "def incomplete_func("  # SyntaxError
        create_file("syntax_bug.py", script_content, config=exec_config)

        result = execute_python("syntax_bug.py", approved=True, config=exec_config)

        assert result["success"] is False
        assert result["exit_code"] != 0
        assert "SyntaxError" in result["stderr"]

    def test_execute_infinite_loop_timeout(self, exec_config):
        # Program that loops infinitely
        script_content = (
            "import time\n"
            "print('Starting loop...', flush=True)\n"
            "while True:\n"
            "    time.sleep(0.05)\n"
        )
        create_file("infinite.py", script_content, config=exec_config)

        # Execute with an aggressive 1-second timeout
        result = execute_python("infinite.py", approved=True, timeout=1, config=exec_config)

        assert result["success"] is False
        assert result["timed_out"] is True
        assert result["exit_code"] == -1
        assert "timed out after 1 seconds" in result["error"]
        assert "Starting loop..." in result["stdout"]

    def test_execute_approval_enforcement(self, exec_config):
        create_file("harmless.py", "print('Hello')", config=exec_config)

        # Unapproved execution attempt
        result = execute_python("harmless.py", approved=False, config=exec_config)

        assert result["success"] is False
        assert result.get("approval_required") is True
        assert "Explicit user approval is required" in result["error"]

    def test_execute_credential_isolation(self, exec_config, monkeypatch):
        # Inject sensitive host credentials into the current process
        monkeypatch.setenv("GROQ_API_KEY", "secret_groq_api_token_12345")
        monkeypatch.setenv("OPENAI_API_KEY", "secret_openai_api_token_67890")

        # Create a script that attempts to inspect environment variables
        script_content = (
            "import os\n"
            "print('GROQ:', os.environ.get('GROQ_API_KEY'))\n"
            "print('OPENAI:', os.environ.get('OPENAI_API_KEY'))\n"
        )
        create_file("inspect_env.py", script_content, config=exec_config)

        result = execute_python("inspect_env.py", approved=True, config=exec_config)

        assert result["success"] is True
        assert "GROQ: None" in result["stdout"]
        assert "OPENAI: None" in result["stdout"]

    def test_execute_path_traversal_blocked(self, exec_config):
        result = execute_python("../../outside_script.py", approved=True, config=exec_config)
        assert result["success"] is False
        assert "Path traversal detected" in result["error"]

    def test_execute_nonexistent_script(self, exec_config):
        result = execute_python("not_real.py", approved=True, config=exec_config)
        assert result["success"] is False
        assert "Script not found" in result["error"]
