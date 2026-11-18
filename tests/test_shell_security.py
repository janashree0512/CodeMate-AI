"""
Tests for Shell Security and Sandbox Jail Validation (tests/test_shell_security.py).
Ensures that all path escape attempts, unsafe shell operators, arbitrary python execution,
and malicious imports are strictly blocked.
"""

import os
import sys
from pathlib import Path
import pytest

from config import Config
from tools.execution_tool import run_shell, execute_python, check_python_script_safety
from tools.file_tools import create_file


@pytest.fixture
def security_test_config(tmp_path):
    """Fixture providing isolated workspace and project root for security tests."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    env_file = project_root / ".env"
    env_file.write_text("GROQ_API_KEY=gsk_test1234\nMODEL_NAME=openai/gpt-oss-120b\n", encoding="utf-8")

    ws = project_root / "workspace"
    ws.mkdir()

    # Create dummy sorter.py and test_sorter.py inside workspace
    sorter_code = (
        "def bubble_sort(arr):\n"
        "    return sorted(arr)\n"
        "if __name__ == '__main__':\n"
        "    assert bubble_sort([2, 1]) == [1, 2]\n"
        "    print('All tests passed.')\n"
    )
    (ws / "sorter.py").write_text(sorter_code, encoding="utf-8")

    test_sorter_code = (
        "from sorter import bubble_sort\n\n"
        "def test_bubble():\n"
        "    assert bubble_sort([3, 1]) == [1, 3]\n"
    )
    (ws / "test_sorter.py").write_text(test_sorter_code, encoding="utf-8")

    return Config(
        groq_api_key="gsk_test1234",
        model_name="openai/gpt-oss-120b",
        max_iterations=10,
        execution_timeout_seconds=5,
        require_user_approval=True,
        max_tool_output_chars=4000,
        max_history_messages=20,
        workspace_dir=ws,
        project_root=project_root
    )


BLOCKED_COMMAND_CASES = [
    ("cat /etc/passwd", "Absolute path /etc/passwd"),
    ("ls /", "Absolute path /"),
    ("cat C:\\Windows\\win.ini", "Windows drive letter absolute path"),
    ("cat ..\\.env", "Path traversal to .env"),
    ('python -c "print(1)"', "python -c arbitrary code execution"),
    ("python -m http.server", "python -m arbitrary server module"),
    ("python -", "python standard input"),
    ("python ../escape.py", "python directory traversal"),
    ("pytest --rootdir /", "pytest --rootdir pointing to root"),
    ("pytest -c /etc/pytest.ini", "pytest -c pointing to outside config"),
    ("pip install git+https://github.com/evil/repo.git", "pip remote git repo"),
    ("pip install -e /", "pip editable install pointing to root"),
    ("pip install --target /tmp foo", "pip target pointing outside"),
    ("pip install -r /etc/requirements.txt", "pip requirements pointing to absolute path"),
    ("pip install ./evil.whl", "pip local wheel file path"),
    ("ls /var", "ls absolute path"),
    ("ls C:\\", "ls drive letter"),
]


@pytest.mark.parametrize("cmd,desc", BLOCKED_COMMAND_CASES)
def test_blocked_escape_commands(security_test_config, cmd, desc):
    """Assert that every blocked attack command returns success=False with an explanatory error."""
    res = run_shell(cmd, approved=True, config=security_test_config)
    assert res["success"] is False, f"Command '{cmd}' ({desc}) unexpectedly succeeded!"
    assert res["error"] is not None
    assert "Security violation" in res["error"] or "strictly forbidden" in res["error"] or "not found" in res["error"]


def test_blocked_cat_env_absolute_path(security_test_config):
    """Test blocking cat with the actual absolute path to .env."""
    env_path = str(security_test_config.workspace_dir.parent / ".env")
    res = run_shell(f"cat {env_path}", approved=True, config=security_test_config)
    assert res["success"] is False
    assert "Security violation" in res["error"]


def test_blocked_symlink_escape(security_test_config):
    """Test blocking a symlink placed inside workspace pointing to .env or /etc/passwd."""
    symlink_path = security_test_config.workspace_dir / "env_link"
    target = security_test_config.workspace_dir.parent / ".env"
    try:
        os.symlink(str(target), str(symlink_path))
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlink creation not permitted on this host OS: {exc}")

    res = run_shell("cat env_link", approved=True, config=security_test_config)
    assert res["success"] is False
    assert "Security violation" in res["error"]


class TestAllowedCommands:
    """Verify that legitimate sandbox commands succeed properly."""

    def test_ls_allowed(self, security_test_config):
        res = run_shell("ls", approved=True, config=security_test_config)
        assert res["success"] is True
        assert res["exit_code"] == 0
        assert "sorter.py" in res["stdout"]

    def test_cat_allowed(self, security_test_config):
        res = run_shell("cat sorter.py", approved=True, config=security_test_config)
        assert res["success"] is True
        assert res["exit_code"] == 0
        assert "def bubble_sort" in res["stdout"]

    def test_python_script_allowed(self, security_test_config):
        res = run_shell("python sorter.py", approved=True, config=security_test_config)
        assert res["success"] is True
        assert res["exit_code"] == 0
        assert "All tests passed." in res["stdout"]

    def test_pytest_allowed(self, security_test_config):
        res = run_shell("pytest -q", approved=True, config=security_test_config)
        assert res["success"] is True
        assert res["exit_code"] == 0
        assert "1 passed" in res["stdout"]

    def test_python_m_pytest_allowed(self, security_test_config):
        res = run_shell("python -m pytest -q", approved=True, config=security_test_config)
        assert res["success"] is True
        assert res["exit_code"] == 0
        assert "1 passed" in res["stdout"]

    def test_pip_dry_run_allowed(self, security_test_config):
        res = run_shell("pip install requests --dry-run", approved=True, config=security_test_config)
        assert res["success"] is True
        assert res["exit_code"] == 0


class TestAstSafetyCheck:
    """Verify AST security analysis for execute_python."""

    @pytest.mark.parametrize("bad_code,expected_token", [
        ("import subprocess\nsubprocess.run(['ls'])", "subprocess"),
        ("import socket\ns = socket.socket()", "socket"),
        ("import shutil\nshutil.rmtree('/tmp')", "shutil"),
        ("import ctypes\nctypes.CDLL(None)", "ctypes"),
        ("import os\nos.system('ls')", "os.system"),
        ("import os\nos.popen('ls')", "os.popen"),
        ("eval('1 + 1')", "eval"),
        ("exec('print(1)')", "exec"),
        ("open('/etc/passwd').read()", "/etc/passwd"),
        ("open('../.env').read()", "../.env"),
        ("open('C:\\\\Windows\\\\win.ini').read()", "win.ini"),
    ])
    def test_ast_rejects_unsafe_code(self, security_test_config, bad_code, expected_token):
        script_file = security_test_config.workspace_dir / "unsafe_test.py"
        script_file.write_text(bad_code, encoding="utf-8")

        # Direct AST check
        err = check_python_script_safety(script_file)
        assert err is not None
        assert expected_token in err or "strictly forbidden" in err or "forbidden" in err

        # Integration with execute_python
        exec_res = execute_python("unsafe_test.py", approved=True, config=security_test_config)
        assert exec_res["success"] is False
        assert "Security violation" in exec_res["error"]
