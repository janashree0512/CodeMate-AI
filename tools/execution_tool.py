"""
CodeMate AI - Execution Tool (tools/execution_tool.py)
======================================================

Controlled Python code execution tool using subprocess.
Enforces execution timeouts, workspace path isolation, environment variable
scrubbing (removing GROQ_API_KEY and host credentials), and human approval gates.
"""

import os
import sys
import time
import re
import shlex
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import Config, load_config
from tools.file_tools import resolve_safe_path


class BaseExecutor(ABC):
    """
    Abstract base interface for execution backends.
    Allows swapping between local restricted subprocess and containerized (Docker) execution.
    """

    @abstractmethod
    def execute(
        self,
        filename: str,
        approved: bool = False,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute a target script and return structured results."""
        pass


def build_sanitized_environment(workspace_dir: Path) -> Dict[str, str]:
    """
    Construct a minimal, sanitized environment for child Python processes.
    Strips all sensitive credentials, host tokens, and GROQ_API_KEY.
    """
    clean_env: Dict[str, str] = {
        "PYTHONPATH": str(workspace_dir.resolve()),
        "PYTHONUNBUFFERED": "1",
    }

    # Pass minimal necessary OS variables for Python interpreter to run
    system_pass_keys = [
        "SYSTEMROOT",
        "WINDIR",
        "PATHEXT",
        "COMSPEC",
        "TEMP",
        "TMP",
        "PATH"
    ]
    for key in system_pass_keys:
        val = os.environ.get(key)
        if val is not None:
            clean_env[key] = val

    # Explicitly ensure sensitive keys are never passed
    sensitive_keys = [
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GITHUB_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ]
    for key in sensitive_keys:
        clean_env.pop(key, None)

    return clean_env


class LocalRestrictedExecutor(BaseExecutor):
    """
    Executes Python scripts as isolated local subprocesses within the workspace directory.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or load_config(strict_api_key=False)

    def execute(
        self,
        filename: str,
        approved: bool = False,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute a Python script residing in the workspace directory.

        Args:
            filename: Relative path to the Python file in workspace.
            approved: Explicit user approval flag. If required and False, execution is aborted.
            timeout: Optional override for execution timeout (seconds).

        Returns:
            Structured execution report dictionary.
        """
        # 1. Enforce Human-in-the-Loop approval gate
        if self.config.require_user_approval and not approved:
            return {
                "success": False,
                "filename": filename,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "execution_time_seconds": 0.0,
                "mode": "restricted_local",
                "error": (
                    f"Execution of '{filename}' blocked: Explicit user approval is required "
                    "before running generated code on the host system."
                ),
                "approval_required": True
            }

        # 2. Path verification: target must be inside workspace/
        try:
            target_path = resolve_safe_path(filename, self.config.workspace_dir)
        except (PermissionError, ValueError) as err:
            return {
                "success": False,
                "filename": filename,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "execution_time_seconds": 0.0,
                "mode": "restricted_local",
                "error": str(err),
                "approval_required": False
            }

        if not target_path.exists():
            return {
                "success": False,
                "filename": filename,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "execution_time_seconds": 0.0,
                "mode": "restricted_local",
                "error": f"Script not found: '{filename}'",
                "approval_required": False
            }

        if target_path.is_dir():
            return {
                "success": False,
                "filename": filename,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "execution_time_seconds": 0.0,
                "mode": "restricted_local",
                "error": f"Path '{filename}' is a directory, not an executable file.",
                "approval_required": False
            }

        # 3. Determine timeout threshold
        effective_timeout = timeout if timeout is not None else self.config.execution_timeout_seconds

        # 4. Prepare execution environment
        child_env = build_sanitized_environment(self.config.workspace_dir)
        cmd = [sys.executable, str(target_path)]

        start_time = time.perf_counter()
        try:
            process = subprocess.run(
                cmd,
                cwd=str(self.config.workspace_dir.resolve()),
                env=child_env,
                capture_output=True,
                text=True,
                timeout=effective_timeout
            )
            elapsed = time.perf_counter() - start_time

            return {
                "success": (process.returncode == 0),
                "filename": filename,
                "exit_code": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "timed_out": False,
                "execution_time_seconds": round(elapsed, 4),
                "mode": "restricted_local",
                "error": None if process.returncode == 0 else "Execution failed with non-zero exit code.",
                "approval_required": False
            }

        except subprocess.TimeoutExpired as exc:
            elapsed = time.perf_counter() - start_time
            stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout.decode("utf-8", "replace") if exc.stdout else "")
            stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr.decode("utf-8", "replace") if exc.stderr else "")

            return {
                "success": False,
                "filename": filename,
                "exit_code": -1,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": True,
                "execution_time_seconds": round(elapsed, 4),
                "mode": "restricted_local",
                "error": f"Execution timed out after {effective_timeout} seconds (potential infinite loop).",
                "approval_required": False
            }

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            return {
                "success": False,
                "filename": filename,
                "exit_code": -1,
                "stdout": "",
                "stderr": str(exc),
                "timed_out": False,
                "execution_time_seconds": round(elapsed, 4),
                "mode": "restricted_local",
                "error": f"Unexpected execution error: {exc}",
                "approval_required": False
            }


def execute_python(
    filename: str,
    approved: bool = False,
    timeout: Optional[int] = None,
    config: Optional[Config] = None
) -> Dict[str, Any]:
    """
    Functional entry point for executing Python scripts in the workspace.
    """
    executor = LocalRestrictedExecutor(config=config)
    return executor.execute(filename=filename, approved=approved, timeout=timeout)


# Allowed command prefixes for run_shell
ALLOWED_COMMANDS = {"pytest", "python", "pip", "ls", "cat"}

# Disallowed dangerous operators / tokens (e.g. redirection, chaining, traversal, escalation)
DANGEROUS_TOKENS = {"rm", "sudo", "curl", "wget", "|", ";", "&&", ">", ">>", "<", ".."}


def run_shell(
    command: str,
    approved: bool = False,
    timeout: Optional[int] = None,
    config: Optional[Config] = None
) -> Dict[str, Any]:
    """
    Run an approved allowlisted command inside the sandboxed workspace without shell=True.

    Allowlist:
    - pytest, python, pip (install only), ls, cat.
    - Dangerous tokens (rm, sudo, curl, wget, |, ;, &&, >, ..) are strictly blocked.

    Args:
        command: Command string to execute.
        approved: User approval flag. Required if require_user_approval is enabled.
        timeout: Execution timeout in seconds (defaults to config.execution_timeout_seconds).
        config: Optional configuration instance.

    Returns:
        Structured result dict matching execute_python shape.
    """
    cfg = config or load_config(strict_api_key=False)

    if not command or not command.strip():
        return {
            "success": False,
            "command": command,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "execution_time_seconds": 0.0,
            "mode": "restricted_shell",
            "error": "Command string cannot be empty.",
            "approval_required": False
        }

    # 1. Enforce Human-in-the-Loop approval gate
    if cfg.require_user_approval and not approved:
        return {
            "success": False,
            "command": command,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "execution_time_seconds": 0.0,
            "mode": "restricted_shell",
            "error": "Execution of shell commands requires explicit user approval.",
            "approval_required": True
        }

    # 2. Split command using shlex
    try:
        args = shlex.split(command, posix=True)
    except ValueError as exc:
        return {
            "success": False,
            "command": command,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "execution_time_seconds": 0.0,
            "mode": "restricted_shell",
            "error": f"Failed to parse command line: {exc}",
            "approval_required": False
        }

    if not args:
        return {
            "success": False,
            "command": command,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "execution_time_seconds": 0.0,
            "mode": "restricted_shell",
            "error": "Parsed command is empty.",
            "approval_required": False
        }

    # Check raw command and args for dangerous tokens / operators
    for token in DANGEROUS_TOKENS:
        # Check if token exists in args or raw command symbols
        if token in args or (token in {"|", ";", "&&", ">", ">>", "<", ".."} and token in command):
            return {
                "success": False,
                "command": command,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "execution_time_seconds": 0.0,
                "mode": "restricted_shell",
                "error": f"Security violation: Token or operator '{token}' is strictly forbidden.",
                "approval_required": False
            }

    base_cmd = Path(args[0]).name.lower()
    if base_cmd.endswith(".exe"):
        base_cmd = base_cmd[:-4]

    # Check allowlist
    if base_cmd not in ALLOWED_COMMANDS:
        return {
            "success": False,
            "command": command,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "execution_time_seconds": 0.0,
            "mode": "restricted_shell",
            "error": f"Command '{base_cmd}' is not in the allowlist ({sorted(ALLOWED_COMMANDS)}).",
            "approval_required": False
        }

    # Check pip restriction: pip install only
    if base_cmd == "pip":
        if len(args) < 2 or args[1] != "install":
            return {
                "success": False,
                "command": command,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "execution_time_seconds": 0.0,
                "mode": "restricted_shell",
                "error": "pip usage is restricted: only 'pip install' is permitted.",
                "approval_required": False
            }

    # Check argument paths for directory traversal (".." or "../")
    for arg in args[1:]:
        if ".." in arg:
            return {
                "success": False,
                "command": command,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "execution_time_seconds": 0.0,
                "mode": "restricted_shell",
                "error": f"Security violation: Path traversal '..' in argument '{arg}' is forbidden.",
                "approval_required": False
            }

    # Resolve executable: use sys.executable if running python
    exec_args = list(args)
    if base_cmd == "python":
        exec_args[0] = sys.executable

    effective_timeout = timeout if timeout is not None else cfg.execution_timeout_seconds
    child_env = build_sanitized_environment(cfg.workspace_dir)

    start_time = time.perf_counter()
    try:
        process = subprocess.run(
            exec_args,
            cwd=str(cfg.workspace_dir.resolve()),
            env=child_env,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            shell=False
        )
        elapsed = time.perf_counter() - start_time

        return {
            "success": (process.returncode == 0),
            "command": command,
            "exit_code": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "timed_out": False,
            "execution_time_seconds": round(elapsed, 4),
            "mode": "restricted_shell",
            "error": None if process.returncode == 0 else "Command exited with non-zero exit code.",
            "approval_required": False
        }

    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start_time
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout.decode("utf-8", "replace") if exc.stdout else "")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr.decode("utf-8", "replace") if exc.stderr else "")

        return {
            "success": False,
            "command": command,
            "exit_code": -1,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": True,
            "execution_time_seconds": round(elapsed, 4),
            "mode": "restricted_shell",
            "error": f"Command timed out after {effective_timeout} seconds.",
            "approval_required": False
        }

    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        return {
            "success": False,
            "command": command,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(exc),
            "timed_out": False,
            "execution_time_seconds": round(elapsed, 4),
            "mode": "restricted_shell",
            "error": f"Unexpected execution error: {exc}",
            "approval_required": False
        }


def run_tests(
    path: str = "",
    approved: bool = False,
    timeout: Optional[int] = None,
    config: Optional[Config] = None
) -> Dict[str, Any]:
    """
    Convenience wrapper that runs pytest inside workspace/ and returns a compact
    pass/fail summary along with names of failing tests.

    Args:
        path: Optional relative test file or test folder path inside workspace/.
        approved: User approval flag. Required if require_user_approval is enabled.
        timeout: Execution timeout in seconds.
        config: Optional configuration instance.

    Returns:
        Structured test run dictionary with summary metrics and failure details.
    """
    cfg = config or load_config(strict_api_key=False)

    # Validate path if provided
    if path and path.strip():
        try:
            target = resolve_safe_path(path, cfg.workspace_dir)
            if not target.exists():
                return {
                    "success": False,
                    "path": path,
                    "exit_code": None,
                    "error": f"Test path not found in workspace: '{path}'",
                    "approval_required": False,
                    "summary": "Path not found",
                    "failed_tests": []
                }
            pytest_target = str(target.relative_to(cfg.workspace_dir.resolve())).replace("\\", "/")
        except (PermissionError, ValueError) as err:
            return {
                "success": False,
                "path": path,
                "exit_code": None,
                "error": str(err),
                "approval_required": False,
                "summary": "Invalid test path",
                "failed_tests": []
            }
    else:
        pytest_target = ""

    # Build pytest command: run pytest module via sys.executable
    cmd_parts = ["python", "-m", "pytest", "-q"]
    if pytest_target:
        cmd_parts.append(pytest_target)
    cmd_str = " ".join(cmd_parts)

    shell_result = run_shell(
        command=cmd_str,
        approved=approved,
        timeout=timeout,
        config=cfg
    )

    if not shell_result.get("success", False) and shell_result.get("approval_required"):
        return {
            "success": False,
            "path": path,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "error": "Running tests requires explicit user approval.",
            "approval_required": True,
            "summary": "Approval required",
            "failed_tests": []
        }

    stdout = shell_result.get("stdout", "")
    stderr = shell_result.get("stderr", "")
    exit_code = shell_result.get("exit_code")

    # Parse failed test names from pytest output
    # Typical pytest output lines:
    # FAILED test_calc.py::test_fail - AssertionError
    failed_tests: List[str] = []
    for line in (stdout + "\n" + stderr).splitlines():
        line_clean = line.strip()
        if line_clean.startswith("FAILED "):
            test_id = line_clean.split()[1] if len(line_clean.split()) > 1 else line_clean
            failed_tests.append(test_id)

    # Derive compact summary
    summary = "Tests passed successfully" if exit_code == 0 else f"{len(failed_tests)} test(s) failed"
    if shell_result.get("timed_out"):
        summary = "Test run timed out"

    return {
        "success": (exit_code == 0),
        "path": path,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": shell_result.get("timed_out", False),
        "execution_time_seconds": shell_result.get("execution_time_seconds", 0.0),
        "summary": summary,
        "failed_tests": failed_tests,
        "error": shell_result.get("error"),
        "approval_required": False
    }

