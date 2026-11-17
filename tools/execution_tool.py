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
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

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
