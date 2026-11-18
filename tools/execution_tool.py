"""
CodeMate AI - Execution Tool (tools/execution_tool.py)
======================================================

Controlled Python code execution tool using subprocess.
Enforces execution timeouts, workspace path isolation, environment variable
scrubbing (removing GROQ_API_KEY and host credentials), and human approval gates.
"""

import ast
import os
import sys
import time
import re
import shlex
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import Config, load_config
from tools.file_tools import resolve_safe_path


def _truncate_output(res: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    """Truncate stdout and stderr in execution result dictionary."""
    for key in ("stdout", "stderr"):
        val = res.get(key)
        if isinstance(val, str) and len(val) > max_chars:
            diff = len(val) - max_chars
            res[key] = val[:max_chars] + f"...[truncated {diff} chars]"
    return res


def check_python_script_safety(script_path: Path) -> Optional[str]:
    """
    Perform a static AST security scan on a Python script before execution.
    Rejects scripts that import restricted modules (subprocess, socket, shutil, ctypes),
    call dangerous OS functions (os.system, os.popen), use dynamic code evaluation (eval, exec),
    or open() paths that are absolute or contain '..'.
    """
    try:
        content = script_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(content, filename=str(script_path))
    except SyntaxError:
        # Allow syntax errors to surface naturally via Python runtime
        return None
    except Exception as exc:
        return f"Could not parse script for security analysis: {exc}"

    for node in ast.walk(tree):
        # 1. Imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod in {"subprocess", "socket", "shutil", "ctypes"}:
                    return f"Importing restricted module '{alias.name}' is strictly forbidden."
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_mod = node.module.split(".")[0]
                if root_mod in {"subprocess", "socket", "shutil", "ctypes"}:
                    return f"Importing from restricted module '{node.module}' is strictly forbidden."

        # 2. Function / Method Calls
        elif isinstance(node, ast.Call):
            # eval, exec
            if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
                return f"Calling '{node.func.id}()' is strictly forbidden."

            # os.system, os.popen
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"system", "popen"}:
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                    return f"Calling 'os.{node.func.attr}()' is strictly forbidden."

            # open() with absolute path or '..'
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    path_val = node.args[0].value
                    normalized_path = path_val.replace("\\", "/")
                    if (
                        normalized_path.startswith("/")
                        or normalized_path.startswith("~")
                        or re.match(r'^[a-zA-Z]:', path_val)
                        or ".." in path_val
                        or ".env" in path_val
                    ):
                        return f"Calling open() with an absolute, traversal, or protected path '{path_val}' is forbidden."

    return None


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

        # 2b. Static AST safety check
        safety_error = check_python_script_safety(target_path)
        if safety_error:
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
                    f"Security violation in '{filename}': {safety_error} "
                    "Please rewrite the code without importing restricted modules or accessing paths outside the workspace."
                ),
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

            res = {
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
            return _truncate_output(res, self.config.max_tool_output_chars)

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

# Disallowed dangerous operators / tokens (redirection, chaining, traversal, expansion, escalation)
FORBIDDEN_OPERATORS = [";", "&", "|", "`", "$", "%", ">", "<"]
DANGEROUS_COMMAND_TOKENS = {"rm", "sudo", "curl", "wget", "type"}


def _validate_shell_argument(
    arg: str,
    orig_arg: str,
    workspace_dir: Path,
    project_root: Path
) -> Optional[str]:
    """
    Validate a single command argument against sandbox escape attempts.
    Returns None if safe, or a security violation error message.
    """
    # 1. Reject absolute paths across platforms
    if arg.startswith("/") or arg.startswith("\\"):
        return f"Security violation: Absolute path '{orig_arg}' is forbidden."
    if re.match(r'^[a-zA-Z]:', orig_arg) or re.match(r'^[a-zA-Z]:', arg):
        return f"Security violation: Absolute path with drive letter '{orig_arg}' is forbidden."
    if orig_arg.startswith("\\\\") or arg.startswith("//"):
        return f"Security violation: UNC network path '{orig_arg}' is forbidden."
    if arg.startswith("~") or orig_arg.startswith("~"):
        return f"Security violation: User home directory '~' in '{orig_arg}' is forbidden."

    # 2. Reject traversal, shell expansion, and backticks
    if ".." in arg or ".." in orig_arg:
        return f"Security violation: Directory traversal '..' in '{orig_arg}' is forbidden."
    if "$" in arg or "$" in orig_arg:
        return f"Security violation: Character '$' in '{orig_arg}' is forbidden."
    if "%" in arg or "%" in orig_arg:
        return f"Security violation: Character '%' in '{orig_arg}' is forbidden."
    if "`" in arg or "`" in orig_arg:
        return f"Security violation: Backticks in '{orig_arg}' are forbidden."

    # 3. Path assignments (e.g. KEY=/etc/passwd or --flag=C:\Windows)
    if "=" in arg:
        _, val = arg.split("=", 1)
        val_orig = orig_arg.split("=", 1)[1] if "=" in orig_arg else val
        if (
            val.startswith("/")
            or val.startswith("\\")
            or val.startswith("~")
            or re.match(r'^[a-zA-Z]:', val_orig)
            or re.match(r'^[a-zA-Z]:', val)
            or "/" in val
            or "\\" in val_orig
            or ".." in val
        ):
            return f"Security violation: Assignment to path in argument '{orig_arg}' is forbidden."

    # 4. Sensitive targets: .env, .git, venv
    norm_lower = arg.lower().replace("\\", "/")
    for sensitive in [".env", ".git", "venv"]:
        if sensitive in norm_lower:
            return f"Security violation: Access to protected target '{sensitive}' in '{orig_arg}' is forbidden."

    # Check project root reference outside workspace
    root_str = str(project_root.resolve()).lower().replace("\\", "/")
    if root_str in norm_lower:
        return f"Security violation: Reference to project root in '{orig_arg}' is forbidden."

    # 5. Path resolution & jail check
    target_candidate = workspace_dir / arg
    looks_like_path = (
        ("/" in arg or "\\" in orig_arg or re.search(r'\.[a-zA-Z0-9_-]+$', arg))
        and not arg.startswith("-")
    )
    if looks_like_path or target_candidate.is_symlink():
        try:
            resolved = target_candidate.resolve()
            resolved.relative_to(workspace_dir.resolve())
        except (ValueError, RuntimeError):
            return f"Security violation: Path '{orig_arg}' escapes workspace boundary."

    return None


def _validate_pytest_subargs(subargs: List[str], orig_subargs: List[str], workspace_dir: Path, project_root: Path) -> Optional[str]:
    """Validate arguments passed to pytest or python -m pytest."""
    safe_flags = {"-q", "-v", "-x", "-s", "--tb=short"}
    i = 0
    while i < len(subargs):
        arg = subargs[i]
        orig_arg = orig_subargs[i] if i < len(orig_subargs) else arg

        # Reject explicitly forbidden flags
        for forbidden in ["--rootdir", "-p", "--basetemp", "-c"]:
            if arg == forbidden or arg.startswith(f"{forbidden}="):
                return f"Security violation: Flag '{orig_arg}' is strictly forbidden for pytest."

        # Validate general argument safety
        gen_err = _validate_shell_argument(arg, orig_arg, workspace_dir, project_root)
        if gen_err:
            return gen_err

        if arg.startswith("-"):
            if arg in safe_flags:
                i += 1
                continue
            elif arg == "-k":
                if i + 1 >= len(subargs):
                    return "Security violation: Missing pattern for '-k' flag."
                pattern = subargs[i + 1]
                pattern_orig = orig_subargs[i + 1] if i + 1 < len(orig_subargs) else pattern
                p_err = _validate_shell_argument(pattern, pattern_orig, workspace_dir, project_root)
                if p_err:
                    return p_err
                i += 2
                continue
            elif arg.startswith("-k="):
                i += 1
                continue
            else:
                return f"Security violation: Unrecognized or unsafe pytest flag '{orig_arg}'."
        else:
            try:
                target = (workspace_dir / arg).resolve()
                target.relative_to(workspace_dir.resolve())
            except (ValueError, RuntimeError):
                return f"Security violation: Test target '{orig_arg}' escapes workspace boundary."
            i += 1
    return None


def _validate_pip_subargs(subargs: List[str], orig_subargs: List[str], workspace_dir: Path, project_root: Path) -> Optional[str]:
    """Validate arguments passed to pip."""
    if not subargs or subargs[0] != "install":
        return "Security violation: pip usage is restricted: only 'pip install' is permitted."

    packages_and_flags = subargs[1:]
    orig_p_and_f = orig_subargs[1:] if len(orig_subargs) > 1 else packages_and_flags
    if not packages_and_flags:
        return "Security violation: 'pip install' requires at least one package name."

    safe_pip_flags = {"--dry-run", "-q", "--quiet", "--no-deps", "--upgrade"}
    for idx, pkg in enumerate(packages_and_flags):
        orig_p = orig_p_and_f[idx] if idx < len(orig_p_and_f) else pkg

        # Check for forbidden flags
        for forbidden in ["-r", "-e", "--target", "--index-url", "--extra-index-url"]:
            if pkg == forbidden or pkg.startswith(f"{forbidden}="):
                return f"Security violation: Flag '{orig_p}' is forbidden for pip install."

        # Check for URLs and git
        if pkg.startswith("git+") or "://" in pkg:
            return f"Security violation: Remote package URL '{orig_p}' is forbidden for pip install."

        # Check for wheel and path files
        if pkg.endswith(".whl") or "/" in pkg or "\\" in orig_p or pkg.startswith("."):
            return f"Security violation: File path or wheel '{orig_p}' is forbidden for pip install."

        # Validate general arg safety
        gen_err = _validate_shell_argument(pkg, orig_p, workspace_dir, project_root)
        if gen_err:
            return gen_err

        if pkg.startswith("-"):
            if pkg not in safe_pip_flags:
                return f"Security violation: Unrecognized or unsafe pip flag '{orig_p}'."
        else:
            # Package name regex check
            if not re.match(r'^[a-zA-Z0-9_.-]+$', pkg):
                return f"Security violation: Invalid package name '{orig_p}' for pip install."

    return None


def _builtin_ls(target_path: Path) -> Dict[str, Any]:
    """Cross-platform directory lister for sandboxed workspace."""
    if not target_path.exists():
        return {
            "success": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": f"ls: cannot access '{target_path.name}': No such file or directory",
            "error": f"Path '{target_path.name}' does not exist.",
        }
    if target_path.is_file():
        return {
            "success": True,
            "exit_code": 0,
            "stdout": target_path.name,
            "stderr": "",
            "error": None,
        }
    entries = sorted([p.name for p in target_path.iterdir() if not p.name.startswith(".git")])
    return {
        "success": True,
        "exit_code": 0,
        "stdout": "\n".join(entries),
        "stderr": "",
        "error": None,
    }


def _builtin_cat(target_path: Path) -> Dict[str, Any]:
    """Cross-platform file reader for sandboxed workspace."""
    if not target_path.exists():
        return {
            "success": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": f"cat: {target_path.name}: No such file or directory",
            "error": f"File '{target_path.name}' does not exist.",
        }
    if target_path.is_dir():
        return {
            "success": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": f"cat: {target_path.name}: Is a directory",
            "error": f"Path '{target_path.name}' is a directory, not a file.",
        }
    try:
        content = target_path.read_text(encoding="utf-8", errors="replace")
        return {
            "success": True,
            "exit_code": 0,
            "stdout": content,
            "stderr": "",
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": str(exc),
            "error": f"Failed to read file '{target_path.name}': {exc}",
        }


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
    - Path escapes, traversal, shell operators, and sensitive files are strictly blocked.

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

    # 2. Check forbidden operators and traversal in raw command string
    for op in FORBIDDEN_OPERATORS:
        if op in command:
            return {
                "success": False,
                "command": command,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "execution_time_seconds": 0.0,
                "mode": "restricted_shell",
                "error": f"Security violation: Forbidden operator or character '{op}' in command.",
                "approval_required": False
            }

    if ".." in command:
        return {
            "success": False,
            "command": command,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "execution_time_seconds": 0.0,
            "mode": "restricted_shell",
            "error": "Security violation: Path traversal '..' is strictly forbidden.",
            "approval_required": False
        }

    # 3. Parse command arguments
    try:
        normalized_cmd = command.replace("\\", "/")
        args = shlex.split(normalized_cmd)
        raw_args = shlex.split(command, posix=False)
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

    # Check for dangerous command tokens
    for token in DANGEROUS_COMMAND_TOKENS:
        if token in args or (len(raw_args) > 0 and token == raw_args[0]):
            return {
                "success": False,
                "command": command,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "execution_time_seconds": 0.0,
                "mode": "restricted_shell",
                "error": f"Security violation: Command '{token}' is strictly forbidden.",
                "approval_required": False
            }

    base_cmd = Path(args[0]).name.lower()
    if base_cmd.endswith(".exe"):
        base_cmd = base_cmd[:-4]

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

    workspace_dir = cfg.workspace_dir
    project_root = workspace_dir.resolve().parent

    # Validate each argument with general security checks
    for idx, arg in enumerate(args[1:], start=1):
        orig_arg = raw_args[idx].strip('"').strip("'") if idx < len(raw_args) else arg
        arg_err = _validate_shell_argument(arg, orig_arg, workspace_dir, project_root)
        if arg_err:
            return {
                "success": False,
                "command": command,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "execution_time_seconds": 0.0,
                "mode": "restricted_shell",
                "error": arg_err,
                "approval_required": False
            }

    # 4. Command-specific validation and execution
    effective_timeout = timeout if timeout is not None else cfg.execution_timeout_seconds
    start_time = time.perf_counter()

    # --- LS ---
    if base_cmd == "ls":
        if len(args) == 1:
            target_path = workspace_dir.resolve()
        elif len(args) == 2:
            try:
                target_path = (workspace_dir / args[1]).resolve()
                target_path.relative_to(workspace_dir.resolve())
            except (ValueError, RuntimeError):
                return {
                    "success": False,
                    "command": command,
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                    "execution_time_seconds": 0.0,
                    "mode": "restricted_shell",
                    "error": f"Security violation: Path '{args[1]}' escapes workspace boundary.",
                    "approval_required": False
                }
        else:
            return {
                "success": False,
                "command": command,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "execution_time_seconds": 0.0,
                "mode": "restricted_shell",
                "error": "ls accepts at most one optional relative path.",
                "approval_required": False
            }

        ls_res = _builtin_ls(target_path)
        elapsed = time.perf_counter() - start_time
        res = {
            "success": ls_res["success"],
            "command": command,
            "exit_code": ls_res["exit_code"],
            "stdout": ls_res["stdout"],
            "stderr": ls_res["stderr"],
            "timed_out": False,
            "execution_time_seconds": round(elapsed, 4),
            "mode": "restricted_shell",
            "error": ls_res["error"],
            "approval_required": False
        }
        return _truncate_output(res, cfg.max_tool_output_chars)

    # --- CAT ---
    if base_cmd == "cat":
        if len(args) != 2:
            return {
                "success": False,
                "command": command,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "execution_time_seconds": 0.0,
                "mode": "restricted_shell",
                "error": "cat requires exactly one target file argument.",
                "approval_required": False
            }
        try:
            target_path = (workspace_dir / args[1]).resolve()
            target_path.relative_to(workspace_dir.resolve())
        except (ValueError, RuntimeError):
            return {
                "success": False,
                "command": command,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "execution_time_seconds": 0.0,
                "mode": "restricted_shell",
                "error": f"Security violation: Path '{args[1]}' escapes workspace boundary.",
                "approval_required": False
            }

        cat_res = _builtin_cat(target_path)
        elapsed = time.perf_counter() - start_time
        res = {
            "success": cat_res["success"],
            "command": command,
            "exit_code": cat_res["exit_code"],
            "stdout": cat_res["stdout"],
            "stderr": cat_res["stderr"],
            "timed_out": False,
            "execution_time_seconds": round(elapsed, 4),
            "mode": "restricted_shell",
            "error": cat_res["error"],
            "approval_required": False
        }
        return _truncate_output(res, cfg.max_tool_output_chars)

    # --- PIP ---
    if base_cmd == "pip":
        pip_err = _validate_pip_subargs(args[1:], raw_args[1:], workspace_dir, project_root)
        if pip_err:
            return {
                "success": False,
                "command": command,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "execution_time_seconds": 0.0,
                "mode": "restricted_shell",
                "error": pip_err,
                "approval_required": False
            }
        exec_args = [sys.executable, "-m", "pip"] + args[1:]

    # --- PYTEST ---
    elif base_cmd == "pytest":
        pytest_err = _validate_pytest_subargs(args[1:], raw_args[1:], workspace_dir, project_root)
        if pytest_err:
            return {
                "success": False,
                "command": command,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "execution_time_seconds": 0.0,
                "mode": "restricted_shell",
                "error": pytest_err,
                "approval_required": False
            }
        exec_args = [sys.executable, "-m", "pytest"] + args[1:]

    # --- PYTHON ---
    elif base_cmd == "python":
        if len(args) < 2:
            return {
                "success": False,
                "command": command,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "execution_time_seconds": 0.0,
                "mode": "restricted_shell",
                "error": "python requires a target script or '-m pytest'.",
                "approval_required": False
            }

        first_arg = args[1]
        orig_first = raw_args[1] if len(raw_args) > 1 else first_arg

        if first_arg == "-c":
            return {
                "success": False,
                "command": command,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "execution_time_seconds": 0.0,
                "mode": "restricted_shell",
                "error": "Security violation: Flag '-c' is forbidden for python.",
                "approval_required": False
            }

        if first_arg == "-":
            return {
                "success": False,
                "command": command,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "execution_time_seconds": 0.0,
                "mode": "restricted_shell",
                "error": "Security violation: Standard input '-' is forbidden for python.",
                "approval_required": False
            }

        if first_arg in {"-i", "-x"}:
            return {
                "success": False,
                "command": command,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "execution_time_seconds": 0.0,
                "mode": "restricted_shell",
                "error": f"Security violation: Flag '{orig_first}' is forbidden for python.",
                "approval_required": False
            }

        if first_arg == "-m":
            if len(args) < 3 or args[2] != "pytest":
                target_mod = args[2] if len(args) > 2 else ""
                return {
                    "success": False,
                    "command": command,
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                    "execution_time_seconds": 0.0,
                    "mode": "restricted_shell",
                    "error": f"Security violation: Only 'python -m pytest' is permitted. '-m {target_mod}' is forbidden.",
                    "approval_required": False
                }
            pytest_err = _validate_pytest_subargs(args[3:], raw_args[3:], workspace_dir, project_root)
            if pytest_err:
                return {
                    "success": False,
                    "command": command,
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                    "execution_time_seconds": 0.0,
                    "mode": "restricted_shell",
                    "error": pytest_err,
                    "approval_required": False
                }
            exec_args = [sys.executable, "-m", "pytest"] + args[3:]
        else:
            if first_arg.startswith("-"):
                return {
                    "success": False,
                    "command": command,
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                    "execution_time_seconds": 0.0,
                    "mode": "restricted_shell",
                    "error": f"Security violation: Unrecognized python flag '{orig_first}'.",
                    "approval_required": False
                }

            if not first_arg.endswith(".py"):
                return {
                    "success": False,
                    "command": command,
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                    "execution_time_seconds": 0.0,
                    "mode": "restricted_shell",
                    "error": f"Security violation: Target script '{orig_first}' must have a .py extension.",
                    "approval_required": False
                }

            try:
                script_path = (workspace_dir / first_arg).resolve()
                script_path.relative_to(workspace_dir.resolve())
            except (ValueError, RuntimeError):
                return {
                    "success": False,
                    "command": command,
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                    "execution_time_seconds": 0.0,
                    "mode": "restricted_shell",
                    "error": f"Security violation: Script '{orig_first}' escapes workspace boundary.",
                    "approval_required": False
                }

            if not script_path.exists():
                return {
                    "success": False,
                    "command": command,
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                    "execution_time_seconds": 0.0,
                    "mode": "restricted_shell",
                    "error": f"Script '{orig_first}' not found inside workspace.",
                    "approval_required": False
                }

            ast_err = check_python_script_safety(script_path)
            if ast_err:
                return {
                    "success": False,
                    "command": command,
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                    "execution_time_seconds": 0.0,
                    "mode": "restricted_shell",
                    "error": f"Security violation in '{orig_first}': {ast_err}",
                    "approval_required": False
                }

            exec_args = [sys.executable, str(script_path)] + args[2:]

    # 5. Dispatch subprocess execution with sanitized environment
    child_env = build_sanitized_environment(cfg.workspace_dir)

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

        res = {
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
        return _truncate_output(res, cfg.max_tool_output_chars)

    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start_time
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout.decode("utf-8", "replace") if exc.stdout else "")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr.decode("utf-8", "replace") if exc.stderr else "")

        res = {
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
        return _truncate_output(res, cfg.max_tool_output_chars)

    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        res = {
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
        return _truncate_output(res, cfg.max_tool_output_chars)


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

