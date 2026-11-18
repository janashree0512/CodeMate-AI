"""
CodeMate AI - Configuration Module
==================================

This module provides centralized, validated configuration for the application.
It loads variables from the environment and '.env' file, validates limits and paths,
and ensures that the workspace directory is properly created and isolated.

Design Principles:
- Single Source of Truth: All modules import runtime settings from this module.
- Fail Fast: Invalid types or non-positive bounds raise explicit ConfigurationError.
- Safe Defaults: Sensible limits protect against infinite loops and process hangs.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when environment variables or application settings are invalid."""
    pass


# Base project root directory (absolute path to CodeMate_AI)
PROJECT_ROOT: Path = Path(__file__).resolve().parent


def parse_bool(value: Optional[str], default: bool = True) -> bool:
    """
    Safely parse common string representations of booleans.
    
    Accepts: 'true', '1', 'yes', 't' -> True
             'false', '0', 'no', 'f' -> False
    """
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "t"}:
        return True
    if normalized in {"false", "0", "no", "f"}:
        return False
    raise ConfigurationError(
        f"Invalid boolean value '{value}'. Expected 'true', 'false', '1', or '0'."
    )


def parse_positive_int(var_name: str, value: Optional[str], default: int) -> int:
    """
    Safely parse an integer environment variable, ensuring it is strictly positive (> 0).
    """
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value.strip())
        if parsed <= 0:
            raise ConfigurationError(
                f"Configuration setting '{var_name}' must be a positive integer (> 0). Received: {parsed}"
            )
        return parsed
    except ValueError as exc:
        raise ConfigurationError(
            f"Configuration setting '{var_name}' must be a valid integer. Received: '{value}'"
        ) from exc


PLACEHOLDER_KEY = "your_groq_api_key_here"

DEFAULT_MODEL: str = "openai/gpt-oss-120b"
SUPPORTED_MODELS: list[str] = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]


def mask_key(key: Optional[str]) -> str:
    """Safely mask an API key, showing only the first 4 and last 4 characters."""
    if not key or key == PLACEHOLDER_KEY:
        return "[NOT SET]"
    if len(key) <= 8:
        return "[SET]"
    return f"{key[:4]}...{key[-4:]}"


def validate_api_key_string(key: str, env_path: Optional[Path] = None) -> None:
    """
    Validate the format and presence of the Groq API key with specific error messages.
    """
    if env_path is not None and not env_path.exists():
        raise ConfigurationError(f".env not found at {env_path.resolve()}")

    if not key or key == PLACEHOLDER_KEY:
        raise ConfigurationError(
            "Missing GROQ_API_KEY: GROQ_API_KEY is empty or still the placeholder in .env"
        )

    if not key.startswith("gsk_") and not key.startswith("mock"):
        masked = mask_key(key)
        raise ConfigurationError(
            f"Key should start with gsk_ (check for a copy/paste mistake; found {masked})"
        )


@dataclass(frozen=True)
class Config:
    """
    Immutable application configuration container.
    """
    # LLM Settings
    groq_api_key: str
    model_name: str

    # Agent Loop Boundaries
    max_iterations: int
    execution_timeout_seconds: int

    # Safety Controls
    require_user_approval: bool

    # Output / history limits (Stage 2)
    max_tool_output_chars: int      # Truncate tool output sent to LLM
    max_history_messages: int       # Cap total conversation messages

    # Sandboxed Workspace Path
    workspace_dir: Path

    # Root Path
    project_root: Path

    def ensure_workspace(self) -> Path:
        """
        Guarantees that the sandboxed workspace directory exists on disk.
        Returns the absolute Path.
        """
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        return self.workspace_dir

    def validate_api_key(self) -> None:
        """
        Checks that a valid Groq API key is present and well-formed.
        Raises ConfigurationError if the key is missing, placeholder, or invalid.
        """
        validate_api_key_string(self.groq_api_key)


def load_config(env_file: Optional[Path | str] = None, strict_api_key: bool = False) -> Config:
    """
    Load, parse, and validate configuration settings.
    Always uses an absolute path and override=True so changes to .env take effect immediately.

    Parameters:
    - env_file: Optional path to an alternate '.env' file.
    - strict_api_key: If True, immediately validates that GROQ_API_KEY is populated and valid.
                      If False, allows loading without an API key (useful for offline unit testing).

    Returns:
    - Config: A frozen, validated configuration instance.
    """
    # 1. Load environment variables from .env using an absolute path with override=True
    if env_file:
        resolved_env = Path(env_file).resolve()
        if not resolved_env.exists():
            raise ConfigurationError(f".env not found at {resolved_env}")
        load_dotenv(dotenv_path=resolved_env, override=True)
    elif "PYTEST_CURRENT_TEST" not in os.environ:
        default_env = (PROJECT_ROOT / ".env").resolve()
        if default_env.exists():
            load_dotenv(dotenv_path=default_env, override=True)

    # 2. Parse and sanitize LLM settings (strip whitespace, quotes, carriage returns, BOM)
    # If a UTF-8 BOM in .env created a \ufeff-prefixed key in os.environ, migrate it and clean it up
    for bom_key in [k for k in os.environ if k.startswith("\ufeff")]:
        clean_k = bom_key.replace("\ufeff", "").strip()
        val = os.environ.pop(bom_key)
        if clean_k not in os.environ:
            os.environ[clean_k] = val

    raw_key = os.getenv("GROQ_API_KEY", "")
    groq_api_key = raw_key.replace("\r", "").replace("\ufeff", "").strip().strip('"').strip("'")

    raw_model = os.getenv("MODEL_NAME", DEFAULT_MODEL)
    model_name = raw_model.replace("\r", "").replace("\ufeff", "").strip().strip('"').strip("'")
    if not model_name:
        model_name = DEFAULT_MODEL

    # 3. Parse and validate numerical execution boundaries
    max_iterations = parse_positive_int(
        "MAX_ITERATIONS",
        os.getenv("MAX_ITERATIONS"),
        default=10
    )

    execution_timeout_seconds = parse_positive_int(
        "EXECUTION_TIMEOUT_SECONDS",
        os.getenv("EXECUTION_TIMEOUT_SECONDS"),
        default=10
    )

    # 4. Parse safety switches
    require_user_approval = parse_bool(
        os.getenv("REQUIRE_USER_APPROVAL"),
        default=True
    )

    # 5a. Parse output / history limits (Stage 2)
    max_tool_output_chars = parse_positive_int(
        "MAX_TOOL_OUTPUT_CHARS",
        os.getenv("MAX_TOOL_OUTPUT_CHARS"),
        default=4000
    )
    max_history_messages = parse_positive_int(
        "MAX_HISTORY_MESSAGES",
        os.getenv("MAX_HISTORY_MESSAGES"),
        default=30
    )

    # 5. Resolve workspace directory path
    raw_workspace = os.getenv("WORKSPACE_DIR", "workspace").strip()
    workspace_path = Path(raw_workspace)
    if not workspace_path.is_absolute():
        workspace_path = (PROJECT_ROOT / workspace_path).resolve()
    else:
        workspace_path = workspace_path.resolve()

    # 6. Instantiate configuration object
    config = Config(
        groq_api_key=groq_api_key,
        model_name=model_name,
        max_iterations=max_iterations,
        execution_timeout_seconds=execution_timeout_seconds,
        require_user_approval=require_user_approval,
        max_tool_output_chars=max_tool_output_chars,
        max_history_messages=max_history_messages,
        workspace_dir=workspace_path,
        project_root=PROJECT_ROOT
    )

    # 7. Auto-create workspace folder on configuration load
    config.ensure_workspace()

    # 8. Strict validation check if requested
    if strict_api_key:
        config.validate_api_key()

    return config
