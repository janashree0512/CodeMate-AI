"""
Unit tests for the configuration module (config.py).
Verifies safe defaults, environment variable overrides, input validation,
error handling, and workspace directory resolution.
"""

import os
from pathlib import Path
import pytest

from config import (
    Config,
    ConfigurationError,
    load_config,
    parse_bool,
    parse_positive_int,
    PROJECT_ROOT,
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
)


class TestConfigParsing:
    """Tests for individual parsing helper functions."""

    def test_parse_bool_defaults_and_values(self):
        # Default behavior
        assert parse_bool(None, default=True) is True
        assert parse_bool(None, default=False) is False

        # Truthy string values
        assert parse_bool("true") is True
        assert parse_bool("True") is True
        assert parse_bool("1") is True
        assert parse_bool("yes") is True
        assert parse_bool("t") is True

        # Falsy string values
        assert parse_bool("false") is False
        assert parse_bool("False") is False
        assert parse_bool("0") is False
        assert parse_bool("no") is False
        assert parse_bool("f") is False

    def test_parse_bool_invalid(self):
        with pytest.raises(ConfigurationError) as exc_info:
            parse_bool("not_a_bool")
        assert "Invalid boolean value" in str(exc_info.value)

    def test_parse_positive_int_valid(self):
        assert parse_positive_int("TEST_VAR", "10", default=5) == 10
        assert parse_positive_int("TEST_VAR", " 42 ", default=5) == 42
        assert parse_positive_int("TEST_VAR", None, default=7) == 7
        assert parse_positive_int("TEST_VAR", "", default=7) == 7

    def test_parse_positive_int_zero_or_negative(self):
        with pytest.raises(ConfigurationError) as exc_0:
            parse_positive_int("TEST_VAR", "0", default=5)
        assert "must be a positive integer (> 0)" in str(exc_0.value)

        with pytest.raises(ConfigurationError) as exc_neg:
            parse_positive_int("TEST_VAR", "-3", default=5)
        assert "must be a positive integer (> 0)" in str(exc_neg.value)

    def test_parse_positive_int_non_numeric(self):
        with pytest.raises(ConfigurationError) as exc_nan:
            parse_positive_int("TEST_VAR", "five", default=5)
        assert "must be a valid integer" in str(exc_nan.value)


class TestConfigLoading:
    """Tests for the load_config factory and Config dataclass."""

    def test_default_config(self, monkeypatch):
        # Clear any environment variables that might interfere
        for var in [
            "GROQ_API_KEY",
            "MODEL_NAME",
            "MAX_ITERATIONS",
            "EXECUTION_TIMEOUT_SECONDS",
            "REQUIRE_USER_APPROVAL",
            "WORKSPACE_DIR",
        ]:
            monkeypatch.delenv(var, raising=False)

        config = load_config(strict_api_key=False)

        assert config.model_name == DEFAULT_MODEL
        assert config.max_iterations == 10
        assert config.execution_timeout_seconds == 10
        assert config.require_user_approval is True
        assert config.workspace_dir == (PROJECT_ROOT / "workspace").resolve()
        assert config.workspace_dir.exists()
        assert config.project_root == PROJECT_ROOT

    def test_custom_environment_overrides(self, monkeypatch, tmp_path):
        custom_workspace = tmp_path / "custom_sandbox"
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key_123456789")
        monkeypatch.setenv("MODEL_NAME", "custom-override-model")
        monkeypatch.setenv("MAX_ITERATIONS", "15")
        monkeypatch.setenv("EXECUTION_TIMEOUT_SECONDS", "25")
        monkeypatch.setenv("REQUIRE_USER_APPROVAL", "false")
        monkeypatch.setenv("WORKSPACE_DIR", str(custom_workspace))

        config = load_config(strict_api_key=False)

        assert config.groq_api_key == "gsk_test_key_123456789"
        assert config.model_name == "custom-override-model"
        assert config.max_iterations == 15
        assert config.execution_timeout_seconds == 25
        assert config.require_user_approval is False
        assert config.workspace_dir == custom_workspace.resolve()
        assert config.workspace_dir.exists()

    def test_strict_api_key_validation_missing(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(ConfigurationError) as exc:
            load_config(strict_api_key=True)
        assert "Missing GROQ_API_KEY" in str(exc.value)

    def test_strict_api_key_validation_placeholder(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "your_groq_api_key_here")
        with pytest.raises(ConfigurationError) as exc:
            load_config(strict_api_key=True)
        assert "Missing GROQ_API_KEY" in str(exc.value)

    def test_strict_api_key_validation_valid(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_valid_live_key_xyz")
        config = load_config(strict_api_key=True)
        assert config.groq_api_key == "gsk_valid_live_key_xyz"

    def test_workspace_auto_creation(self, monkeypatch, tmp_path):
        nested_dir = tmp_path / "deep" / "nested" / "workspace"
        assert not nested_dir.exists()

        monkeypatch.setenv("WORKSPACE_DIR", str(nested_dir))
        config = load_config(strict_api_key=False)

        assert config.workspace_dir == nested_dir.resolve()
        assert config.workspace_dir.exists()


class TestModelNameConsistency:
    """Verifies that the LLM model name is unified across the codebase."""

    def test_default_model_constant_and_values(self):
        assert DEFAULT_MODEL == "openai/gpt-oss-120b"
        assert DEFAULT_MODEL in SUPPORTED_MODELS
        assert "openai/gpt-oss-20b" in SUPPORTED_MODELS
        assert "qwen/qwen3.8-27b" in SUPPORTED_MODELS

    def test_default_config_model_matches_constant(self, monkeypatch):
        monkeypatch.delenv("MODEL_NAME", raising=False)
        cfg = load_config(strict_api_key=False)
        assert cfg.model_name == DEFAULT_MODEL

    def test_no_hardcoded_model_names_in_source_files(self):
        """Fails if any Python source file other than config.py hardcodes a model string in executable code."""
        source_roots = [
            PROJECT_ROOT / "app.py",
            PROJECT_ROOT / "cli.py",
            PROJECT_ROOT / "ui",
            PROJECT_ROOT / "agent",
            PROJECT_ROOT / "tools",
        ]
        hardcoded_violations = []
        forbidden_substrings = [DEFAULT_MODEL, "llama-3.3-70b-versatile", "llama3-70b-8192"]

        for entry in source_roots:
            files = [entry] if entry.is_file() else list(entry.glob("**/*.py"))
            for py_file in files:
                if py_file.name == "config.py":
                    continue
                content = py_file.read_text(encoding="utf-8")
                for idx, line in enumerate(content.splitlines(), 1):
                    stripped = line.strip()
                    if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                        continue
                    for forbidden in forbidden_substrings:
                        if f'"{forbidden}"' in line or f"'{forbidden}'" in line:
                            hardcoded_violations.append(f"{py_file.name}:{idx} -> {line.strip()}")

        assert not hardcoded_violations, f"Found hardcoded model names in source files:\n" + "\n".join(hardcoded_violations)
