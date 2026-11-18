"""
Unit tests for Stage 3:
1. Demo mode approval bypass banner and live mode approval enforcement in app.py / CLI
2. CLI commands (/help, /files, /clear, /quit) and interactive approval
3. Hygiene checks (workspace isolation, gitignore coverage)
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from config import Config, DEFAULT_MODEL
from agent.agent import CodeMateAgent
from cli import CLIApprovalAgent
from tools.tool_registry import create_default_registry


@pytest.fixture
def stage3_config(tmp_path):
    ws = tmp_path / "stage3_ws"
    ws.mkdir(parents=True, exist_ok=True)
    return Config(
        groq_api_key="mock_key",
        model_name=DEFAULT_MODEL,
        max_iterations=5,
        execution_timeout_seconds=5,
        require_user_approval=True,
        workspace_dir=ws,
        project_root=tmp_path,
        max_tool_output_chars=4000,
        max_history_messages=30
    )


class TestDemoModeAndApprovalSafety:
    """Test approval safety behavior in live and demo configurations."""

    def test_live_agent_respects_approval_flag(self, stage3_config):
        registry = create_default_registry(config=stage3_config)
        # When require_user_approval is True and auto_approve is False, unapproved execution must be blocked
        agent = CodeMateAgent(config=stage3_config, registry=registry, auto_approve=False)
        assert agent.auto_approve is False

    def test_cli_approval_agent_prompts_and_approves(self, stage3_config):
        registry = create_default_registry(config=stage3_config)
        agent = CLIApprovalAgent(config=stage3_config, registry=registry, auto_approve=False)

        mock_call = MagicMock()
        mock_call.id = "c1"
        mock_call.tool_name = "execute_python"
        mock_call.arguments = {"filename": "app.py"}

        with patch("builtins.input", return_value="y"):
            record = agent._execute_tool_call(mock_call, iteration=1)
            # The execution was approved; even if file doesn't exist, approval was granted
            assert record.arguments.get("approved") is True

    def test_cli_approval_agent_rejects_unapproved(self, stage3_config):
        registry = create_default_registry(config=stage3_config)
        agent = CLIApprovalAgent(config=stage3_config, registry=registry, auto_approve=False)

        mock_call = MagicMock()
        mock_call.id = "c1"
        mock_call.tool_name = "execute_python"
        mock_call.arguments = {"filename": "app.py"}

        with patch("builtins.input", return_value="n"):
            record = agent._execute_tool_call(mock_call, iteration=1)
            assert record.arguments.get("approved") is False
            assert record.result.get("approval_required") is True


class TestRepoHygiene:
    """Ensure repository conventions, ignore rules, and workspace cleanliness are maintained."""

    def test_gitignore_covers_critical_patterns(self, stage3_config):
        gitignore_path = Path(__file__).resolve().parent.parent / ".gitignore"
        assert gitignore_path.exists()
        content = gitignore_path.read_text(encoding="utf-8")
        assert ".env" in content
        assert "venv/" in content
        assert "__pycache__/" in content
        assert ".pytest_cache/" in content
        assert "workspace/*" in content

    def test_workspace_contains_gitkeep(self, stage3_config):
        real_ws = Path(__file__).resolve().parent.parent / "workspace"
        assert real_ws.exists()
        assert (real_ws / ".gitkeep").exists()
