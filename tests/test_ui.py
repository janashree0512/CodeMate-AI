"""
Unit tests for Streamlit UI Components (ui/ and app.py).
Validates style strings, status badge mapping, session state initialization,
and metric formatting without requiring a live browser.
"""

from unittest.mock import MagicMock, patch
import pytest

from config import Config, DEFAULT_MODEL
from ui.styles import CUSTOM_CSS
from ui.views import render_status_badge, render_metrics
import app


@pytest.fixture
def test_config(tmp_path):
    return Config(
        groq_api_key="mock",
        model_name=DEFAULT_MODEL,
        max_iterations=5,
        execution_timeout_seconds=5,
        require_user_approval=True,
        workspace_dir=tmp_path,
        project_root=tmp_path,
        max_tool_output_chars=4000,
        max_history_messages=30
    )


def test_custom_css_structure():
    assert "JetBrains Mono" in CUSTOM_CSS
    assert ".terminal-box" in CUSTOM_CSS
    assert ".badge-status" in CUSTOM_CSS
    assert ".diag-card" in CUSTOM_CSS


@patch("streamlit.markdown")
def test_render_status_badge(mock_markdown):
    render_status_badge("running")
    mock_markdown.assert_called_once()
    args, _ = mock_markdown.call_args
    assert "badge-running" in args[0]
    assert "STATUS: EXECUTING RE-ACT LOOP" in args[0]

    mock_markdown.reset_mock()
    render_status_badge("completed")
    args, _ = mock_markdown.call_args
    assert "badge-success" in args[0]
    assert "STATUS: TASK COMPLETED" in args[0]


@patch("streamlit.markdown")
@patch("streamlit.columns")
def test_render_metrics(mock_columns, mock_markdown):
    mock_cols = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    mock_columns.return_value = mock_cols

    render_metrics(iterations=3, max_iterations=10, correction_attempts=1, total_tokens=450)
    assert mock_markdown.call_count == 4


def test_init_session_state(test_config):
    mock_session = {}
    with patch("streamlit.session_state", mock_session):
        app.init_session_state(test_config)
        assert "tasks_history" in mock_session
        assert "current_result" in mock_session
        assert "agent_status" in mock_session
        assert mock_session["agent_status"] == "idle"


def test_missing_api_key_prevents_mock_run_in_workflow(test_config):
    """Test that run_agent_workflow never starts a mock run and raises or reports error on bad key."""
    bad_config = Config(
        groq_api_key="",
        model_name=DEFAULT_MODEL,
        max_iterations=5,
        execution_timeout_seconds=5,
        require_user_approval=True,
        workspace_dir=test_config.workspace_dir,
        project_root=test_config.project_root,
        max_tool_output_chars=4000,
        max_history_messages=30
    )
    mock_session = {
        "tasks_history": [],
        "current_result": "stale_result",
        "agent_status": "idle"
    }
    with patch("streamlit.session_state", mock_session), \
         patch("streamlit.error") as mock_err, \
         patch("streamlit.info"):
        app.run_agent_workflow(
            task="Compute primes",
            config=bad_config,
            runtime_settings={"model_name": DEFAULT_MODEL, "require_approval": False}
        )
        assert mock_session["agent_status"] == "failed"
        mock_err.assert_called_once()
        assert "Authentication Error" in mock_err.call_args[0][0]


def test_session_state_cleared_between_runs(test_config):
    """Verify that prior results are wiped from session state when a new run begins."""
    mock_session = {
        "tasks_history": [],
        "current_result": {"stale": "data"},
        "agent_status": "completed"
    }
    # Mock agent.run to inspect session_state right after run_agent_workflow initializes
    with patch("streamlit.session_state", mock_session), \
         patch("app.CodeMateAgent") as mock_agent_cls, \
         patch("streamlit.spinner"):
        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.iterations = 2
        mock_instance.run.return_value = mock_result
        mock_agent_cls.return_value = mock_instance

        app.run_agent_workflow(
            task="New fresh task",
            config=test_config,
            runtime_settings={"model_name": DEFAULT_MODEL, "require_approval": False}
        )
        # Should record the new task and result
        assert mock_session["current_result"] == mock_result
        assert mock_session["agent_status"] == "completed"
        assert len(mock_session["tasks_history"]) == 1
        assert mock_session["tasks_history"][0]["task"] == "New fresh task"
