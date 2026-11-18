"""
Unit tests for Streamlit UI Components (ui/ and app.py).
Validates style strings, status badge mapping, session state initialization,
and metric formatting without requiring a live browser.
"""

from unittest.mock import MagicMock, patch
import pytest

from config import Config
from ui.styles import CUSTOM_CSS
from ui.views import render_status_badge, render_metrics
import app


@pytest.fixture
def test_config(tmp_path):
    return Config(
        groq_api_key="mock",
        model_name="llama-3.3-70b-versatile",
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
