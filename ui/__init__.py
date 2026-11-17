"""
UI package for CodeMate AI.
Contains Streamlit styling, sidebar controls, and dashboard views.
"""

from ui.styles import apply_custom_styles
from ui.sidebar import render_sidebar
from ui.views import (
    render_status_badge,
    render_welcome_screen,
    render_metrics,
    render_terminal_output,
    render_diagnostics,
    render_tool_timeline,
    render_code_viewer,
    render_approval_card,
)

__all__ = [
    "apply_custom_styles",
    "render_sidebar",
    "render_status_badge",
    "render_welcome_screen",
    "render_metrics",
    "render_terminal_output",
    "render_diagnostics",
    "render_tool_timeline",
    "render_code_viewer",
    "render_approval_card",
]
