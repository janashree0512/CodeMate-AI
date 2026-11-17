"""
CodeMate AI - UI Sidebar Component (ui/sidebar.py)
==================================================

Renders the professional developer sidebar:
- CodeMate AI Branding
- New Task controller
- Past Task Run History
- Workspace File Explorer
- Configurable Agent Settings (Model, Iterations, Timeouts, Approvals)
"""

from pathlib import Path
from typing import Any, Callable, Dict, List
import streamlit as st
from config import Config


def render_sidebar(
    config: Config,
    history: List[Dict[str, Any]],
    workspace_dir: Path,
    on_new_task: Callable[[], None],
    on_select_history: Callable[[int], None]
) -> Dict[str, Any]:
    """
    Render the sidebar and return modified runtime settings.
    """
    with st.sidebar:
        # Branding
        st.markdown("### 🤖 CodeMate AI")
        st.caption("Autonomous Coding Agent (ReAct Loop)")

        # New Task Button
        if st.button("➕ New Task", use_container_width=True, type="primary"):
            on_new_task()

        st.divider()

        # Workspace Files Explorer
        st.markdown("##### 📁 Workspace Files")
        py_files = list(workspace_dir.glob("*.py"))
        if not py_files:
            st.caption("Workspace is currently empty.")
        else:
            for pf in sorted(py_files):
                size = pf.stat().st_size
                st.markdown(f"- `{pf.name}` <span style='color: #6B7280; font-size: 0.75rem;'>({size}B)</span>", unsafe_allow_html=True)

        st.divider()

        # Past Runs History
        st.markdown("##### 📜 Recent Tasks")
        if not history:
            st.caption("No past tasks recorded.")
        else:
            for idx, item in enumerate(reversed(history)):
                real_idx = len(history) - 1 - idx
                status_icon = "🟢" if item.get("success") else "🔴"
                task_snippet = item.get("task", "Task")[:30] + "..."
                if st.button(f"{status_icon} {task_snippet}", key=f"hist_{real_idx}", use_container_width=True):
                    on_select_history(real_idx)

        st.divider()

        # Agent Settings & Boundaries
        st.markdown("##### ⚙️ Runtime Settings")

        model_options = [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama3-70b-8192"
        ]
        selected_model = st.selectbox(
            "Model",
            model_options,
            index=0 if config.model_name not in model_options else model_options.index(config.model_name)
        )

        max_iter = st.slider(
            "Max Iterations",
            min_value=3,
            max_value=20,
            value=config.max_iterations,
            step=1
        )

        timeout = st.slider(
            "Execution Timeout (s)",
            min_value=2,
            max_value=30,
            value=config.execution_timeout_seconds,
            step=1
        )

        require_approval = st.checkbox(
            "Require Manual Approval Before Execution",
            value=config.require_user_approval
        )

        # API Key Status (Never prints or exposes secret key)
        st.divider()
        has_key = bool(config.groq_api_key and config.groq_api_key != "your_groq_api_key_here")
        if has_key:
            st.success("API Key: Configured in .env")
        else:
            st.warning("API Key: Not Found (Offline Mock Mode Available)")

        return {
            "model_name": selected_model,
            "max_iterations": max_iter,
            "timeout": timeout,
            "require_approval": require_approval
        }
