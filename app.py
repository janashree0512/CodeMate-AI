"""
CodeMate AI - Streamlit User Interface (app.py)
==============================================

Frontend dashboard for CodeMate AI.
Displays task input, real-time action summaries, tool execution logs,
workspace file previews, and human-in-the-loop approval dialogs.

Note:
Internal model chain-of-thought is kept private; only action plans,
tool calls, execution outputs, and user summaries are displayed.
"""

import streamlit as st
from config import load_config
from tools.tool_registry import create_default_registry


def render_ui():
    """Render the primary Streamlit interface."""
    st.set_page_config(
        page_title="CodeMate AI - Autonomous Coding Agent",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 CodeMate AI")
    st.caption("Autonomous AI-Powered Coding Agent (ReAct Loop & Self-Healing Sandbox)")

    # Load configuration
    config = load_config(strict_api_key=False)

    # Sidebar: System configuration and parameters
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info(f"**Model:** {config.model_name}")
        st.write(f"**Max Iterations:** {config.max_iterations}")
        st.write(f"**Execution Timeout:** {config.execution_timeout_seconds}s")
        st.write(f"**User Approval:** {'Enabled' if config.require_user_approval else 'Disabled'}")
        st.write(f"**Workspace:** `{config.workspace_dir.name}/`")

        st.divider()
        st.subheader("🛠️ Available Tools")
        registry = create_default_registry(config=config)
        for tool in registry.list_tools():
            st.markdown(f"- **`{tool['name']}`**: {tool['description'][:60]}...")

    # Main column layout
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("💬 Task Instruction")
        user_input = st.text_area(
            "Enter a programming task:",
            placeholder="e.g., Create a script that generates prime numbers up to 100, save it, and verify the output."
        )
        if st.button("🚀 Start Agent", type="primary"):
            if not user_input.strip():
                st.warning("Please provide a task description.")
            else:
                st.success(f"Task submitted: {user_input[:50]}...")
                st.info("Agent controller loop will run here upon full activation.")

    with col2:
        st.subheader("📁 Workspace Files")
        st.caption("Live files residing in the sandboxed workspace.")
        files_result = registry.dispatch("list_files")
        if files_result.get("success"):
            entries = files_result.get("entries", [])
            if not entries:
                st.text("Workspace is currently empty.")
            else:
                for entry in entries:
                    st.text(f"• {entry['path']} ({entry['size_bytes']} bytes)")
        else:
            st.error(files_result.get("error"))


if __name__ == "__main__":
    render_ui()
