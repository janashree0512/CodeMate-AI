"""
CodeMate AI - Professional Developer Dashboard (app.py)
======================================================

Modern Streamlit interface for CodeMate AI.
Features real-time ReAct loop monitoring, workspace file inspection,
terminal output capture, error diagnostics, and Human-in-the-Loop execution gates.
"""

import sys
import streamlit as st
from pathlib import Path
from typing import Any, Dict, Optional

from config import Config, load_config
from agent.agent import CodeMateAgent, AgentResult, run_agent
from agent.llm_client import GroqClient, LLMResponse, ToolCallRequest, LLMAuthenticationError, LLMClientError
from tools.tool_registry import create_default_registry
from ui import (
    apply_custom_styles,
    render_sidebar,
    render_status_badge,
    render_welcome_screen,
    render_metrics,
    render_terminal_output,
    render_diagnostics,
    render_tool_timeline,
    render_code_viewer,
    render_approval_card,
)


def init_session_state(config: Config):
    """Initialize persistent Streamlit session state."""
    if "tasks_history" not in st.session_state:
        st.session_state["tasks_history"] = []
    if "current_result" not in st.session_state:
        st.session_state["current_result"] = None
    if "task_input" not in st.session_state:
        st.session_state["task_input"] = ""
    if "agent_status" not in st.session_state:
        st.session_state["agent_status"] = "idle"
    if "pending_approval" not in st.session_state:
        st.session_state["pending_approval"] = None
    if "selected_example" not in st.session_state:
        st.session_state["selected_example"] = None


def run_agent_workflow(
    task: str,
    config: Config,
    runtime_settings: Dict[str, Any]
):
    """
    Execute the agent ReAct loop with real GroqClient, progress indication, and error boundaries.
    Clears any prior results before running.
    """
    # Fix stale results: Clear previous run data from session state
    st.session_state["current_result"] = None
    st.session_state["agent_status"] = "running"

    # Merge runtime settings with base configuration
    effective_config = Config(
        groq_api_key=config.groq_api_key,
        model_name=runtime_settings.get("model_name", config.model_name),
        max_iterations=runtime_settings.get("max_iterations", config.max_iterations),
        execution_timeout_seconds=runtime_settings.get("timeout", config.execution_timeout_seconds),
        require_user_approval=runtime_settings.get("require_approval", config.require_user_approval),
        max_tool_output_chars=config.max_tool_output_chars,
        max_history_messages=config.max_history_messages,
        workspace_dir=config.workspace_dir,
        project_root=config.project_root
    )
    st.session_state["last_run_model"] = effective_config.model_name

    registry = create_default_registry(config=effective_config)

    # Always use the real agent with GroqClient
    agent = CodeMateAgent(
        config=effective_config,
        registry=registry,
        auto_approve=not effective_config.require_user_approval
    )

    try:
        with st.spinner(f"Agent ({effective_config.model_name}) is planning, generating code, and verifying in the sandbox..."):
            result: AgentResult = agent.run(task=task)

        st.session_state["current_result"] = result
        st.session_state["agent_status"] = "completed" if result.success else "failed"

        # Record into task history
        st.session_state["tasks_history"].append({
            "task": task,
            "success": result.success,
            "iterations": result.iterations,
            "result": result
        })

    except LLMAuthenticationError as err:
        st.session_state["agent_status"] = "failed"
        st.error(f"**Groq Authentication Error**: {err}")
        st.info("Please set your valid `GROQ_API_KEY` in the `.env` file to use CodeMate AI.")

    except Exception as err:
        st.session_state["agent_status"] = "failed"
        st.error(f"**Agent Error**: {err}")


def main():
    st.set_page_config(
        page_title="CodeMate AI - Autonomous Coding Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply Developer Theme CSS
    apply_custom_styles()

    # Load configuration
    base_config = load_config(strict_api_key=False)
    init_session_state(base_config)

    # Callbacks for sidebar
    def handle_new_task():
        st.session_state.current_result = None
        st.session_state.task_input = ""
        st.session_state.agent_status = "idle"
        st.session_state.pending_approval = None
        st.session_state.selected_example = None
        st.rerun()

    def handle_select_history(index: int):
        selected_record = st.session_state.tasks_history[index]
        st.session_state.current_result = selected_record["result"]
        st.session_state.task_input = selected_record["task"]
        st.session_state.agent_status = "completed" if selected_record["success"] else "failed"
        st.rerun()

    # Render Sidebar
    runtime_settings = render_sidebar(
        config=base_config,
        history=st.session_state.tasks_history,
        workspace_dir=base_config.workspace_dir,
        on_new_task=handle_new_task,
        on_select_history=handle_select_history
    )

    # Main Dashboard Header
    title_col, badge_col = st.columns([3, 1])
    with title_col:
        st.markdown("## 🤖 CodeMate AI")
        st.caption("Autonomous Software Engineering Agent | Closed-Loop ReAct Engine")
    with badge_col:
        st.markdown("<div style='text-align: right; margin-top: 10px;'>", unsafe_allow_html=True)
        render_status_badge(st.session_state.agent_status)
        st.markdown("</div>", unsafe_allow_html=True)

    # Task Input Area
    def on_example_click(text: str):
        st.session_state.task_input = text
        st.rerun()

    current_val = st.session_state.task_input
    user_prompt = st.text_area(
        "Programming Task Instruction",
        value=current_val,
        placeholder="Describe the programming problem to solve, including target files, logic, and test verifications...",
        height=95
    )

    # Action Buttons Row
    btn_col1, btn_col2, _ = st.columns([1.5, 1.2, 5])

    with btn_col1:
        run_live_clicked = st.button("🚀 Run Agent", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("Clear", use_container_width=True):
            handle_new_task()

    # Trigger Agent Execution
    if run_live_clicked:
        # Re-read fresh config from .env on every run
        fresh_config = load_config(strict_api_key=False)
        target_env = fresh_config.project_root / ".env"

        if not user_prompt.strip():
            st.warning("Please provide a programming task description.")
        elif not target_env.exists():
            st.error(f".env not found at {target_env.resolve()}")
        elif not fresh_config.groq_api_key or fresh_config.groq_api_key == "your_groq_api_key_here":
            st.error("GROQ_API_KEY is empty or still the placeholder in .env")
        elif not fresh_config.groq_api_key.startswith("gsk_"):
            masked = f"{fresh_config.groq_api_key[:4]}...{fresh_config.groq_api_key[-4:]}" if len(fresh_config.groq_api_key) > 8 else "[INVALID]"
            st.error(f"Key should start with gsk_ (check for a copy/paste mistake; found {masked})")
        else:
            run_agent_workflow(user_prompt, fresh_config, runtime_settings)
            st.rerun()

    st.divider()

    # Content Area
    current_result: Optional[AgentResult] = st.session_state.current_result

    if current_result is None:
        # Show Welcome & Architecture Screen when idle
        render_welcome_screen(on_example_selected=on_example_click)

    else:
        # Header Badge displaying Real Model Information used for this run
        active_model = st.session_state.get("last_run_model", runtime_settings.get("model_name", base_config.model_name))
        st.markdown(
            f"<div style='margin-bottom: 15px; padding: 6px 12px; background-color: #1E293B; "
            f"border-left: 4px solid #3B82F6; border-radius: 4px; font-size: 0.85rem; color: #94A3B8;'>"
            f"🔵 <strong>LIVE RUN:</strong> Model: <code style='color: #60A5FA;'>{active_model}</code> &nbsp;|&nbsp; "
            f"Task: <em>{current_result.task[:70]}{'...' if len(current_result.task) > 70 else ''}</em>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Render Telemetry Metrics Bar
        render_metrics(
            iterations=current_result.iterations,
            max_iterations=current_result.max_iterations,
            correction_attempts=current_result.correction_attempts,
            total_tokens=current_result.total_tokens
        )

        st.markdown("<br/>", unsafe_allow_html=True)

        # Two-Column Developer Dashboard
        col_left, col_right = st.columns([3, 2])

        with col_left:
            # 1. Final Verified Response
            st.markdown("##### 📝 Final Agent Response")
            if current_result.success:
                st.success(current_result.final_response)
            else:
                st.error(current_result.final_response)

            # 2. Tool Execution Timeline
            st.markdown("##### ⏱️ Tool Execution History")
            render_tool_timeline(current_result.action_history)

        with col_right:
            # 1. Workspace Code Viewer
            st.markdown("##### 💻 Generated Code")
            render_code_viewer(base_config.workspace_dir)

            # 2. Latest Terminal Stream
            st.markdown("##### 🖥️ Latest Execution Stream")
            if current_result.execution_history:
                last_exec = current_result.execution_history[-1]
                render_terminal_output(
                    stdout=last_exec.get("stdout") or "",
                    stderr=last_exec.get("stderr") or "",
                    exit_code=last_exec.get("exit_code"),
                    timed_out=last_exec.get("timed_out", False)
                )

                # 3. Diagnostics Panel (if execution had errors)
                if last_exec.get("diagnostics"):
                    st.markdown("##### 🔍 Error Diagnostics")
                    render_diagnostics(last_exec["diagnostics"])
            else:
                st.caption("No executions performed during this task.")


if __name__ == "__main__":
    main()
