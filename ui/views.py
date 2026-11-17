"""
CodeMate AI - UI View Components (ui/views.py)
==============================================

Renders developer dashboard views:
- Welcome screen & architecture overview
- Execution status badges & telemetry metrics
- Tool execution timeline
- Terminal console output
- Error diagnostics panel
- Workspace code viewer with syntax highlighting
- Explicit Human-in-the-Loop approval gate
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit as st


def render_status_badge(status: str) -> None:
    """Render a clean status pill."""
    status_map = {
        "idle": ("badge-idle", "STATUS: IDLE"),
        "running": ("badge-running", "STATUS: EXECUTING RE-ACT LOOP"),
        "awaiting_approval": ("badge-approval", "STATUS: AWAITING USER APPROVAL"),
        "completed": ("badge-success", "STATUS: TASK COMPLETED"),
        "failed": ("badge-failed", "STATUS: EXECUTION HALTED"),
    }
    css_class, label = status_map.get(status.lower(), ("badge-idle", f"STATUS: {status.upper()}"))
    st.markdown(f'<span class="badge-status {css_class}">{label}</span>', unsafe_allow_html=True)


def render_welcome_screen(on_example_selected) -> None:
    """Render the welcome view when no task is running."""
    st.markdown("""
    ### Autonomous AI-Powered Coding Agent
    CodeMate AI operates via a closed-loop **ReAct (Reason + Act + Observe)** execution architecture.
    It takes high-level programming instructions, formulates an execution plan, writes code into a sandboxed workspace,
    runs the code in a restricted subprocess environment, detects errors, and autonomously corrects failures.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        **1. Planning**  
        Deconstructs requirements into verifiable tasks.
        """)
    with col2:
        st.markdown("""
        **2. Tool Calling**  
        First-principles file creation and editing in `workspace/`.
        """)
    with col3:
        st.markdown("""
        **3. Subprocess Sandbox**  
        Isolated execution with timeouts and credential protection.
        """)
    with col4:
        st.markdown("""
        **4. Error Healing**  
        Categorizes syntax and runtime bugs and re-runs fixes.
        """)

    st.divider()
    st.markdown("##### Quick Start Tasks")

    examples = [
        "Create a Python script named 'primes.py' that computes primes up to 50, execute it, and verify the output.",
        "Implement a bubble sort algorithm in 'sorter.py' with test assertions for sorted, reversed, and empty lists.",
        "Write a script 'stats.py' that computes the mean, variance, and standard deviation of [12, 15, 18, 22, 30].",
    ]

    for ex in examples:
        if st.button(f"➔ {ex}", key=f"ex_{hash(ex)}"):
            on_example_selected(ex)


def render_metrics(
    iterations: int,
    max_iterations: int,
    correction_attempts: int,
    total_tokens: int
) -> None:
    """Render horizontal telemetry metric pills."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-pill">
            <div class="metric-pill-label">ReAct Iterations</div>
            <div class="metric-pill-val">{iterations} / {max_iterations}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-pill">
            <div class="metric-pill-label">Correction Attempts</div>
            <div class="metric-pill-val">{correction_attempts}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-pill">
            <div class="metric-pill-label">Total Tokens</div>
            <div class="metric-pill-val">{total_tokens}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-pill">
            <div class="metric-pill-label">Environment</div>
            <div class="metric-pill-val">RESTRICTED LOCAL</div>
        </div>
        """, unsafe_allow_html=True)


def render_terminal_output(
    stdout: str,
    stderr: str,
    exit_code: Optional[int],
    timed_out: bool = False
) -> None:
    """Render a styled terminal box displaying raw subprocess output."""
    status_str = f"EXIT: {exit_code}" if exit_code is not None else "STATUS: PENDING"
    if timed_out:
        status_str = "TERMINATED (TIMEOUT EXPIRED)"

    body = ""
    if stdout:
        body += stdout
    if stderr:
        if body:
            body += "\n" + "-" * 40 + " [STDERR TRACEBACK] " + "-" * 40 + "\n"
        body += stderr

    if not body.strip():
        body = "[No stream output generated]"

    st.markdown(f"""
    <div class="terminal-box">
        <div class="terminal-header">
            <span>Terminal Output</span>
            <span>{status_str}</span>
        </div>
        <div>{body}</div>
    </div>
    """, unsafe_allow_html=True)


def render_diagnostics(diagnostics: Dict[str, Any]) -> None:
    """Render a structured error diagnostic card."""
    cat = diagnostics.get("category", "error").upper()
    name = diagnostics.get("exception_name", "Exception")
    line = diagnostics.get("failed_line")
    loc = f" at line {line}" if line else ""
    msg = diagnostics.get("exception_message", "")
    sug = diagnostics.get("actionable_suggestion", "")

    st.markdown(f"""
    <div class="diag-card">
        <div class="diag-title">[{cat}] {name}{loc}</div>
        <div class="diag-body">
            <strong>Detail:</strong> {msg}<br/>
            <strong>Actionable Guidance:</strong> {sug}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_tool_timeline(action_history: List[Any]) -> None:
    """Render chronological tool execution records."""
    if not action_history:
        st.caption("No tool executions recorded yet.")
        return

    for idx, action in enumerate(action_history, 1):
        status_color = "🟢" if action.success else "🔴"
        with st.expander(f"{status_color} Iteration {action.iteration}: Tool `{action.tool_name}`", expanded=False):
            st.markdown(f"**Arguments:** `{action.arguments}`")
            if action.tool_name == "execute_python":
                render_terminal_output(
                    stdout=action.result.get("stdout", ""),
                    stderr=action.result.get("stderr", ""),
                    exit_code=action.result.get("exit_code"),
                    timed_out=action.result.get("timed_out", False)
                )
                if action.result.get("diagnostics"):
                    render_diagnostics(action.result["diagnostics"])
            else:
                st.json(action.result)


def render_code_viewer(workspace_dir: Path) -> None:
    """Render a syntax-highlighted code inspector for files residing in workspace/."""
    files = [f for f in sorted(workspace_dir.rglob("*.py")) if f.is_file()]

    if not files:
        st.info("No Python files in workspace yet.")
        return

    rel_paths = [str(f.relative_to(workspace_dir)).replace("\\", "/") for f in files]
    selected = st.selectbox("Inspect Workspace Script", rel_paths, label_visibility="collapsed")

    if selected:
        target_path = workspace_dir / selected
        if target_path.exists():
            content = target_path.read_text(encoding="utf-8")
            st.code(content, language="python", line_numbers=True)
            st.caption(f"Path: `workspace/{selected}` | Size: {target_path.stat().st_size} bytes")


def render_approval_card(tool_name: str, args: Dict[str, Any], on_approve, on_reject) -> None:
    """Render an explicit Human-in-the-Loop approval gate."""
    st.warning("⚠️ **Approval Required Before Execution**")
    st.markdown(f"""
    The agent requests to invoke tool **`{tool_name}`** with arguments:
    ```json
    {args}
    ```
    Do you authorize this operation in the sandboxed workspace?
    """)
    col1, col2, _ = st.columns([1, 1, 3])
    with col1:
        if st.button("✅ Approve & Proceed", type="primary", use_container_width=True):
            on_approve()
    with col2:
        if st.button("❌ Deny Action", type="secondary", use_container_width=True):
            on_reject()
