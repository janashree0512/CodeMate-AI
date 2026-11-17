"""
CodeMate AI - UI Theme & Styling (ui/styles.py)
==============================================

Professional developer dark theme CSS styles.
Monospaced terminals, high-contrast badges, clean borders, and zero childish animations.
"""

CUSTOM_CSS = """
<style>
    /* Global Typography & Font Family */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre, .terminal-text {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 95rem;
    }

    /* Status Badges */
    .badge-status {
        display: inline-block;
        padding: 4px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-success {
        background-color: #064E3B;
        color: #34D399;
        border: 1px solid #059669;
    }
    .badge-failed {
        background-color: #7F1D1D;
        color: #F87171;
        border: 1px solid #DC2626;
    }
    .badge-running {
        background-color: #1E3A8A;
        color: #60A5FA;
        border: 1px solid #3B82F6;
    }
    .badge-idle {
        background-color: #334155;
        color: #94A3B8;
        border: 1px solid #475569;
    }
    .badge-approval {
        background-color: #78350F;
        color: #FBBF24;
        border: 1px solid #D97706;
    }

    /* Terminal Console Display */
    .terminal-box {
        background-color: #030712;
        border: 1px solid #1F2937;
        border-radius: 6px;
        padding: 12px 16px;
        color: #E5E7EB;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        line-height: 1.45;
        overflow-x: auto;
        white-space: pre-wrap;
        margin-top: 8px;
        margin-bottom: 8px;
    }
    .terminal-header {
        color: #9CA3AF;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
        border-bottom: 1px solid #1F2937;
        padding-bottom: 4px;
    }

    /* Diagnostic Card */
    .diag-card {
        background-color: #18181B;
        border-left: 4px solid #EF4444;
        border-radius: 4px;
        padding: 12px 16px;
        margin-top: 10px;
        margin-bottom: 12px;
    }
    .diag-title {
        color: #F87171;
        font-weight: 600;
        font-size: 0.88rem;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 4px;
    }
    .diag-body {
        color: #D1D5DB;
        font-size: 0.82rem;
        line-height: 1.4;
    }

    /* Metric Cards */
    .metric-pill {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 8px 14px;
        text-align: center;
    }
    .metric-pill-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        color: #94A3B8;
        letter-spacing: 0.05em;
    }
    .metric-pill-val {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC;
        font-family: 'JetBrains Mono', monospace;
    }
</style>
"""


def apply_custom_styles():
    """Inject custom styles into the Streamlit app."""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
