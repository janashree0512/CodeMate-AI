# CodeMate AI – Autonomous AI-Powered Coding Agent

CodeMate AI is an autonomous, first-principles AI coding agent built from scratch in Python. It accepts high-level natural language programming tasks, plans execution steps, generates code, interacts with the local workspace using tools, executes scripts via controlled subprocesses, inspects execution streams (stdout/stderr/exit codes), and autonomously self-heals runtime and syntax errors using a closed-loop ReAct (Reason + Act + Observe) cycle.

---

## 1. Project Objectives & Problem Statement

### The Problem
Traditional LLMs operate in an open-loop, stateless manner: they generate code snippets but have no ability to execute, observe, or verify the validity of their output. When runtime errors or dependency bugs emerge, a human must manually copy tracebacks back into the chat. Furthermore, existing agentic frameworks (LangChain, AutoGen, CrewAI) obscure the underlying mechanics with excessive abstractions.

### The Solution: CodeMate AI
CodeMate AI implements an explicit, transparent ReAct control loop:
1. **Autonomous Planning**: Deconstructs user tasks into discrete, verifiable actions with `update_plan` and `show_plan`.
2. **First-Principles Tool Calling**: Direct integration with Groq API and Llama 3 models without heavy framework dependencies.
3. **Sandboxed Workspace**: All file modifications are restricted to an isolated `workspace/` directory with path traversal guards.
4. **Controlled Subprocess Execution**: Scripts and allowlisted shell commands run in a sanitized child environment with strict execution timeouts and separate stdout/stderr capture.
5. **Self-Correction Loop**: Execution failures produce structured diagnostic reports that the agent uses to patch code and re-verify until success or iteration limits are reached.
6. **Safety & Stability Guards**: Repeated-failure detection (stops loops at 3 failures), output truncation (caps huge outputs at 4,000 chars), and conversation history compression (prevents token overflow).

---

## 2. System Architecture

```text
+-------------------------------------------------------------------------+
|                  User Interfaces: Streamlit UI or CLI                   |
|   (Task Input | Action Summaries | Execution Logs | Workspace Preview)  |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                         Agent Controller (ReAct)                        |
|   (History Compression | Repeated-Failure Guard | Diagnostics Analyzer) |
+------------------+-----------------------------------+------------------+
                   |                                   |
                   v                                   v
+--------------------------------------+   +------------------------------+
|             LLM Gateway              |   |        Tool Registry         |
|   (Groq Llama 3.3 | Auto-Retry)      |   |   (Dispatcher & Schemas)     |
+--------------------------------------+   +--------------+---------------+
                                                          |
                 +----------------------------------------+----------------------------------------+
                 |                                                                                 |
                 v                                                                                 v
+-------------------------------------------------+                             +------------------------------------+
|             File Management Tools               |                             |       Code Execution Engine        |
| (create_file, read_file, edit_file, delete,     |                             | (execute_python, run_shell,        |
|  list_files, search_files)                      |                             |  run_tests - subprocess, timeout)  |
+------------------------+------------------------+                             +------------------+-----------------+
                         |                                                                         |
                         +--------------------------------+----------------------------------------+
                                                          |
                                                          v
                                       +------------------------------------+
                                       |       Sandboxed Workspace          |
                                       |  (/workspace directory jail)       |
                                       +------------------------------------+
```

---

## 3. Project Directory Structure

```text
CodeMate_AI/
├── .env.example              # Template for environment variables and secrets
├── .gitignore                # Excludes .env, venv/, caches, and workspace files
├── README.md                 # Complete system documentation and viva guide
├── requirements.txt          # Pinned core dependencies
├── config.py                 # Central configuration and validation module
├── app.py                    # Streamlit frontend dashboard
├── cli.py                    # Interactive terminal chat CLI with approval prompts
├── demo_agent.py             # Offline demonstration script
├── demo_error_correction.py  # Demonstrates syntax, runtime, and timeout self-healing
├── demo_live_selfcorrection.py # Live Groq API self-correction demonstration
│
├── agent/                    # Core agent reasoning & LLM communication
│   ├── __init__.py           # Agent package exports
│   ├── agent.py              # Autonomous ReAct agent controller with state tracker
│   ├── diagnostics.py        # Traceback parser and actionable error diagnostics
│   ├── prompts.py            # System prompts, few-shot templates, and error schemas
│   └── llm_client.py         # Groq API client with auto-retry and schema formatting
│
├── tools/                    # Tool definitions and sandboxed execution boundaries
│   ├── __init__.py           # Tools package exports
│   ├── file_tools.py         # File CRUD, edit_file (surgical replace), search_files (grep)
│   ├── execution_tool.py     # Controlled subprocess runner (execute_python, run_shell, run_tests)
│   ├── code_executor.py      # Backward-compatible execution alias
│   └── tool_registry.py      # Dynamic tool registration, schema generator, and dispatcher
│
├── ui/                       # Streamlit UI styling and dashboard components
│   ├── __init__.py           # UI package exports
│   ├── styles.py             # Dark developer theme CSS rules
│   ├── views.py              # Timeline, telemetry metrics, and terminal views
│   └── sidebar.py            # Sidebar navigation, history, and workspace explorer
│
├── workspace/                # Isolated directory where the agent creates & runs code
│   └── .gitkeep
│
└── tests/                    # Comprehensive automated unit tests (100% passing)
    ├── __init__.py
    ├── test_config.py        # Configuration, environment variable, and path tests
    ├── test_file_tools.py    # Path traversal, CRUD, and file overwrite approval tests
    ├── test_executor.py      # Subprocess execution, timeouts, and credential scrub tests
    ├── test_tool_registry.py # Tool registration and keyword dispatch tests
    ├── test_agent_controller.py # ReAct loop execution, mocking, and self-correction tests
    ├── test_agent_structure.py  # Schema formatting and controller architecture tests
    ├── test_diagnostics.py   # Traceback categorization and suggestion tests
    ├── test_llm_client.py    # LLM client parsing, authentication, and retry tests
    ├── test_stage1_tools.py  # Stage 1 tools: edit_file, run_shell, search_files, run_tests, plan
    ├── test_stage2_robustness.py # Stage 2: repeated-failure guard, truncation, history trimming
    ├── test_stage3_safety_cli.py # Stage 3: demo mode safety banner, CLI functionality
    └── test_ui.py            # Streamlit dashboard layout and session state tests
```

---

## 4. File-by-File Explanation

| File | Module / Responsibility |
| :--- | :--- |
| `config.py` | Validates `.env` variables, enforces positive numerical limits for `MAX_ITERATIONS`, `EXECUTION_TIMEOUT_SECONDS`, `MAX_TOOL_OUTPUT_CHARS`, and `MAX_HISTORY_MESSAGES`. |
| `app.py` | Professional Streamlit dashboard displaying task inputs, action timelines, live terminal output, and workspace inspection. In demonstration mode, clearly indicates approval bypass. |
| `cli.py` | Full-featured interactive terminal chat interface. Supports `/help`, `/files`, `/clear`, `/quit` and prompts for user approval before mutating workspace files or executing commands. |
| `demo_live_selfcorrection.py` | Demonstrates real closed-loop error correction with the live Groq API without mock objects. |
| `agent/agent.py` | Implements `CodeMateAgent` and `AgentExecutionState`. Manages conversation history compression, repeated-failure protection, and output truncation. |
| `agent/diagnostics.py` | Analyzes stderr tracebacks, identifies exception classes (`SyntaxError`, `IndexError`, `ZeroDivisionError`), and provides actionable suggestions to the agent. |
| `agent/prompts.py` | Contains `SYSTEM_PROMPT` guiding the model to plan first, make surgical replacements via `edit_file`, and run verification. |
| `agent/llm_client.py` | Wraps Groq API with exponential backoff retry for network/rate-limit errors, and safely catches malformed JSON from the model. |
| `tools/file_tools.py` | Safe file operations (`create_file`, `read_file`, `edit_file`, `search_files`, `list_files`, `delete_file`) jailed to `workspace/`. |
| `tools/execution_tool.py` | Executes Python scripts (`execute_python`), allowlisted shell commands (`run_shell`), and pytest suites (`run_tests`) in a sanitized subprocess. |
| `tools/tool_registry.py` | Central dynamic registry storing tool definitions, generating Groq function schemas, and dispatching validated calls. |
| `ui/` | Modular Streamlit components (views, custom CSS styling, and sidebar). |
| `tests/` | Comprehensive test suite covering all modules, security boundaries, and edge cases. |

---

## 5. Agent Tools Reference

| Tool Name | Parameters | Purpose & Safety Rules |
| :--- | :--- | :--- |
| `update_plan` | `steps: list[str]` | Stores a numbered task plan in `AgentExecutionState` and action logs. |
| `show_plan` | (none) | Returns the current active plan. |
| `create_file` | `filename`, `content`, `overwrite=False`, `approved=False` | Creates a new file. Requires `approved=True` if overwriting an existing file. |
| `edit_file` | `filename`, `old_text`, `new_text`, `approved=False` | Replaces exactly one occurrence of `old_text` with `new_text`. Preferred over rewriting whole files. |
| `read_file` | `filename` | Reads file content within `workspace/`. |
| `search_files` | `pattern`, `directory=""` | Grep-like search across workspace files. Returns matching file, line number, and line content (capped at 50 matches). |
| `list_files` | `directory=""` | Lists files and folders inside `workspace/`. |
| `delete_file` | `filename`, `approved=False` | Deletes a file inside `workspace/`. Mandatory approval required. |
| `execute_python` | `filename`, `approved=False`, `timeout=None` | Runs a Python script with timeout watchdog and sanitized environment. |
| `run_shell` | `command`, `approved=False`, `timeout=None` | Runs an allowlisted command (`pytest`, `python`, `pip install`, `ls`, `cat`) without `shell=True`. Blocks tokens like `rm`, `sudo`, `curl`, `wget`, `|`, `;`, `&&`, `>`, `..`. |
| `run_tests` | `path=""`, `approved=False`, `timeout=None` | Convenience wrapper that executes `pytest` and returns compact pass/fail counts and failed test names. |

---

## 6. Installation & Quick Start

### 1. Clone the Repository & Create Virtual Environment
```bash
git clone https://github.com/[YOUR-USERNAME-OR-ORGANIZATION]/CodeMate_AI.git
cd CodeMate_AI
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and set your Groq API key:
```bash
# Windows:
copy .env.example .env
# Linux/macOS:
cp .env.example .env
```
Edit `.env`:
```env
GROQ_API_KEY=gsk_your_actual_key_here
MODEL_NAME=llama-3.3-70b-versatile
MAX_ITERATIONS=10
EXECUTION_TIMEOUT_SECONDS=10
REQUIRE_USER_APPROVAL=true
WORKSPACE_DIR=workspace
MAX_TOOL_OUTPUT_CHARS=4000
MAX_HISTORY_MESSAGES=30
```

### 4. Run the Test Suite
```bash
python -m pytest tests/ -v
```

### 5. Launch User Interfaces

**Option A: Streamlit Web Dashboard**
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

**Option B: Interactive Terminal CLI**
```bash
python cli.py
```
Use `/help`, `/files`, `/clear`, or type any task directly.

**Option C: Live Self-Correction Demo**
```bash
python demo_live_selfcorrection.py
```
Demonstrates autonomous multi-turn bug diagnosis and patching on the live Groq API.
