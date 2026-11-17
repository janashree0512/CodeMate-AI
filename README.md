# CodeMate AI – Autonomous AI-Powered Coding Agent

CodeMate AI is an autonomous, first-principles AI coding agent built from scratch in Python. It accepts high-level natural language programming tasks, plans execution steps, generates code, interacts with the local workspace using tools, executes scripts via controlled subprocesses, inspects execution streams (stdout/stderr/exit codes), and autonomously self-heals runtime and syntax errors using a closed-loop ReAct (Reason + Act + Observe) cycle.

---

## 1. Project Objectives & Problem Statement

### The Problem
Traditional LLMs operate in an open-loop, stateless manner: they generate code snippets but have no ability to execute, observe, or verify the validity of their output. When runtime errors or dependency bugs emerge, a human must manually copy tracebacks back into the chat. Furthermore, existing agentic frameworks (LangChain, AutoGen, CrewAI) obscure the underlying mechanics with excessive abstractions.

### The Solution: CodeMate AI
CodeMate AI implements an explicit, transparent ReAct control loop:
1. **Autonomous Planning**: Deconstructs user tasks into discrete, verifiable actions.
2. **First-Principles Tool Calling**: Direct integration with Groq API and Llama 3 models without heavy framework dependencies.
3. **Sandboxed Workspace**: All file modifications are restricted to an isolated `workspace/` directory with path traversal guards.
4. **Controlled Subprocess Execution**: Scripts run in a sanitized child environment with strict execution timeouts and separate stdout/stderr capture.
5. **Self-Correction Loop**: Execution failures produce structured diagnostic reports that the agent uses to patch code and re-verify until success or iteration limits are reached.

---

## 2. System Architecture

```text
+-------------------------------------------------------------------------+
|                        Streamlit User Interface                         |
|   (Task Input | Action Summaries | Execution Logs | Workspace Preview)  |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                         Agent Controller (ReAct)                        |
|        (History Management | Iteration Guard | Termination Checks)      |
+------------------+-----------------------------------+------------------+
                   |                                   |
                   v                                   v
+--------------------------------------+   +------------------------------+
|             LLM Gateway              |   |        Tool Registry         |
|        (Groq / Llama 3.3 70B)        |   |   (Dispatcher & Schemas)     |
+--------------------------------------+   +--------------+---------------+
                                                          |
                               +--------------------------+--------------------------+
                               |                                                     |
                               v                                                     v
            +------------------------------------+                +------------------------------------+
            |        File Management Tools       |                |        Code Execution Engine       |
            | (create_file, read_file, delete)   |                |  (subprocess, timeout, sanitized)  |
            +------------------+-----------------+                +------------------+-----------------+
                               |                                                     |
                               +--------------------------+--------------------------+
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
├── .gitignore                # Version control exclusions (ignores .env, cache, workspace files)
├── README.md                 # Complete system documentation and interview guide
├── requirements.txt          # Pinned core dependencies
├── config.py                 # Central configuration and validation module
├── app.py                    # Streamlit frontend dashboard
│
├── agent/                    # Core agent reasoning & LLM communication
│   ├── __init__.py           # Agent package exports
│   ├── agent.py              # Autonomous ReAct agent controller and state tracker
│   ├── prompts.py            # System prompts, few-shot templates, and error schemas
│   └── llm_client.py         # Groq API client with tool schema formatting
│
├── tools/                    # Tool definitions and sandboxed execution boundaries
│   ├── __init__.py           # Tools package exports
│   ├── file_tools.py         # Sandboxed file operations with path jail protection
│   ├── execution_tool.py     # Controlled subprocess runner (timeout + credential stripping)
│   ├── code_executor.py      # Backward-compatible execution alias
│   └── tool_registry.py      # Dynamic tool registration and dispatch engine
│
├── workspace/                # Isolated directory where the agent creates & runs code
│   └── .gitkeep
│
└── tests/                    # Automated unit tests
    ├── __init__.py
    ├── test_config.py        # Configuration, environment variable, and path tests
    ├── test_file_tools.py    # Path traversal, CRUD, and file overwrite approval tests
    ├── test_executor.py      # Subprocess execution, timeouts, and credential scrub tests
    └── test_tool_registry.py # Tool registration and keyword dispatch tests
```

---

## 4. File-by-File Explanation

| File | Module / Responsibility |
| :--- | :--- |
| `config.py` | Validates `.env` variables, enforces positive numerical limits for `MAX_ITERATIONS` and `EXECUTION_TIMEOUT_SECONDS`, and guarantees workspace directory resolution. |
| `app.py` | Provides a clean, real-time Streamlit dashboard displaying task inputs, action summaries, tool calls, and workspace file contents. (Never exposes private model chain-of-thought). |
| `agent/agent.py` | Implements `CodeMateAgent` and `AgentExecutionState`. Manages conversation history, controls ReAct iterations, and tracks termination conditions. |
| `agent/prompts.py` | Contains `SYSTEM_PROMPT`, `TASK_PLANNING_TEMPLATE`, and `ERROR_CORRECTION_PROMPT` to instruct the LLM on autonomous planning, verification, and diagnostics. |
| `agent/llm_client.py` | Wraps the Groq API and formats registered tools into standard OpenAI/Groq function-calling JSON schemas. |
| `tools/file_tools.py` | Safe file CRUD operations (`create_file`, `read_file`, `list_files`, `delete_file`). Enforces a directory jail via `Path.relative_to()`. |
| `tools/execution_tool.py` | Spawns isolated `subprocess.run()` calls with timeout protection, captures stdout/stderr, and scrubs sensitive host credentials from the child environment. |
| `tools/tool_registry.py` | Dynamic registry managing tool definitions, inspecting function signatures, and safely dispatching calls with keyword arguments. |
| `tests/` | Comprehensive unit test suite covering path security, timeouts, approvals, environment isolation, and tool dispatching. |

---

## 5. Security Architecture & Interview Talking Points

### Restricted Local Execution vs. Container Sandbox
- **Current Implementation (Restricted Local Execution)**:
  - Process isolation via Python's `subprocess` (prevents global state / memory pollution).
  - Directory jail (blocks path traversal outside `workspace/`).
  - Environment variable sanitization (strips `GROQ_API_KEY` and host credentials).
  - Execution watchdog (hard timeout kills runaway infinite loops).
  - Human-in-the-loop approval gates (for code execution and file overwrites).
- **Production Container Boundary (Docker / Firecracker)**:
  - For multi-tenant production hosting, the `BaseExecutor` abstract class is designed to be backed by Docker or Firecracker MicroVMs, providing kernel-level cgroups, network namespace isolation, and ephemeral filesystems.

---

## 6. Installation & Quick Start

### 1. Clone the Repository & Create Virtual Environment
```bash
git clone https://github.com/your-username/CodeMate_AI.git
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
copy .env.example .env
```
Edit `.env`:
```env
GROQ_API_KEY=your_actual_groq_api_key_here
MODEL_NAME=llama-3.3-70b-versatile
MAX_ITERATIONS=10
EXECUTION_TIMEOUT_SECONDS=10
REQUIRE_USER_APPROVAL=true
WORKSPACE_DIR=workspace
```

### 4. Run the Test Suite
```bash
python -m pytest tests/ -v
```

### 5. Launch the Streamlit Frontend
```bash
streamlit run app.py
```
