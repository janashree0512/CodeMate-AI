"""
CodeMate AI - Interactive Terminal CLI (cli.py)
================================================
A direct terminal chat loop for CodeMate AI reusing CodeMateAgent.

Features:
- Commands: /help, /clear, /files, /quit, /plan
- Displays execution steps, active plan, tool calls, and results in the terminal.
- Human-in-the-Loop approval: When approval is needed, prompts "Approve <tool> <args>? [y/n]"
- Reuses CodeMateAgent without duplicating logic.
"""

import sys
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config, load_config, ConfigurationError
from agent.agent import CodeMateAgent, AgentResult, ToolExecutionRecord
from tools.tool_registry import create_default_registry


def print_banner():
    print("=" * 65)
    print("           CodeMate AI - Autonomous Coding Assistant (CLI)")
    print("=" * 65)
    print(" Type your programming task or use one of the commands below:")
    print("   /help   - Show this help message")
    print("   /files  - List current files in the sandboxed workspace")
    print("   /clear  - Clear screen")
    print("   /quit   - Exit CodeMate AI")
    print("=" * 65)


def list_workspace_files(config: Config):
    print(f"\n[Workspace Files: {config.workspace_dir}]")
    if not config.workspace_dir.exists():
        print("  (Workspace directory does not exist)")
        return
    files = list(config.workspace_dir.glob("**/*"))
    if not files:
        print("  (Workspace is empty)")
        return
    for f in sorted(files):
        if f.is_file():
            rel = f.relative_to(config.workspace_dir)
            size = f.stat().st_size
            print(f"  - {rel.as_posix():<35} ({size} bytes)")
    print()


class CLIApprovalAgent(CodeMateAgent):
    """
    Subclass of CodeMateAgent that intercepts tools requiring approval
    and prompts the user interactively in the terminal.
    """
    def _execute_tool_call(self, tool_call, iteration: int) -> ToolExecutionRecord:
        tool_def = self.registry.get_tool(tool_call.tool_name)
        args = dict(tool_call.arguments)

        # Check if the tool has an 'approved' parameter and user approval is required
        if self.config.require_user_approval and tool_def and "approved" in tool_def.signature.parameters:
            if not self.auto_approve:
                print(f"\n[APPROVAL REQUIRED - Iteration {iteration}]")
                print(f"  Tool: {tool_call.tool_name}")
                print(f"  Arguments: {json.dumps(args, indent=2)}")
                try:
                    ans = input("  Approve this action? [y/N]: ").strip().lower()
                    args["approved"] = ans in ("y", "yes")
                except (EOFError, KeyboardInterrupt):
                    args["approved"] = False
                    print("\n  [Aborted by user]")
            else:
                args["approved"] = True

        # Call parent with the potentially user-approved args
        # We temporarily update tool_call.arguments with the approved flag
        original_args = tool_call.arguments
        tool_call.arguments = args
        try:
            record = super()._execute_tool_call(tool_call, iteration)
        finally:
            tool_call.arguments = original_args

        # Print step progress to terminal
        status_tag = "[OK]" if record.success else "[FAILED]"
        print(f"  -> {status_tag} {record.tool_name}")
        if not record.success and "error" in record.result:
            print(f"     Error: {record.result['error']}")
        return record


def run_cli():
    # Load configuration
    try:
        config = load_config(strict_api_key=False)
    except ConfigurationError as exc:
        print(f"Error loading configuration: {exc}")
        sys.exit(1)

    print_banner()

    # Inform user about API key status
    has_key = bool(config.groq_api_key and config.groq_api_key != "your_groq_api_key_here")
    if not has_key:
        print("\n[WARNING] GROQ_API_KEY is not configured in .env.")
        print("Live model queries will fail. Set your key in .env or run offline tests.\n")

    registry = create_default_registry(config=config)
    # By default, auto_approve matches require_user_approval
    agent = CLIApprovalAgent(
        config=config,
        registry=registry,
        auto_approve=not config.require_user_approval
    )

    while True:
        try:
            user_input = input("\nCodeMate> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/quit", "/exit", "exit", "quit"):
            print("Goodbye!")
            break
        elif cmd == "/help":
            print_banner()
            continue
        elif cmd == "/files":
            list_workspace_files(config)
            continue
        elif cmd == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            print_banner()
            continue

        # Run agent on task
        print(f"\n[Running Agent on Task]: {user_input}")
        print("-" * 65)

        try:
            result: AgentResult = agent.run(task=user_input)

            print("-" * 65)
            print(f"\n[Execution Summary]")
            print(f"  Status: {'SUCCESS' if result.success else 'FAILED'}")
            print(f"  Iterations: {result.iterations} / {result.max_iterations}")
            print(f"  Termination Reason: {result.termination_reason}")
            print(f"  Correction Attempts: {result.correction_attempts}")
            print(f"  Total Tokens: {result.total_tokens}")

            if result.plan:
                print("\n  Task Plan:")
                for i, step in enumerate(result.plan, 1):
                    print(f"    {i}. {step}")

            print(f"\n[Final Response]:\n{result.final_response}\n")

        except Exception as err:
            print(f"\n[Error executing task]: {err}")


if __name__ == "__main__":
    run_cli()
