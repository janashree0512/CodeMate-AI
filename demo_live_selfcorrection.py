"""
demo_live_selfcorrection.py
===========================
Live demonstration of CodeMate AI's autonomous self-correction loop using
the REAL Groq client (no mocks).

Task:
"Write a Python program that reads numbers, computes the average and handles
an empty list, then test it."

Demonstration workflow:
- The agent intentionally starts with a buggy version (ZeroDivisionError on empty list),
- Executes it inside the sandboxed workspace,
- Observes the runtime failure & diagnostic traceback,
- Analyzes the error and patches the file using edit_file / create_file,
- Re-executes until all assertions pass.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config, ConfigurationError
from agent.agent import CodeMateAgent
from tools.tool_registry import create_default_registry


def main():
    print("=" * 70)
    print("  CodeMate AI - Live Autonomous Error Detection & Self-Correction Demo")
    print("=" * 70)

    # 1. Check for valid Groq API Key
    try:
        config = load_config(strict_api_key=True)
    except ConfigurationError as exc:
        print("\n[GROQ_API_KEY NOT CONFIGURED]")
        print(f"Error: {exc}")
        print("\nTo run this live demonstration:")
        print("  1. Copy .env.example to .env (or edit .env)")
        print("  2. Set GROQ_API_KEY=gsk_your_actual_key_here")
        print("  3. Run: python demo_live_selfcorrection.py\n")
        sys.exit(0)

    print(f"\n[Configuration Loaded]")
    print(f"  Model: {config.model_name}")
    print(f"  Workspace: {config.workspace_dir}")
    print(f"  Max Iterations: {config.max_iterations}")
    print(f"  Max Tool Output Chars: {config.max_tool_output_chars}")
    print(f"  Max History Messages: {config.max_history_messages}\n")

    registry = create_default_registry(config=config)
    agent = CodeMateAgent(config=config, registry=registry, auto_approve=True)

    task = (
        "Write a Python script named 'compute_average.py' that computes the average "
        "of a list of numbers and safely handles an empty list (returning 0.0). "
        "First, write a version that intentionally fails on an empty list by doing sum(nums)/len(nums) "
        "without checking for len == 0. Run it using execute_python to observe the ZeroDivisionError, "
        "then edit the file using edit_file to add the empty check, and re-run until all assertions pass."
    )

    print("-" * 70)
    print(f"TASK: {task}")
    print("-" * 70)
    print("Starting autonomous ReAct loop...\n")

    result = agent.run(task=task)

    print("\n" + "=" * 70)
    print("  EXECUTION TRACE & TELEMETRY")
    print("=" * 70)
    print(f"Iterations: {result.iterations} / {result.max_iterations}")
    print(f"Success: {result.success}")
    print(f"Termination Reason: {result.termination_reason}")
    print(f"Correction Attempts: {result.correction_attempts}")
    print(f"Total Tokens Used: {result.total_tokens}")

    if result.plan:
        print("\nActive Plan:")
        for idx, step in enumerate(result.plan, 1):
            print(f"  {idx}. {step}")

    print("\nAction Timeline:")
    for record in result.action_history:
        status_sym = "[OK]" if record.success else "[ERR]"
        print(f"  Iter {record.iteration}: {status_sym} {record.tool_name}({record.arguments})")
        if not record.success and "error" in record.result:
            print(f"         Error: {record.result.get('error')}")

    if result.execution_history:
        print("\nPython Subprocess Execution History:")
        for exec_info in result.execution_history:
            iter_no = exec_info.get("iteration")
            passed = exec_info.get("success")
            code = exec_info.get("exit_code")
            print(f"  [Iter {iter_no}] exit_code={code} (Passed: {passed})")
            if exec_info.get("stdout"):
                print(f"    stdout: {exec_info['stdout'].strip()[:200]}")
            if exec_info.get("stderr"):
                print(f"    stderr: {exec_info['stderr'].strip()[:200]}")
            if exec_info.get("diagnostics"):
                diag = exec_info["diagnostics"]
                print(f"    Diagnostics: Category={diag.get('category')}, Type={diag.get('exception_name')}")

    print("\n" + "-" * 70)
    print("FINAL AGENT RESPONSE:")
    print("-" * 70)
    print(result.final_response)
    print("=" * 70)


if __name__ == "__main__":
    main()
