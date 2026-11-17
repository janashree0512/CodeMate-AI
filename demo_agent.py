"""
CodeMate AI - Agent Controller Demonstration Script (demo_agent.py)
===================================================================

Runs an autonomous coding task through the CodeMate AI agent loop.
Prints real-time iteration logs, tool dispatches, execution outputs,
and the verified final answer.

Usage:
  python demo_agent.py          # Runs live against Groq (requires GROQ_API_KEY in .env)
  python demo_agent.py --mock   # Runs deterministically using a mock LLM for offline demonstration
"""

import sys
from unittest.mock import MagicMock
from config import load_config
from agent.agent import CodeMateAgent, run_agent, AgentResult
from agent.llm_client import GroqClient, LLMResponse, ToolCallRequest, LLMAuthenticationError
from tools.tool_registry import create_default_registry


def run_mock_demonstration():
    """Runs a simulated multi-turn ReAct self-correction task offline."""
    print("==================================================")
    print("CodeMate AI - Simulated Agent Execution Loop")
    print("==================================================")

    config = load_config(strict_api_key=False)
    registry = create_default_registry(config=config)
    mock_llm = MagicMock()
    mock_llm.format_tool_schema.return_value = [{"type": "function"}]

    task_description = (
        "Write a Python script named 'math_stats.py' that computes the mean, "
        "median, and mode of [1, 2, 2, 3, 4, 7, 9], execute it, and report the results."
    )

    print(f"Task: {task_description}\n")

    # Turn 1: LLM decides to create the file
    turn_1_code = (
        "import statistics\n"
        "data = [1, 2, 2, 3, 4, 7, 9]\n"
        "mean_val = statistics.mean(data)\n"
        "median_val = statistics.median(data)\n"
        "mode_val = statistics.mode(data)\n"
        "print(f'Mean: {mean_val:.2f}, Median: {median_val}, Mode: {mode_val}')\n"
    )
    resp_1 = LLMResponse(
        content="I will write a Python script 'math_stats.py' to compute the statistics.",
        tool_calls=[
            ToolCallRequest(
                id="call_create_stats",
                tool_name="create_file",
                arguments={"filename": "math_stats.py", "content": turn_1_code}
            )
        ],
        finish_reason="tool_calls",
        prompt_tokens=45,
        completion_tokens=60,
        total_tokens=105
    )

    # Turn 2: LLM executes the file
    resp_2 = LLMResponse(
        content="File created. Now I will execute it to verify the calculations.",
        tool_calls=[
            ToolCallRequest(
                id="call_exec_stats",
                tool_name="execute_python",
                arguments={"filename": "math_stats.py"}
            )
        ],
        finish_reason="tool_calls",
        prompt_tokens=120,
        completion_tokens=25,
        total_tokens=145
    )

    # Turn 3: LLM observes stdout and synthesizes final response
    resp_3 = LLMResponse(
        content=(
            "The script 'math_stats.py' was successfully created and executed in the workspace.\n"
            "Execution Output:\n"
            "• Mean   : 4.00\n"
            "• Median : 3\n"
            "• Mode   : 2\n"
            "All calculations have been programmatically verified."
        ),
        tool_calls=[],
        finish_reason="stop",
        prompt_tokens=165,
        completion_tokens=70,
        total_tokens=235
    )

    mock_llm.chat.side_effect = [resp_1, resp_2, resp_3]

    agent = CodeMateAgent(config=config, registry=registry, llm_client=mock_llm, auto_approve=True)
    result: AgentResult = agent.run(task=task_description, max_iterations=5)

    print("--------------------------------------------------")
    print(f"Status           : {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Iterations Taken : {result.iterations} / {result.max_iterations}")
    print(f"Termination      : {result.termination_reason}")
    print(f"Total Tokens     : {result.total_tokens}")
    print("--------------------------------------------------")
    print("Action History:")
    for idx, action in enumerate(result.action_history, 1):
        status_sym = "✓" if action.success else "✗"
        print(f"  {status_sym} [Step {idx}] Tool: '{action.tool_name}' | Result: {action.result}")
    print("--------------------------------------------------")
    print("Final Agent Response:")
    print(result.final_response)
    print("--------------------------------------------------")


def run_live_agent():
    """Runs a live agent task using Groq API."""
    print("==================================================")
    print("CodeMate AI - Live Agent Execution Loop (Groq)")
    print("==================================================")

    config = load_config(strict_api_key=False)
    if not config.groq_api_key or config.groq_api_key == "your_groq_api_key_here":
        print("\n[AUTHENTICATION NOTICE]: GROQ_API_KEY is not configured in .env.")
        print("To run the live agent:")
        print("1. Set GROQ_API_KEY in .env")
        print("2. Run: python demo_agent.py\n")
        print("Or run the offline demonstration:")
        print("   python demo_agent.py --mock\n")
        sys.exit(2)

    task = (
        "Create a Python script named 'reverse_words.py' that takes the string "
        "'CodeMate AI is an autonomous coding agent' and reverses each word in place. "
        "Run the script, verify the output, and present the result."
    )

    print(f"Task: {task}\n")
    try:
        result = run_agent(task=task, config=config, auto_approve=True)
        print("\n--------------------------------------------------")
        print(f"Status           : {'SUCCESS' if result.success else 'FAILED'}")
        print(f"Iterations Taken : {result.iterations} / {result.max_iterations}")
        print(f"Termination      : {result.termination_reason}")
        print(f"Total Tokens     : {result.total_tokens}")
        print("--------------------------------------------------")
        print("Final Agent Response:")
        print(result.final_response)
        print("--------------------------------------------------")

    except LLMAuthenticationError as err:
        print(f"\n[AUTHENTICATION ERROR]: {err}")
    except Exception as err:
        print(f"\n[EXECUTION ERROR]: {err}")


def main():
    if "--mock" in sys.argv:
        run_mock_demonstration()
    else:
        run_live_agent()


if __name__ == "__main__":
    main()
