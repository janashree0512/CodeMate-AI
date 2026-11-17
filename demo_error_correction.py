"""
CodeMate AI - Error Detection & Self-Correction Demonstration (demo_error_correction.py)
========================================================================================

Demonstrates the 3 required scenarios:
1. A working Python program (direct success, 0 corrections).
2. A program containing a syntax error (detected, diagnosed, auto-corrected).
3. A program containing a runtime error (detected, diagnosed, auto-corrected).
"""

from unittest.mock import MagicMock
from config import load_config
from agent.agent import CodeMateAgent, AgentResult
from agent.llm_client import LLMResponse, ToolCallRequest
from tools.tool_registry import create_default_registry


def demo_working_program():
    print("\n=======================================================")
    print("SCENARIO 1: Working Python Program (Direct Success)")
    print("=======================================================")

    config = load_config(strict_api_key=False)
    registry = create_default_registry(config=config)
    mock_llm = MagicMock()
    mock_llm.format_tool_schema.return_value = []

    script_content = (
        "def fib(n):\n"
        "    seq = [0, 1]\n"
        "    for _ in range(2, n):\n"
        "        seq.append(seq[-1] + seq[-2])\n"
        "    return seq[:n]\n\n"
        "res = fib(6)\n"
        "print('Fibonacci sequence:', res)\n"
        "assert res == [0, 1, 1, 2, 3, 5]\n"
    )

    resp_1 = LLMResponse(
        content="Creating working Fibonacci script.",
        tool_calls=[
            ToolCallRequest(
                id="call_fib_create",
                tool_name="create_file",
                arguments={"filename": "fibonacci.py", "content": script_content}
            )
        ]
    )
    resp_2 = LLMResponse(
        content="Executing to verify Fibonacci sequence.",
        tool_calls=[
            ToolCallRequest(
                id="call_fib_exec",
                tool_name="execute_python",
                arguments={"filename": "fibonacci.py"}
            )
        ]
    )
    resp_3 = LLMResponse(
        content="The script fibonacci.py ran successfully and the assertions verified [0, 1, 1, 2, 3, 5].",
        tool_calls=[]
    )

    mock_llm.chat.side_effect = [resp_1, resp_2, resp_3]

    agent = CodeMateAgent(config=config, registry=registry, llm_client=mock_llm, auto_approve=True)
    result: AgentResult = agent.run("Generate and verify Fibonacci sequence")

    print(f"Task Status          : {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Correction Attempts  : {result.correction_attempts}")
    print(f"Total Iterations     : {result.iterations}")
    print(f"Execution History    : {len(result.execution_history)} run(s)")
    print(f"Verified Output      : {result.execution_history[0]['stdout'].strip()}")
    print(f"Final Agent Answer   : {result.final_response}")


def demo_syntax_error_correction():
    print("\n=======================================================")
    print("SCENARIO 2: Program with Syntax Error (Self-Correction)")
    print("=======================================================")

    config = load_config(strict_api_key=False)
    registry = create_default_registry(config=config)
    mock_llm = MagicMock()
    mock_llm.format_tool_schema.return_value = []

    # Buggy code with unclosed parenthesis
    broken_code = "def calculate_total(prices\n    return sum(prices)\nprint(calculate_total([10, 20]))\n"
    fixed_code = "def calculate_total(prices):\n    return sum(prices)\nprint('Total is:', calculate_total([10, 20]))\n"

    # Turn 1: Create buggy file
    resp_1 = LLMResponse(
        content="Creating script with syntax bug.",
        tool_calls=[
            ToolCallRequest(
                id="call_syn_create",
                tool_name="create_file",
                arguments={"filename": "broken_syntax.py", "content": broken_code}
            )
        ]
    )
    # Turn 2: Run script (will fail with SyntaxError)
    resp_2 = LLMResponse(
        content="Running script to test syntax.",
        tool_calls=[
            ToolCallRequest(
                id="call_syn_exec",
                tool_name="execute_python",
                arguments={"filename": "broken_syntax.py"}
            )
        ]
    )
    # Turn 3: Observe SyntaxError at line 1, patch file
    resp_3 = LLMResponse(
        content="Diagnostics detected SyntaxError at line 1. Fixing missing parenthesis and colon.",
        tool_calls=[
            ToolCallRequest(
                id="call_syn_fix",
                tool_name="create_file",
                arguments={"filename": "broken_syntax.py", "content": fixed_code, "overwrite": True, "approved": True}
            )
        ]
    )
    # Turn 4: Re-execute fixed code
    resp_4 = LLMResponse(
        content="Re-running broken_syntax.py to confirm the fix.",
        tool_calls=[
            ToolCallRequest(
                id="call_syn_reexec",
                tool_name="execute_python",
                arguments={"filename": "broken_syntax.py"}
            )
        ]
    )
    # Turn 5: Conclude after successful execution
    resp_5 = LLMResponse(
        content="The syntax error was diagnosed and resolved. Output: Total is: 30.",
        tool_calls=[]
    )

    mock_llm.chat.side_effect = [resp_1, resp_2, resp_3, resp_4, resp_5]

    agent = CodeMateAgent(config=config, registry=registry, llm_client=mock_llm, auto_approve=True)
    result: AgentResult = agent.run("Calculate total with syntax error recovery")

    print(f"Task Status          : {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Correction Attempts  : {result.correction_attempts}")
    print(f"Total Iterations     : {result.iterations}")
    print("Execution Attempts Detail:")
    for i, ex in enumerate(result.execution_history, 1):
        if not ex["success"]:
            diag = ex["diagnostics"]
            print(f"  Attempt {i}: FAILED - [{diag['category'].upper()}] {diag['exception_name']} at line {diag['failed_line']}")
            print(f"             Suggestion: {diag['actionable_suggestion']}")
        else:
            print(f"  Attempt {i}: PASSED - Output: {ex['stdout'].strip()}")

    print(f"Final Agent Answer   : {result.final_response}")


def demo_runtime_error_correction():
    print("\n=======================================================")
    print("SCENARIO 3: Program with Runtime Error (Self-Correction)")
    print("=======================================================")

    config = load_config(strict_api_key=False)
    registry = create_default_registry(config=config)
    mock_llm = MagicMock()
    mock_llm.format_tool_schema.return_value = []

    # Buggy code with IndexError
    buggy_code = (
        "items = ['apple', 'banana', 'cherry']\n"
        "target_index = 5\n"
        "print('Item at index 5 is:', items[target_index])\n"
    )
    fixed_code = (
        "items = ['apple', 'banana', 'cherry']\n"
        "target_index = 1\n"
        "print('Item at index 1 is:', items[target_index])\n"
    )

    # Turn 1: Create buggy file
    resp_1 = LLMResponse(
        content="Creating script with IndexError.",
        tool_calls=[
            ToolCallRequest(
                id="call_rt_create",
                tool_name="create_file",
                arguments={"filename": "list_bounds.py", "content": buggy_code}
            )
        ]
    )
    # Turn 2: Run script (will fail with IndexError)
    resp_2 = LLMResponse(
        content="Executing list_bounds.py.",
        tool_calls=[
            ToolCallRequest(
                id="call_rt_exec",
                tool_name="execute_python",
                arguments={"filename": "list_bounds.py"}
            )
        ]
    )
    # Turn 3: Observe IndexError, update file
    resp_3 = LLMResponse(
        content="Diagnostics detected IndexError at line 3. Updating index to within bounds (1).",
        tool_calls=[
            ToolCallRequest(
                id="call_rt_fix",
                tool_name="create_file",
                arguments={"filename": "list_bounds.py", "content": fixed_code, "overwrite": True, "approved": True}
            )
        ]
    )
    # Turn 4: Re-execute fixed code
    resp_4 = LLMResponse(
        content="Re-running fixed list_bounds.py.",
        tool_calls=[
            ToolCallRequest(
                id="call_rt_reexec",
                tool_name="execute_python",
                arguments={"filename": "list_bounds.py"}
            )
        ]
    )
    # Turn 5: Conclude after verified output
    resp_5 = LLMResponse(
        content="The IndexError was resolved. Output: Item at index 1 is: banana.",
        tool_calls=[]
    )

    mock_llm.chat.side_effect = [resp_1, resp_2, resp_3, resp_4, resp_5]

    agent = CodeMateAgent(config=config, registry=registry, llm_client=mock_llm, auto_approve=True)
    result: AgentResult = agent.run("Safely access list item with error recovery")

    print(f"Task Status          : {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Correction Attempts  : {result.correction_attempts}")
    print(f"Total Iterations     : {result.iterations}")
    print("Execution Attempts Detail:")
    for i, ex in enumerate(result.execution_history, 1):
        if not ex["success"]:
            diag = ex["diagnostics"]
            print(f"  Attempt {i}: FAILED - [{diag['category'].upper()}] {diag['exception_name']} at line {diag['failed_line']}")
            print(f"             Suggestion: {diag['actionable_suggestion']}")
        else:
            print(f"  Attempt {i}: PASSED - Output: {ex['stdout'].strip()}")

    print(f"Final Agent Answer   : {result.final_response}")


if __name__ == "__main__":
    demo_working_program()
    demo_syntax_error_correction()
    demo_runtime_error_correction()
