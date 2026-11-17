"""
Unit tests for the Error Diagnostics & Analysis Engine (agent/diagnostics.py).
Validates syntax error detection, runtime exception parsing, timeout classification,
and line-number extraction from real Python tracebacks.
"""

import pytest
from agent.diagnostics import parse_execution_error, DiagnosticReport


class TestErrorDiagnostics:
    """Tests for parse_execution_error."""

    def test_diagnose_syntax_error(self):
        stderr = (
            '  File "workspace/calc_syntax.py", line 6\n'
            '    def add_numbers(a, b\n'
            '                        ^\n'
            "SyntaxError: '(' was never closed\n"
        )
        report = parse_execution_error(stdout="", stderr=stderr, exit_code=1)

        assert isinstance(report, DiagnosticReport)
        assert report.category == "syntax_error"
        assert report.exception_name == "SyntaxError"
        assert report.failed_line == 6
        assert "calc_syntax.py" in report.failed_file
        assert "unmatched parentheses" in report.actionable_suggestion.lower() or "syntax issue" in report.actionable_suggestion.lower()

    def test_diagnose_indentation_error(self):
        stderr = (
            '  File "workspace/calc_indent.py", line 3\n'
            '    print("indented incorrectly")\n'
            'IndentationError: unexpected indent\n'
        )
        report = parse_execution_error(stdout="", stderr=stderr, exit_code=1)

        assert report.category == "syntax_error"
        assert report.exception_name == "IndentationError"
        assert report.failed_line == 3
        assert "calc_indent.py" in report.failed_file

    def test_diagnose_zerodivision_runtime_error(self):
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "workspace/div.py", line 14, in <module>\n'
            "    res = 50 / 0\n"
            "          ~~~^~~\n"
            "ZeroDivisionError: division by zero\n"
        )
        report = parse_execution_error(stdout="", stderr=stderr, exit_code=1)

        assert report.category == "runtime_error"
        assert report.exception_name == "ZeroDivisionError"
        assert report.failed_line == 14
        assert "div.py" in report.failed_file
        assert "divisor is not zero" in report.actionable_suggestion

    def test_diagnose_index_error(self):
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "workspace/list_test.py", line 8, in get_elem\n'
            "    return arr[100]\n"
            "IndexError: list index out of range\n"
        )
        report = parse_execution_error(stdout="", stderr=stderr, exit_code=1)

        assert report.category == "runtime_error"
        assert report.exception_name == "IndexError"
        assert report.failed_line == 8
        assert "index out of bounds" in report.actionable_suggestion.lower()

    def test_diagnose_assertion_error(self):
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "workspace/test_algo.py", line 22, in <module>\n'
            "    assert compute(4) == 20\n"
            "AssertionError\n"
        )
        report = parse_execution_error(stdout="", stderr=stderr, exit_code=1)

        assert report.category == "assertion_error"
        assert report.exception_name == "AssertionError"
        assert report.failed_line == 22
        assert "Assertion failed" in report.actionable_suggestion

    def test_diagnose_timeout_infinite_loop(self):
        report = parse_execution_error(
            stdout="Starting while loop...",
            stderr="",
            exit_code=-1,
            timed_out=True
        )

        assert report.category == "timeout"
        assert report.exception_name == "TimeoutExpired"
        assert "infinite loop" in report.actionable_suggestion.lower()

    def test_diagnose_general_error_fallback(self):
        report = parse_execution_error(
            stdout="Process failed unexpectedly.",
            stderr="Unspecified crash occurred.",
            exit_code=137,
            timed_out=False
        )

        assert report.category == "general_error"
        assert report.exception_name == "NonZeroExitCode"
        assert "137" in report.exception_message
