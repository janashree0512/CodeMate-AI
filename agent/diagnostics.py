"""
CodeMate AI - Error Diagnostics & Analysis Engine (agent/diagnostics.py)
========================================================================

Parses standard error output, tracebacks, and process exit codes from Python
executions. Extracts exception categories, line numbers, and root causes,
and crafts actionable structured feedback to drive autonomous error self-correction.

Categories:
- syntax_error: SyntaxError, IndentationError, TabError
- runtime_error: ZeroDivisionError, IndexError, KeyError, TypeError, ValueError, etc.
- timeout: Process hung / infinite loop killed by timeout watchdog
- assertion_error: AssertionError from test assertion failures
- general_error: Non-zero exit code without standard traceback
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DiagnosticReport:
    """Structured analysis of an execution failure."""
    category: str  # "syntax_error" | "runtime_error" | "timeout" | "assertion_error" | "general_error"
    exception_name: str
    exception_message: str
    failed_file: Optional[str]
    failed_line: Optional[int]
    traceback_summary: str
    actionable_suggestion: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert diagnostic report to a dictionary for LLM context injection."""
        return {
            "category": self.category,
            "exception_name": self.exception_name,
            "exception_message": self.exception_message,
            "failed_file": self.failed_file,
            "failed_line": self.failed_line,
            "actionable_suggestion": self.actionable_suggestion
        }


def parse_execution_error(
    stdout: str,
    stderr: str,
    exit_code: Optional[int],
    timed_out: bool = False
) -> DiagnosticReport:
    """
    Parse execution streams to identify and categorize errors.
    """
    # 1. Check for timeout / infinite loop
    if timed_out:
        return DiagnosticReport(
            category="timeout",
            exception_name="TimeoutExpired",
            exception_message="Process was terminated after exceeding maximum execution time.",
            failed_file=None,
            failed_line=None,
            traceback_summary=stderr or "Process killed by watchdog due to timeout.",
            actionable_suggestion=(
                "An infinite loop or long-running block was detected. "
                "Check 'while' or 'for' loop termination conditions, recursion base cases, "
                "or blocking I/O calls."
            )
        )

    clean_stderr = stderr.strip() if stderr else ""

    # 2. Check for SyntaxError / IndentationError
    syntax_match = re.search(
        r'File\s+["\']([^"\']+)["\'],\s+line\s+(\d+).*?\n\s*(.*?)\n\s*(\w*(?:SyntaxError|IndentationError|TabError)):\s*(.*)',
        clean_stderr,
        re.DOTALL
    )
    if syntax_match:
        failed_file = syntax_match.group(1)
        failed_line = int(syntax_match.group(2))
        err_type = syntax_match.group(4)
        err_msg = syntax_match.group(5).strip()

        return DiagnosticReport(
            category="syntax_error",
            exception_name=err_type,
            exception_message=err_msg,
            failed_file=failed_file,
            failed_line=failed_line,
            traceback_summary=clean_stderr,
            actionable_suggestion=(
                f"Syntax issue detected at line {failed_line} in '{failed_file}'. "
                "Check for missing colons (:), unmatched parentheses/brackets, invalid operators, "
                "or inconsistent indentation."
            )
        )

    # 3. Check for standard Python Runtime Exceptions (Traceback ... Exception: message)
    # Match the last line of standard traceback
    exception_match = re.search(r'([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)):?\s*(.*)$', clean_stderr)
    line_match = re.findall(r'File\s+["\']([^"\']+)["\'],\s+line\s+(\d+)', clean_stderr)

    failed_file = line_match[-1][0] if line_match else None
    failed_line = int(line_match[-1][1]) if line_match else None

    if exception_match:
        exc_name = exception_match.group(1)
        exc_msg = exception_match.group(2).strip()

        if exc_name == "AssertionError":
            category = "assertion_error"
            suggestion = (
                f"Assertion failed at line {failed_line}. The programmatic output did not match "
                "the expected test assertion. Re-verify the calculation logic against requirements."
            )
        else:
            category = "runtime_error"
            suggestion = generate_runtime_suggestion(exc_name, exc_msg, failed_line, failed_file)

        return DiagnosticReport(
            category=category,
            exception_name=exc_name,
            exception_message=exc_msg,
            failed_file=failed_file,
            failed_line=failed_line,
            traceback_summary=clean_stderr,
            actionable_suggestion=suggestion
        )

    # 4. Fallback for non-zero exit codes without standard traceback
    return DiagnosticReport(
        category="general_error",
        exception_name="NonZeroExitCode",
        exception_message=f"Process exited with non-zero exit code {exit_code}.",
        failed_file=failed_file,
        failed_line=failed_line,
        traceback_summary=clean_stderr or stdout,
        actionable_suggestion="Review standard output and exit code for unhandled program conditions."
    )


def generate_runtime_suggestion(exc_name: str, exc_msg: str, line: Optional[int], filename: Optional[str]) -> str:
    """Provide targeted advice based on the Python exception type."""
    loc = f"at line {line}" if line else ""
    if exc_name == "ZeroDivisionError":
        return f"ZeroDivisionError {loc}. Ensure the divisor is not zero before performing division or modulo."
    elif exc_name == "IndexError":
        return f"IndexError {loc}. List or sequence index out of bounds. Check sequence length or loop bounds."
    elif exc_name == "KeyError":
        return f"KeyError {loc}. Dictionary key '{exc_msg}' was not found. Use dict.get() or verify key initialization."
    elif exc_name == "TypeError":
        return f"TypeError {loc} ({exc_msg}). Check argument counts and parameter types passed to functions/operations."
    elif exc_name == "NameError":
        return f"NameError {loc} ({exc_msg}). Variable or function name is referenced before assignment or is misspelled."
    elif exc_name == "AttributeError":
        return f"AttributeError {loc} ({exc_msg}). Object does not have this attribute. Verify object type and available methods."
    elif exc_name == "ModuleNotFoundError" or exc_name == "ImportError":
        return f"Import failure {loc} ({exc_msg}). Verify that the imported module exists or is part of standard library."
    elif exc_name == "RecursionError":
        return f"RecursionError {loc}. Maximum recursion depth exceeded. Check base cases in recursive functions."
    else:
        return f"Runtime error '{exc_name}' occurred {loc}. Address: {exc_msg}."
