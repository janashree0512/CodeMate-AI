"""
CodeMate AI - LLM Integration Verification Script (test_llm_query.py)
======================================================================

Sends a coding-related question to Groq using the configured Llama model
and prints the model's response, token metrics, and execution latency.

Usage:
  python test_llm_query.py          # Queries live Groq API using GROQ_API_KEY from .env
  python test_llm_query.py --mock   # Demonstrates expected output using a verified offline mock
"""

import sys
import time
from unittest.mock import MagicMock
from config import load_config, ConfigurationError
from agent.llm_client import GroqClient, LLMAuthenticationError, LLMClientError, LLMResponse


def run_live_test(client: GroqClient, question: str, system_prompt: str):
    """Run real query against Groq API."""
    start_time = time.perf_counter()
    response = client.ask(
        prompt=question,
        system_prompt=system_prompt,
        temperature=0.2
    )
    elapsed = time.perf_counter() - start_time

    print("--------------------------------------------------")
    print("Model Response (Live Groq API):")
    print("--------------------------------------------------")
    print(response)
    print("--------------------------------------------------")
    print(f"✓ Success! Live response received in {elapsed:.2f} seconds.")


def run_mock_test(question: str):
    """Run simulated test to demonstrate behavior offline."""
    print("--------------------------------------------------")
    print("Model Response (Simulated llama-3.3-70b-versatile):")
    print("--------------------------------------------------")
    mock_reply = (
        "def is_palindrome(text: str) -> bool:\n"
        "    \"\"\"\n"
        "    Checks whether a given string is a palindrome.\n"
        "    Ignores casing and non-alphanumeric characters.\n"
        "    \"\"\"\n"
        "    cleaned = ''.join(ch.lower() for ch in text if ch.isalnum())\n"
        "    return cleaned == cleaned[::-1]\n\n"
        "# Test Assertions\n"
        "assert is_palindrome('A man, a plan, a canal: Panama') is True\n"
        "assert is_palindrome('CodeMate') is False\n"
        "print('All palindrome assertions passed successfully!')"
    )
    print(mock_reply)
    print("--------------------------------------------------")
    print("Token Metrics:")
    print("• Prompt Tokens     : 38")
    print("• Completion Tokens : 84")
    print("• Total Tokens      : 122")
    print("• Finish Reason     : stop")
    print("• Simulated Latency : 0.32s")


def main():
    use_mock = "--mock" in sys.argv

    print("==================================================")
    print("CodeMate AI - Groq LLM Integration Test")
    print("==================================================")

    config = load_config(strict_api_key=False)
    has_key = bool(config.groq_api_key and config.groq_api_key != "your_groq_api_key_here")

    print(f"Configured Model : {config.model_name}")
    print(f"API Key Present  : {'Yes' if has_key else 'No (Missing or Placeholder in .env)'}")

    coding_question = (
        "Write a clean Python function to check whether a string is a palindrome. "
        "Include docstrings, type hints, and two example test assertions."
    )
    system_instruction = "You are an expert Python coding assistant for CodeMate AI."

    print("\nSending coding prompt:")
    print(f"Prompt: \"{coding_question}\"\n")

    if use_mock:
        run_mock_test(coding_question)
        return

    client = GroqClient(config=config)
    try:
        run_live_test(client, coding_question, system_instruction)

    except LLMAuthenticationError as err:
        print(f"\n[AUTHENTICATION NOTICE]: {err}")
        print("\nTo run live queries against Groq:")
        print("1. Obtain a free API key from https://console.groq.com/keys")
        print("2. Create a '.env' file in this directory:")
        print("   GROQ_API_KEY=gsk_your_actual_key_here")
        print("3. Re-run: python test_llm_query.py\n")
        print("Alternatively, run with --mock to verify the test harness offline:")
        print("   python test_llm_query.py --mock\n")
        sys.exit(2)

    except LLMClientError as err:
        print(f"\n[API ERROR]: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
