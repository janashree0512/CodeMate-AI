"""
Unit tests for the Groq LLM Gateway Client (agent/llm_client.py).
Uses mocks to verify message formatting, tool schema generation, response parsing,
error transformations, and token usage accounting without requiring a live API key.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from config import Config, DEFAULT_MODEL
from agent.llm_client import (
    GroqClient,
    LLMAuthenticationError,
    LLMClientError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponse,
    ToolCallRequest,
)
import groq


@pytest.fixture
def mock_config(tmp_path):
    """Provides a test config with a fake API key and workspace."""
    return Config(
        groq_api_key="gsk_mock_valid_key_12345",
        model_name=DEFAULT_MODEL,
        max_iterations=5,
        execution_timeout_seconds=5,
        require_user_approval=True,
        workspace_dir=tmp_path,
        project_root=tmp_path,
        max_tool_output_chars=4000,
        max_history_messages=30
    )


class TestGroqClientMocked:
    """Test suite using mocked Groq SDK client."""

    def test_missing_api_key_raises_auth_error(self, tmp_path):
        bad_config = Config(
            groq_api_key="",
            model_name=DEFAULT_MODEL,
            max_iterations=5,
            execution_timeout_seconds=5,
            require_user_approval=True,
            max_tool_output_chars=4000,
            max_history_messages=30,
            workspace_dir=tmp_path,
            project_root=tmp_path
        )
        client = GroqClient(config=bad_config)
        with pytest.raises(LLMAuthenticationError) as exc:
            client.get_client()
        assert "Missing GROQ_API_KEY" in str(exc.value)

    @patch("agent.llm_client.Groq")
    def test_chat_success_response_parsing(self, mock_groq_class, mock_config):
        # Configure mock response
        mock_instance = MagicMock()
        mock_groq_class.return_value = mock_instance

        mock_choice = MagicMock()
        mock_choice.message.content = "Here is a Python function: def add(a, b): return a + b"
        mock_choice.message.tool_calls = None
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 15
        mock_usage.completion_tokens = 25
        mock_usage.total_tokens = 40

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage = mock_usage
        mock_completion.model = DEFAULT_MODEL

        mock_instance.chat.completions.create.return_value = mock_completion

        client = GroqClient(config=mock_config)
        messages = [{"role": "user", "content": "Write an addition function."}]
        response = client.chat(messages=messages)

        assert isinstance(response, LLMResponse)
        assert response.content == "Here is a Python function: def add(a, b): return a + b"
        assert response.finish_reason == "stop"
        assert response.prompt_tokens == 15
        assert response.completion_tokens == 25
        assert response.total_tokens == 40
        assert not response.has_tool_calls

    @patch("agent.llm_client.Groq")
    def test_chat_with_tool_call_parsing(self, mock_groq_class, mock_config):
        mock_instance = MagicMock()
        mock_groq_class.return_value = mock_instance

        # Simulate LLM deciding to invoke 'create_file'
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_abc123"
        mock_tool_call.function.name = "create_file"
        mock_tool_call.function.arguments = json.dumps({
            "filename": "hello.py",
            "content": "print('Hello world!')"
        })

        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_choice.message.tool_calls = [mock_tool_call]
        mock_choice.finish_reason = "tool_calls"

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage = MagicMock(prompt_tokens=30, completion_tokens=20, total_tokens=50)
        mock_completion.model = DEFAULT_MODEL

        mock_instance.chat.completions.create.return_value = mock_completion

        client = GroqClient(config=mock_config)
        response = client.chat(messages=[{"role": "user", "content": "Save hello.py"}])

        assert response.has_tool_calls is True
        assert len(response.tool_calls) == 1
        tc = response.tool_calls[0]
        assert tc.id == "call_abc123"
        assert tc.tool_name == "create_file"
        assert tc.arguments["filename"] == "hello.py"
        assert tc.arguments["content"] == "print('Hello world!')"

    @patch("agent.llm_client.Groq")
    def test_authentication_error_transformation(self, mock_groq_class, mock_config):
        mock_instance = MagicMock()
        mock_groq_class.return_value = mock_instance

        # Create dummy request and response for AuthenticationError
        mock_req = MagicMock()
        mock_res = MagicMock(status_code=401)
        mock_instance.chat.completions.create.side_effect = groq.AuthenticationError(
            message="Invalid API Key",
            response=mock_res,
            body={"error": "unauthorized"}
        )

        client = GroqClient(config=mock_config)
        with pytest.raises(LLMAuthenticationError) as exc:
            client.chat(messages=[{"role": "user", "content": "hi"}])
        assert "Invalid Groq API Key" in str(exc.value)

    @patch("agent.llm_client.Groq")
    def test_rate_limit_error_transformation(self, mock_groq_class, mock_config):
        mock_instance = MagicMock()
        mock_groq_class.return_value = mock_instance

        mock_res = MagicMock(status_code=429)
        mock_instance.chat.completions.create.side_effect = groq.RateLimitError(
            message="Rate limit hit",
            response=mock_res,
            body={"error": "rate_limited"}
        )

        client = GroqClient(config=mock_config)
        with pytest.raises(LLMRateLimitError) as exc:
            client.chat(messages=[{"role": "user", "content": "hi"}])
        assert "rate limit reached" in str(exc.value)

    @patch("agent.llm_client.Groq")
    def test_ask_convenience_helper(self, mock_groq_class, mock_config):
        mock_instance = MagicMock()
        mock_groq_class.return_value = mock_instance

        mock_choice = MagicMock()
        mock_choice.message.content = "Quick answer."
        mock_choice.message.tool_calls = None
        mock_choice.finish_reason = "stop"

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage = MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8)
        mock_completion.model = DEFAULT_MODEL

        mock_instance.chat.completions.create.return_value = mock_completion

        client = GroqClient(config=mock_config)
        answer = client.ask("What is 2+2?", system_prompt="You are a math tutor.")
        assert answer == "Quick answer."

    @patch("agent.llm_client.Groq")
    def test_model_not_found_graceful_handling(self, mock_groq_class, mock_config):
        mock_instance = MagicMock()
        mock_groq_class.return_value = mock_instance

        # Mock models.list()
        mock_model_item = MagicMock()
        mock_model_item.id = "openai/gpt-oss-120b"
        mock_instance.models.list.return_value = MagicMock(data=[mock_model_item])

        # Mock 404 model_not_found error
        mock_res = MagicMock(status_code=404)
        mock_instance.chat.completions.create.side_effect = groq.APIStatusError(
            message="Error code: 404 - {'code': 'model_not_found'}",
            response=mock_res,
            body={"error": {"code": "model_not_found"}}
        )

        client = GroqClient(config=mock_config)
        with pytest.raises(LLMClientError) as exc:
            client.chat(messages=[{"role": "user", "content": "hi"}])

        assert "is not available for your Groq key" in str(exc.value)
        assert "Available:" in str(exc.value)
