"""
Unit tests for Stage 2 features:
1. Repeated-failure guard in agent.py
2. Output truncation for large stdout/stderr/content
3. Conversation history compression and ceiling
4. LLM client retry logic on RateLimitError and APIConnectionError
5. Malformed JSON handling in tool calls
"""

from unittest.mock import MagicMock, patch
import pytest
from groq import RateLimitError, APIConnectionError

from config import Config
from agent.agent import CodeMateAgent, AgentResult, truncate_tool_output, compress_conversation_history
from agent.llm_client import GroqClient, LLMResponse, ToolCallRequest, LLMRateLimitError, LLMConnectionError
from tools.tool_registry import ToolRegistry


@pytest.fixture
def stage2_config(tmp_path):
    ws = tmp_path / "stage2_ws"
    ws.mkdir(parents=True, exist_ok=True)
    return Config(
        groq_api_key="mock_key",
        model_name="llama-3.3-70b-versatile",
        max_iterations=10,
        execution_timeout_seconds=5,
        require_user_approval=True,
        workspace_dir=ws,
        project_root=tmp_path,
        max_tool_output_chars=100,
        max_history_messages=10
    )


class TestOutputTruncation:
    """Test output truncation logic."""

    def test_truncate_tool_output_short_text_unchanged(self):
        res = {"success": True, "stdout": "hello", "stderr": ""}
        truncated = truncate_tool_output(res, max_chars=100)
        assert truncated["stdout"] == "hello"

    def test_truncate_tool_output_long_text_truncated(self):
        long_str = "x" * 250
        res = {"success": True, "stdout": long_str, "stderr": "error" + "y" * 150}
        truncated = truncate_tool_output(res, max_chars=100)
        assert len(truncated["stdout"]) < 250
        assert "...[truncated 150 chars]" in truncated["stdout"]
        assert "...[truncated" in truncated["stderr"]


class TestHistoryCompression:
    """Test conversation history compression preserving system & original task."""

    def test_history_within_limit_unchanged(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "done"}
        ]
        res = compress_conversation_history(msgs, max_messages=10)
        assert len(res) == 3

    def test_history_exceeded_preserves_system_and_task(self):
        msgs = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "initial task"}
        ]
        # Add 15 dummy assistant/tool message pairs
        for i in range(15):
            msgs.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": f"call_{i}", "function": {"name": "read_file"}}]
            })
            msgs.append({"role": "tool", "tool_call_id": f"call_{i}", "name": "read_file", "content": "{}"})

        compressed = compress_conversation_history(msgs, max_messages=10)
        assert compressed[0]["role"] == "system"
        assert compressed[0]["content"] == "system prompt"
        assert compressed[1]["role"] == "user"
        assert compressed[1]["content"] == "initial task"
        assert len(compressed) <= 15  # Significantly compacted


class TestRepeatedFailureGuard:
    """Test that repeating the identical failing tool action 3 times terminates with repeated_failure."""

    def test_repeated_failure_terminates_agent(self, stage2_config):
        registry = ToolRegistry()
        # Register a tool that always fails
        failing_tool = MagicMock(return_value={"success": False, "error": "Disk locked error"})
        registry.register("bad_tool", "Always fails", failing_tool)

        mock_llm = MagicMock()
        mock_llm.format_tool_schema.return_value = []
        # Return the identical failing tool call on every chat call
        failing_call = ToolCallRequest(id="call_fail", tool_name="bad_tool", arguments={"target": "abc"})
        mock_llm.chat.return_value = LLMResponse(content="", tool_calls=[failing_call])

        agent = CodeMateAgent(config=stage2_config, registry=registry, llm_client=mock_llm, auto_approve=True)
        result = agent.run("Do something")

        assert result.success is False
        assert result.termination_reason == "repeated_failure"
        assert "Repeated failure threshold exceeded" in result.final_response


class TestLLMClientRetryAndMalformedJSON:
    """Test retry on API errors and safe handling of malformed JSON."""

    @patch("agent.llm_client.time.sleep", return_value=None)
    def test_retry_on_rate_limit_exceeded(self, mock_sleep, stage2_config):
        client = GroqClient(config=stage2_config)
        mock_raw_client = MagicMock()
        # Mock client to raise RateLimitError twice then succeed
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Success"
        mock_resp.choices[0].message.tool_calls = None
        mock_resp.choices[0].finish_reason = "stop"
        mock_resp.usage = None
        mock_resp.model = "llama-3.3-70b-versatile"

        rate_err = RateLimitError(message="Rate limit", response=MagicMock(status_code=429), body=None)
        mock_raw_client.chat.completions.create.side_effect = [rate_err, rate_err, mock_resp]
        client._client = mock_raw_client

        res = client.chat(messages=[{"role": "user", "content": "hi"}])
        assert res.content == "Success"
        assert mock_raw_client.chat.completions.create.call_count == 3

    def test_malformed_json_in_tool_call_handled_safely(self, stage2_config):
        client = GroqClient(config=stage2_config)
        mock_raw_client = MagicMock()

        mock_tc = MagicMock()
        mock_tc.id = "tc_bad"
        mock_tc.function.name = "create_file"
        mock_tc.function.arguments = "{not valid json: 123"

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = ""
        mock_resp.choices[0].message.tool_calls = [mock_tc]
        mock_resp.choices[0].finish_reason = "tool_calls"
        mock_resp.usage = None
        mock_resp.model = "llama-3.3-70b-versatile"

        mock_raw_client.chat.completions.create.return_value = mock_resp
        client._client = mock_raw_client

        res = client.chat(messages=[{"role": "user", "content": "hi"}])
        assert len(res.tool_calls) == 1
        assert res.tool_calls[0].arguments.get("__error__") == "malformed_json"
