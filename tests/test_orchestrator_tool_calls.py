from fanout_openrouter.orchestrator import _format_candidate_response
from fanout_openrouter.openrouter_client import CompletionResult


def test_format_candidate_with_tool_calls():
    result = CompletionResult(
        content="I will check the weather.",
        model="test-model",
        provider="test-provider",
        system_fingerprint=None,
        choice={},
        usage=None,
        tool_calls=[
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ],
    )
    formatted = _format_candidate_response(result)
    assert "I will check the weather." in formatted
    assert '[Tool Call: get_weather({"city": "Paris"})]' in formatted

    result_no_content = CompletionResult(
        content="",
        model="test-model",
        provider="test-provider",
        system_fingerprint=None,
        choice={},
        usage=None,
        tool_calls=[
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ],
    )
    formatted2 = _format_candidate_response(result_no_content)
    assert '[Tool Call: get_weather({"city": "Paris"})]' in formatted2
