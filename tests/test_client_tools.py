from llm.engine import _try_parse_client_tool_call

ALLOWED = {"policy_search"}

TESTS = [
    '{"tool":"policy_search","arguments":{"policy_number":"DH123"}}',
    '{"name":"policy_search","input":{"policy_number":"DH123"}}',
    '{"type":"tool_use","name":"policy_search","input":{"policy_number":"DH123"}}',
    '{"tool":"policy__policy_search","arguments":{"policy_number":"DH123"}}',
]


def test_client_tool_parsing():
    for sample in TESTS:
        result = _try_parse_client_tool_call(sample, ALLOWED)
        assert result is not None, sample
        assert result["tool"] == "policy_search", result
        assert result["arguments"]["policy_number"] == "DH123"
