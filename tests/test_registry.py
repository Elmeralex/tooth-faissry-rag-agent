import json

from agent.tools import Tool, ToolRegistry, build_default_registry
from agent.tools import base


def make_registry(func):
    return ToolRegistry(
        [Tool(name="echo", description="Echo", parameters={"type": "object", "properties": {}}, func=func)]
    )


def test_default_registry_schemas_are_openai_format():
    schemas = build_default_registry().schemas()
    names = [s["function"]["name"] for s in schemas]
    assert names == ["search_dental_knowledge", "lookup_drug_info", "calculate_max_dose"]
    for schema in schemas:
        assert schema["type"] == "function"
        assert schema["function"]["parameters"]["type"] == "object"
        assert schema["function"]["description"]


def test_call_returns_json():
    registry = make_registry(lambda text: {"echo": text})
    assert json.loads(registry.call("echo", '{"text": "hi"}')) == {"echo": "hi"}


def test_errors_are_returned_not_raised():
    def boom(**_):
        raise ValueError("bad weight")

    registry = make_registry(boom)
    assert "Unknown tool" in json.loads(registry.call("nope", "{}"))["error"]
    assert "not valid JSON" in json.loads(registry.call("echo", "{oops"))["error"]
    assert "must be a JSON object" in json.loads(registry.call("echo", "[1, 2]"))["error"]
    assert json.loads(registry.call("echo", "{}"))["error"] == "ValueError: bad weight"


def test_wrong_argument_names_are_reported():
    registry = make_registry(lambda text: {"echo": text})
    assert "TypeError" in json.loads(registry.call("echo", '{"txt": "hi"}'))["error"]


def test_long_results_are_truncated():
    registry = make_registry(lambda: {"data": "x" * (base.MAX_RESULT_CHARS * 2)})
    output = registry.call("echo", "{}")
    assert output.endswith("...[truncated]")
    assert len(output) < base.MAX_RESULT_CHARS + 50


def test_details_are_kept_out_of_the_model_text():
    registry = make_registry(lambda: {"found": True, base.DETAILS_KEY: {"bm25_score": 9.4}})
    output = registry.run("echo", "{}")
    assert json.loads(output.text) == {"found": True}
    assert output.details == {"bm25_score": 9.4}
    assert registry.call("echo", "{}") == output.text
