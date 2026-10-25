"""Agent loop tests with a scripted fake model: no network, no API key."""

import copy
import json
from types import SimpleNamespace

from agent.loop import FINAL_ANSWER_NUDGE, run_agent
from agent.tools import Tool, ToolRegistry


class FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return self.replies.pop(0)


def reply(content=None, tool_calls=(), prompt_tokens=10, completion_tokens=5):
    calls = [
        SimpleNamespace(id=f"call_{i}", type="function", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
        for i, (name, args) in enumerate(tool_calls)
    ]
    message = SimpleNamespace(content=content, tool_calls=calls or None)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def fake_client(*replies):
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(replies)))


def add_registry():
    return ToolRegistry(
        [
            Tool(
                name="add",
                description="Add two numbers",
                parameters={"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
                func=lambda a, b: {"sum": a + b},
            )
        ]
    )


def test_answers_directly_without_tools():
    client = fake_client(reply("Hello!"))
    result = run_agent("Hi", client=client, registry=add_registry())

    assert result.answer == "Hello!"
    assert [s.type for s in result.steps] == ["answer"]
    assert result.model_calls == 1


def test_tool_call_then_answer():
    client = fake_client(
        reply("I will add the numbers.", tool_calls=[("add", {"a": 2, "b": 3})]),
        reply("2 + 3 = 5"),
    )
    seen = []
    result = run_agent("What is 2+3?", client=client, registry=add_registry(), on_step=seen.append)

    assert result.answer == "2 + 3 = 5"
    assert [s.type for s in result.steps] == ["thought", "tool_call", "answer"]
    assert seen == result.steps
    call = result.tool_calls[0]
    assert call.tool == "add" and call.arguments == {"a": 2, "b": 3}
    assert json.loads(call.result) == {"sum": 5}
    assert (result.prompt_tokens, result.completion_tokens, result.model_calls) == (20, 10, 2)

    # The second request must carry the assistant's tool call and the matching tool result.
    second = client.chat.completions.requests[1]["messages"]
    assert second[-2]["role"] == "assistant"
    assert second[-2]["tool_calls"][0]["id"] == "call_0"
    assert second[-1] == {"role": "tool", "tool_call_id": "call_0", "content": '{"sum": 5}'}


def test_parallel_tool_calls_each_get_a_result():
    client = fake_client(
        reply(tool_calls=[("add", {"a": 1, "b": 1}), ("add", {"a": 2, "b": 2})]),
        reply("2 and 4"),
    )
    result = run_agent("Two sums", client=client, registry=add_registry())

    assert [json.loads(s.result)["sum"] for s in result.tool_calls] == [2, 4]
    tool_messages = [m for m in client.chat.completions.requests[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["call_0", "call_1"]


def test_tool_errors_go_back_to_the_model():
    client = fake_client(reply(tool_calls=[("subtract", {})]), reply("Sorry, I cannot subtract."))
    result = run_agent("5-3?", client=client, registry=add_registry())

    assert "Unknown tool" in result.tool_calls[0].result
    assert result.answer == "Sorry, I cannot subtract."


def test_step_limit_forces_a_final_answer_without_tools():
    looping = [reply(tool_calls=[("add", {"a": 1, "b": 1})]) for _ in range(2)]
    client = fake_client(*looping, reply("Best effort answer"))
    result = run_agent("Loop forever", client=client, registry=add_registry(), max_steps=2)

    assert result.stop_reason == "max_steps"
    assert result.answer == "Best effort answer"
    last = client.chat.completions.requests[-1]
    assert last["tool_choice"] == "none"
    assert last["messages"][-1] == {"role": "user", "content": FINAL_ANSWER_NUDGE}


def test_history_is_sent_before_the_new_question():
    client = fake_client(reply("Yes"))
    history = [{"role": "user", "content": "Earlier"}, {"role": "assistant", "content": "Reply"}]
    run_agent("Follow-up", client=client, registry=add_registry(), history=history)

    messages = client.chat.completions.requests[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[-1]["content"] == "Follow-up"


def test_tool_details_reach_the_step_but_not_the_model():
    registry = ToolRegistry(
        [
            Tool(
                name="search",
                description="Search",
                parameters={"type": "object", "properties": {}},
                func=lambda: {"found": True, "_details": {"rerank_score": 10.6}},
            )
        ]
    )
    client = fake_client(reply(tool_calls=[("search", {})]), reply("Done"))
    result = run_agent("Find it", client=client, registry=registry)

    assert result.tool_calls[0].details == {"rerank_score": 10.6}
    tool_message = client.chat.completions.requests[1]["messages"][-1]
    assert json.loads(tool_message["content"]) == {"found": True}
