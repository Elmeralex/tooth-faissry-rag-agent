"""A hand-written OpenAI function-calling loop (ReAct style).

    user question -> model -> tool calls? --yes--> run tools -> add results -> model -> ...
                                         --no---> final answer

Written without an agent framework on purpose, so every step is visible.
Week 2 ports this to LangGraph.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.prompts import SYSTEM_PROMPT
from agent.tools import ToolRegistry, build_default_registry

DEFAULT_MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")
DEFAULT_MAX_STEPS = 6
FINAL_ANSWER_NUDGE = "Step limit reached. Answer now using only the tool results above."


@dataclass
class Step:
    type: str  # "thought" | "tool_call" | "answer"
    content: str = ""
    tool: str | None = None
    arguments: Any = None
    result: str | None = None
    details: Any = None  # extra tool data for the UI (e.g. retrieval scores); not sent to the model
    seconds: float = 0.0


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_calls: int = 0
    stop_reason: str = "answer"  # "answer" | "max_steps"

    @property
    def tool_calls(self) -> list[Step]:
        return [s for s in self.steps if s.type == "tool_call"]


def run_agent(
    question: str,
    *,
    client: Any = None,
    registry: ToolRegistry | None = None,
    model: str = DEFAULT_MODEL,
    max_steps: int = DEFAULT_MAX_STEPS,
    history: list[dict[str, str]] | None = None,
    on_step: Callable[[Step], None] | None = None,
) -> AgentResult:
    """Answer one question, calling tools as needed.

    history: earlier turns as [{"role": "user"|"assistant", "content": str}].
    on_step: called with each Step as it happens (for live UIs).
    """
    if client is None:
        from openai import OpenAI

        client = OpenAI()  # reads OPENAI_API_KEY (Can be change into ollama)
    registry = registry or build_default_registry()
    tools = registry.schemas()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *(history or []),
        {"role": "user", "content": question},
    ]
    result = AgentResult(answer="")

    def record(step: Step) -> None:
        result.steps.append(step)
        if on_step:
            on_step(step)

    def ask_model(**kwargs: Any) -> Any:
        response = client.chat.completions.create(model=model, messages=messages, tools=tools, **kwargs)
        result.model_calls += 1
        usage = getattr(response, "usage", None)
        if usage:
            result.prompt_tokens += usage.prompt_tokens or 0
            result.completion_tokens += usage.completion_tokens or 0
        return response.choices[0].message

    for _ in range(max_steps):
        message = ask_model(tool_choice="auto")

        if not message.tool_calls:
            result.answer = message.content or ""
            record(Step(type="answer", content=result.answer))
            return result

        if message.content:  # the "Thought" in ReAct: the model's stated plan
            record(Step(type="thought", content=message.content))

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.function.name, "arguments": call.function.arguments},
                    }
                    for call in message.tool_calls
                ],
            }
        )

        for call in message.tool_calls:  # the "Action" and "Observation"
            started = time.perf_counter()
            output = registry.run(call.function.name, call.function.arguments)
            record(
                Step(
                    type="tool_call",
                    tool=call.function.name,
                    arguments=_parse_json(call.function.arguments),
                    result=output.text,
                    details=output.details,
                    seconds=round(time.perf_counter() - started, 2),
                )
            )
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output.text})

    # Out of steps: force a final answer with tools switched off.
    messages.append({"role": "user", "content": FINAL_ANSWER_NUDGE})
    message = ask_model(tool_choice="none")
    result.answer = message.content or ""
    result.stop_reason = "max_steps"
    record(Step(type="answer", content=result.answer))
    return result


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
