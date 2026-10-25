"""Tool registry: turns plain Python functions into OpenAI function-calling tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

# Tool output goes back into the model's context window, so cap it.
MAX_RESULT_CHARS = 4000
# A tool may put extra data for the UI under this key. It is removed before the
# result goes to the model, so it costs no tokens.
DETAILS_KEY = "_details"


@dataclass(frozen=True)
class ToolOutput:
    text: str  # JSON string sent to the model
    details: Any = None  # UI-only data, never sent to the model


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the function arguments
    func: Callable[..., dict[str, Any]]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool] | tuple[Tool, ...] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def call(self, name: str, arguments: str) -> str:
        """Run a tool and return its result as a JSON string (see run())."""
        return self.run(name, arguments).text

    def run(self, name: str, arguments: str) -> ToolOutput:
        """Run a tool. The text is a JSON string for the model; details are for the UI.

        Errors are returned to the model as {"error": ...} instead of raised,
        so the agent can correct its arguments and try again.
        """
        tool = self._tools.get(name)
        if tool is None:
            return _error(f"Unknown tool '{name}'. Available tools: {', '.join(self._tools)}")

        try:
            kwargs = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return _error(f"Arguments are not valid JSON: {exc}")
        if not isinstance(kwargs, dict):
            return _error("Arguments must be a JSON object")

        try:
            result = tool.func(**kwargs)
        except Exception as exc:  # noqa: BLE001 - any tool failure is reported back to the model
            return _error(f"{type(exc).__name__}: {exc}")

        details = None
        if isinstance(result, dict) and DETAILS_KEY in result:
            result = dict(result)
            details = result.pop(DETAILS_KEY)
        return ToolOutput(text=_dump(result), details=details)


def _error(message: str) -> ToolOutput:
    return ToolOutput(text=_dump({"error": message}))


def _dump(obj: Any) -> str:
    text = json.dumps(obj, ensure_ascii=False, default=str)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + " ...[truncated]"
    return text
