from agent.tools import dosing, drug_info, knowledge
from agent.tools.base import Tool, ToolRegistry


def build_default_registry() -> ToolRegistry:
    return ToolRegistry([knowledge.TOOL, drug_info.TOOL, dosing.TOOL])


__all__ = ["Tool", "ToolRegistry", "build_default_registry"]
