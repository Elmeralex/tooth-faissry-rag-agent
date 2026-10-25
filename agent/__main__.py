"""Run the agent from the terminal: uv run python -m agent "your question"."""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

from agent.loop import DEFAULT_MAX_STEPS, DEFAULT_MODEL, Step, run_agent


def print_step(step: Step) -> None:
    if step.type == "thought":
        print(f"\n[thought] {step.content}")
    elif step.type == "tool_call":
        print(f"\n[tool] {step.tool}({json.dumps(step.arguments, ensure_ascii=False)})  {step.seconds}s")
        print(f"  -> {step.result[:600]}")
    elif step.type == "answer":
        print(f"\n[answer]\n{step.content}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dental clinic agent")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = parser.parse_args()

    load_dotenv()
    question = " ".join(args.question).strip() or input("Question: ").strip()
    if not question:
        sys.exit("No question provided.")

    result = run_agent(question, model=args.model, max_steps=args.max_steps, on_step=print_step)
    print(
        f"\n--- {len(result.tool_calls)} tool calls, {result.model_calls} model calls, "
        f"{result.prompt_tokens}+{result.completion_tokens} tokens, stop: {result.stop_reason}"
    )


if __name__ == "__main__":
    main()
