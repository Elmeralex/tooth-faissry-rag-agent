"""Streamlit chat UI that shows the agent's reasoning and tool calls live.

Run from the repo root: uv run streamlit run agent/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from openai import OpenAIError  # noqa: E402

from agent.loop import DEFAULT_MAX_STEPS, DEFAULT_MODEL, Step, run_agent  # noqa: E402

SEARCH_TOOL = "search_dental_knowledge"
TOOL_ICONS = {SEARCH_TOOL: "🔍", "lookup_drug_info": "💊", "calculate_max_dose": "🧮"}


def tool_label(name: str | None) -> str:
    return f"{TOOL_ICONS.get(name or '', '🔧')} {name}"


def score(value: Any) -> str:
    return f"`{value:.4f}`" if isinstance(value, (int, float)) else "`None`"


def render_retrieval_details(details: dict[str, Any], container) -> None:
    """The same retrieval fields the RAG chatbot (src/interface_new.py) shows."""
    container.write(f"Query: `{details.get('query')}`")
    container.write(f"Accepted: `{details.get('accepted')}`")
    container.write(f"Source filter: `{details.get('source_filter')}`")
    container.write(f"Score: {score(details.get('score'))}")
    container.write(f"Vector score: {score(details.get('vector_score'))}")
    container.write(f"BM25 score: {score(details.get('bm25_score'))}")
    container.write(f"RRF score: {score(details.get('rrf_score'))}")
    container.write(f"Rerank score: {score(details.get('rerank_score'))}")
    container.write(f"Rerank accept threshold: `{details.get('rerank_threshold', 0.0):.2f}`")
    container.write("Retrieved chunk:")
    container.code(details.get("retrieved_chunk") or "None", language="text")
    container.write("Metadata:")
    container.json(details.get("metadata") or {})
    if not details.get("accepted"):
        container.error(details.get("reason") or "Not accepted")


def render_step(step: Step, container) -> None:
    if step.type == "thought":
        container.markdown(f"💭 {step.content}")
    elif step.type == "tool_call":
        container.markdown(f"**{tool_label(step.tool)}** · {step.seconds}s")
        container.code(json.dumps(step.arguments, ensure_ascii=False), language="json")
        if step.tool == SEARCH_TOOL and step.details:
            details_tab, model_tab = container.tabs(["Retrieval details", "Sent to model"])
            render_retrieval_details(step.details, details_tab)
            model_tab.code(step.result[:1500], language="json")
        else:
            container.code(step.result[:1500], language="json")


def render_tools_used(steps: list[Step]) -> None:
    calls = [s for s in steps if s.type == "tool_call"]
    if calls:
        st.caption("Tools used: " + " → ".join(tool_label(s.tool) for s in calls))
    else:
        st.caption("Tools used: none")


def render_trace(steps: list[Step]) -> None:
    tool_steps = [s for s in steps if s.type != "answer"]
    if not tool_steps:
        return
    calls = sum(1 for s in steps if s.type == "tool_call")
    with st.expander(f"Agent trace · {calls} tool call{'s' if calls != 1 else ''}"):
        for step in tool_steps:
            render_step(step, st)


def render_latest_retrieval() -> None:
    """Sidebar: the knowledge-base searches behind the latest answer."""
    st.subheader("Latest Retrieval Details")
    answers = [m for m in st.session_state.messages if m["role"] == "assistant"]
    if not answers:
        st.info("No retrieval has been run yet.")
        return

    searches = [s for s in answers[-1].get("steps", []) if s.type == "tool_call" and s.tool == SEARCH_TOOL]
    if not searches:
        st.info("The latest answer did not search the knowledge base.")
        return

    for i, step in enumerate(searches, start=1):
        if len(searches) > 1:
            st.markdown(f"**Search {i} of {len(searches)}**")
        if step.details:
            render_retrieval_details(step.details, st)
        else:  # the tool failed before retrieving, e.g. no FAISS index
            st.error(step.result)
        if i < len(searches):
            st.divider()


def render_disclaimer() -> None:
    st.markdown(
        """
        **Disclaimer**

        This chatbot is a research prototype designed to assist with dental questions. It retrieves information from a vector database built on the MedMCQA dataset and the California Dental Association.

        Can make mistakes. Check important info.
        """
    )


def render_sidebar_details() -> None:
    with st.sidebar:
        st.divider()
        st.header("🔬 Retrieval Information")
        render_latest_retrieval()
        st.divider()
        render_disclaimer()


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Dental Agent", page_icon="🦷")
    st.title("🦷 Dental Clinic Agent")
    st.caption("Tool-calling LLM agent · educational use only, not medical advice")

    with st.sidebar:
        model = st.text_input("Model", DEFAULT_MODEL)
        max_steps = st.slider("Max agent steps", 1, 10, DEFAULT_MAX_STEPS)
        if st.button("New chat"):
            st.session_state.messages = []

    st.session_state.setdefault("messages", [])
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_tools_used(message.get("steps", []))
                render_trace(message.get("steps", []))

    question = st.chat_input("Ask a dental question")
    if not question:
        render_sidebar_details()
        return

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.status("Thinking...", expanded=True)
        try:
            result = run_agent(
                question,
                model=model,
                max_steps=max_steps,
                history=history,
                on_step=lambda step: render_step(step, status),
            )
        except OpenAIError as exc:
            status.update(label="Model call failed", state="error")
            st.error(f"{exc}\n\nCheck OPENAI_API_KEY in your .env file.")
            render_sidebar_details()
            return

        status.update(
            label=(
                f"Done · {len(result.tool_calls)} tool calls · "
                f"{result.prompt_tokens + result.completion_tokens} tokens"
            ),
            state="complete",
            expanded=False,
        )
        st.markdown(result.answer)
        render_tools_used(result.steps)

    st.session_state.messages.append({"role": "assistant", "content": result.answer, "steps": result.steps})
    render_sidebar_details()


main()
