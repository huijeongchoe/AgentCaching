# ---------------------------------------------------------------------------
# LangGraph multi-agent workflow
# ---------------------------------------------------------------------------
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from system_prompt import build_system_prompt
from tools import TOOLS, _set_current_agent


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def _build_agent_node(agent_id: str, assigned_tasks: list[dict]):
    """Build a graph node for one research agent with specific DRB tasks."""

    task_descriptions = "\n".join(
        f"  - [Task {t['id']}] ({t['topic']}) {t['prompt']}"
        for t in assigned_tasks
    )

    def agent_node(state: dict) -> dict:
        _set_current_agent(agent_id)
        messages = state.get("messages", [])

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        llm_with_tools = llm.bind_tools(TOOLS)

        system_msg = {
            "role": "system",
            "content": build_system_prompt(agent_id, task_descriptions),
        }

        conversation = [system_msg] + messages
        max_iterations = 10

        for _ in range(max_iterations):
            response = llm_with_tools.invoke(conversation)
            if not response.tool_calls:
                conversation.append(response)
                break
            conversation.append(response)
            for tc in response.tool_calls:
                tool_fn = next((t for t in TOOLS if t.name == tc["name"]), None)
                if tool_fn:
                    tool_result = tool_fn.invoke(tc["args"])
                    conversation.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    })

        return {"messages": [conversation[-1]]}

    return agent_node


def build_graph(tasks: list[dict]) -> StateGraph:
    """Build a 4-agent LangGraph workflow with overlapping task assignments.

    Tasks are shared across agents to mirror real multi-agent systems where
    agents lack visibility into each other's work, producing both intra-
    and cross-agent duplication.
    """
    n = len(tasks)
    mid = n // 2

    # Overlapping splits
    groups = [
        tasks[:mid],                          # agent 0: first half
        tasks[mid // 2 : mid + mid // 2],     # agent 1: overlaps 0 & 2
        tasks[mid - mid // 2 : n - mid // 2], # agent 2: overlaps 1 & 3
        tasks[mid:],                          # agent 3: second half
    ]

    agent_names = [
        "agent_retriever_1",
        "agent_retriever_2",
        "agent_analyst_1",
        "agent_analyst_2",
    ]

    builder = StateGraph(AgentState)

    for agent_id, group in zip(agent_names, groups):
        node_fn = _build_agent_node(agent_id, group)
        builder.add_node(agent_id, node_fn)
        builder.add_edge(START, agent_id)
        builder.add_edge(agent_id, END)

    return builder.compile()
