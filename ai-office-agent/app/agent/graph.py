from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import plan_node, respond_node, should_use_tool, tool_node
from app.agent.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("plan_node", plan_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("respond_node", respond_node)

    graph.add_edge(START, "plan_node")
    graph.add_conditional_edges(
        "plan_node",
        should_use_tool,
        {
            "tool": "tool_node",
            "respond": "respond_node",
        },
    )
    graph.add_edge("tool_node", "plan_node")
    graph.add_edge("respond_node", END)
    return graph.compile()


@lru_cache(maxsize=1)
def get_agent_app():
    return build_graph()
