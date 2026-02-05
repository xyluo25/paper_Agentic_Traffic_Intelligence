'''
##############################################################
# Created Date: Friday, June 27th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

from IPython.display import Image, display
import json
from typing import Optional
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage
from langchain_core.messages import AIMessage

from chat_bot_supervisor import SupervisorAgent, llm


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish

    try:
        # AIMessage is the last message
        if not last_message.tool_calls:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"
    except AttributeError:
        # ToolMessage is the last message
        if not last_message.tool_call_id:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"


# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = llm.invoke(messages)
    print("Call model response: ", response)

    assert len(response.tool_calls) <= 1, "Only one tool call is allowed at a time"

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the function to execute tools
def call_tool(state):
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    print(f"State messages: {messages}")
    tool_call = messages[-1].tool_calls[0]

    response = llm.invoke(messages)
    # We use the response to create a ToolMessage
    tool_message = ToolMessage(
        content=str(response),
        name=tool_call["name"],
        tool_call_id=tool_call["id"]
    )
    # We return a list, because this will get added to the existing list
    return {"messages": [tool_message]}


# define the graph
graph_builder = SupervisorAgent

# add the nodes to the graph
# graph_builder.add_node("agent", call_model)
graph_builder.add_node("action_rag", call_tool)
graph_builder.add_node("action_osm", call_tool)
graph_builder.add_node("action_sumo", call_tool)
graph_builder.add_node("action_rt", call_tool)


# Set the entry point from the agent
graph_builder.set_entry_point("supervisor")

# add conditional edges to enable human in the loop
graph_builder.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "osm_agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,

    {
        # If `tools`, then we call the tool node.
        "continue": "action_osm",
        # Otherwise we finish.
    },
)
graph_builder.add_conditional_edges(
    "rag_agent",
    should_continue,
    {
        "continue": "action_rag",
    },
)

graph_builder.add_conditional_edges(
    "realtwin_agent",
    should_continue,
    {
        "continue": "action_rt",
    },
)
graph_builder.add_conditional_edges(
    "sumo_agent",
    should_continue,
    {
        "continue": "action_sumo",
    },
)

# add edge from action(tools) to agent
graph_builder.add_edge("action_osm", "osm_agent")
graph_builder.add_edge("action_sumo", "sumo_agent")
graph_builder.add_edge("action_rt", "realtwin_agent")
graph_builder.add_edge("action_rag", "rag_agent")


HIL_Agent = graph_builder.compile(checkpointer=MemorySaver(),
                                  interrupt_before=["action_rag", "action_osm", "action_sumo", "action_rt"])

# Helper function to construct message asking for verification
def generate_verification_message(message: AIMessage) -> None:
    """Generate "verification message" from message with tool calls."""
    serialized_tool_calls = json.dumps(
        message.tool_calls,
        indent=2,
    )
    return AIMessage(
        content=(
            "I plan to invoke the following tools, do you approve?\n"
            "If you do not approve, I will stop and ask you for a different response.\n"
            "Type 'yes' if you approve, anything else to stop.\n"
            "Type 'y' if you do, anything else to stop.\n"
            f"{serialized_tool_calls}"
        ),
        id=message.id,
    )


# Helper function to stream output from the graph
def catch_tool_calls(inputs, thread) -> Optional[AIMessage]:
    """Stream app, catching tool calls."""
    tool_call_message = None
    for event in HIL_Agent.stream(inputs, thread, stream_mode="values"):
        messages = event["messages"]
        message = event["messages"][-1]
        print("messages: ", messages)
        print("catch_tool_calls event: ", event)
        print("catch_tool_calls message: ", message)
        # if isinstance(message, AIMessage) and message.tool_calls:
        #     tool_call_message = message
        if isinstance(message, AIMessage) and message.tool_calls:
            tool_call_message = message
        else:
            message.pretty_print()
            if isinstance(message, AIMessage):
                print(f"tool message: {message.content}")
                return message.content
    return tool_call_message
