
from pathlib import Path
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from typing import Callable
from langchain_core.tools import BaseTool, tool as create_tool
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from langgraph.prebuilt.interrupt import HumanInterruptConfig, HumanInterrupt
# An example of a sensitive tool that requires human review / approval


try:
    from ..proj_llm import llm_openai, llm_llama3
except Exception:
    path_llm = Path(__file__).parent.parent
    # add path to sys.path and env path
    import sys
    sys.path.append(str(path_llm))
    from proj_llm import llm_openai, llm_llama3, embeddings_openai, embeddings_hf, embeddings_llama3


llm = llm_llama3  # or llm_openai


def add_human_in_the_loop(
    tool: Callable | BaseTool,
    *,
    interrupt_config: HumanInterruptConfig = None,
) -> BaseTool:
    """Wrap a tool to support human-in-the-loop review."""
    if not isinstance(tool, BaseTool):
        tool = create_tool(tool)

    if interrupt_config is None:
        interrupt_config = {
            "allow_accept": True,
            "allow_edit": True,
            "allow_respond": True,
        }

    @create_tool(
        tool.name,
        description=tool.description,
        args_schema=tool.args_schema
    )
    def call_tool_with_interrupt(config: RunnableConfig, **tool_input):
        request: HumanInterrupt = {
            "action_request": {
                "action": tool.name,
                "args": tool_input
            },
            "config": interrupt_config,
            "description": "Please review the tool call"
        }
        response = interrupt([request])[0]
        # approve the tool call
        if response["type"] == "accept":
            tool_response = tool.invoke(tool_input, config)
        # update tool call args
        elif response["type"] == "edit":
            tool_input = response["args"]["args"]
            tool_response = tool.invoke(tool_input, config)
        # respond to the LLM with user feedback
        elif response["type"] == "response":
            user_feedback = response["args"]
            tool_response = user_feedback
        else:
            raise ValueError(f"Unsupported interrupt response type: {response['type']}")

        return tool_response

    return call_tool_with_interrupt


@create_tool
def book_hotel(hotel_name: str):
    """Book a hotel"""

    return f"Successfully booked a stay at {hotel_name}."


checkpointer = InMemorySaver()

agent = create_react_agent(
    model=llm,
    tools=[add_human_in_the_loop(book_hotel)],
    checkpointer=checkpointer,
)


config = {"configurable": {"thread_id": "1"}}


for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "book a stay at McKittrick hotel"}]},
    # Command(resume={"type": "accept"}),
    config
):
    print(chunk)
    print("\n")


for chunk_c in agent.stream(
    Command(resume={"type": "accept"}),
    # Command(resume=[{"type": "edit", "args": {"args": {"hotel_name": "McKittrick Hotel"}}}]),
    config
):
    print(chunk_c)
    print("\n")
