'''
##############################################################
# Created Date: Thursday, June 26th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

# Your existing tools (must be decorated with @tool from langchain_core.tools)
from proj_tools import usr_defined_tools
from proj_llm import llm_openai, llm_llama3
from proj_rag import rag_tool, rag_tool_sim_parameters


# Initialize the LLM exactly as before
llm = llm_openai  # llm_llama3

# Read your custom system prompt
bot_prefix = open("./chat_prompt.txt", "r", encoding="utf-8").read()
bot_suffix = "IMPORTANT: If the tool's output is a dict, respond only with that dict as text literal."

# Define agents and tools if needed
sumo_agent = create_react_agent(
    model=llm,
    tools=usr_defined_tools.get("sumo_tools", []),
    name="sumo_agent",
    prompt=f"""You are a SUMO agent that can assist with SUMO related tasks. {bot_prefix}\n"
        {bot_suffix}"""
)

osm_agent = create_react_agent(
    model=llm,
    tools=usr_defined_tools.get("osm_tools", []),
    name="osm_agent",
    prompt=f"""You are an OSM agent that can assist with OpenStreetMap related tasks. {bot_prefix}\n"
        {bot_suffix}"""
)

realtwin_agent = create_react_agent(
    model=llm,
    tools=usr_defined_tools.get("realtwin_tools", []),
    name="realtwin_agent",
    prompt=f"""You are a RealTwin agent that can assist with RealTwin related tasks. {bot_prefix}\n"
        {bot_suffix}"""
)

rag_agent = create_react_agent(
    model=llm,
    tools=[rag_tool, rag_tool_sim_parameters],
    name="rag_agent",
    prompt=f"""You are a RAG agent that can assist with retrieval-augmented generation tasks for Develop team info, lane-changing parameters suggested values and ranges, car-following behavior suggested values and ranges. {bot_prefix}\n"
        {bot_suffix}"""
)

# Create supervisor workflow
SupervisorAgent = create_supervisor(
    agents=[sumo_agent, osm_agent, realtwin_agent, rag_agent],
    model=llm,
    tools=None,
    prompt="""You are a chatbot supervisor responsible for delegating user requests to specialized agents.
        Use **osm_agent** for OpenStreetMap tasks.
        Use **realtwin_agent** for RealTwin-related tasks, such as show, edit, save configurations, input generation and simulation.
        Use **rag_agent** for retrieval-augmented generation tasks for Develop team info, suggested values and ranges for lane-changing, car-following and behavior. Your response have to include: min_gap, acceleration, deceleration, sigma, tau, emergencyDecel. You can include additional parameters as well.
        Use **sumo_agent** for SUMO-related tasks.

        You may combine multiple agents when appropriate.
        Whenever a tool returns a dictionary or JSON, reply with **only** that literal dict/JSON.
        If a tool execution fails, retry until it succeeds.
        Always return the output of any executed tool. """,
    parallel_tool_calls=False,
    output_mode="full_history",  # "last_message"
)


# Compile for supervisor workflow
# class ChatBotSupervisor:
#     def __init__(self):
#         self.app = SupervisorAgent.compile(checkpointer=MemorySaver())
#     def dialogue(self, input_msg: str) -> str:
#         """ Send a message to the bot and get a response. """
#         print("Agentic Real-Twin Assistant is thinking, one sec...")
#         try:
#             result = self.app.invoke(
#                 input={"messages": [{"role": "user", "content": input_msg}]},
#                 config={"configurable": {"thread_id": "812"}},
#             )
#         except Exception as e:
#             return {"messages": [AIMessage(content=f"Error during dialogue: {e}")]}
#         # Return the full message history by default.
#         return result["messages"]


# Initialize bot
# try:
#     ChatBotDialog = ChatBotSupervisor()
# except Exception as e:
#     print(f"Error initializing the bot: {e}")
#     sys.exit(1)
