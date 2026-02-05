'''
##############################################################
# Created Date: Wednesday, July 9th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''
from pathlib import Path

try:
    from .rag_agent import agent_RAG, rag_tool, rag_tool_sim_parameters
except Exception:
    path_llm = Path(__file__).parent
    # add path to sys.path and env path
    import sys
    sys.path.append(str(path_llm))
    from rag_agent import agent_RAG, rag_tool, rag_tool_sim_parameters
