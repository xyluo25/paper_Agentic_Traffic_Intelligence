'''
##############################################################
# Created Date: Friday, June 20th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''


import os
from pathlib import Path
import shutil

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema.messages import SystemMessage

# Load Data sources and save to vector store
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader, TextLoader

try:
    from ..proj_llm import llm_openai, llm_llama3
except Exception:
    path_llm = Path(__file__).parent.parent
    # add path to sys.path and env path
    import sys
    sys.path.append(str(path_llm))
    from proj_llm import llm_openai, llm_llama3, embeddings_openai, embeddings_hf, embeddings_llama3

# LLM
llm = llm_openai  # or llm_llama3

# embeddings
embeddings = embeddings_openai  # or embeddings_hf, embeddings_llama3


# 1. Load local PDF and text documents from the rag_datasets folder
def load_documents(folder_path: str):
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    # text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.lower().endswith('.pdf'):
            # loader = PyMuPDFLoader(filepath)
            # loader = PyPDFLoader(filepath)
            continue
        elif filename.lower().endswith('.txt'):
            print(f"  :Loading file: {filename}")
            loader = TextLoader(filepath)
        else:
            continue
        raw_documents = loader.load()
        splitted_documents = text_splitter.split_documents(raw_documents)
        documents.extend(splitted_documents)
    return documents

PERSIST_DIR = str(Path(__file__).parent / "chroma_store")
DOCS_FOLDER = str(Path(__file__).parent / "rag_datasets")

# delete the folder if it exists
if Path(PERSIST_DIR).exists():
    shutil.rmtree(PERSIST_DIR)
    print("  :Deleted pre-existing Chroma store directory for RAG tools.")

# not present or empty → build & persist
documents = load_documents(DOCS_FOLDER)
vector_store = Chroma.from_documents(documents,
                                     embeddings,
                                     persist_directory=PERSIST_DIR)
# vector_store.persist()   # explicitly write files to disk

retriever = vector_store.as_retriever(search_kwargs={"k": len(documents)})



@tool
def rag_tool(question: str) -> str:
    """This tool uses RAG to answer questions based on local documents and knowledge, specifically designed for Real-Twin and Develop team info."""
    return retriever.invoke(question)


@tool
def rag_tool_sim_parameters(question: str) -> str:
    """This tool uses RAG to answer questions based on local documents and knowledge, specifically designed for transportation simulation parameter recommendation, for example, lane changing parameter recommended values and ranges, car-following behavior parameters, Please note, your response need to include these required parameters: min_gap, acceleration, deceleration, sigma, tau, energencyDecel. You can include addition parameters as well."""
    return retriever.invoke(question)


agent_RAG = create_react_agent(
    model=llm,
    tools=[rag_tool],
    prompt=SystemMessage(content="""You are a Reasoning Agent with access to `rag_tool`.
        Follow this loop strictly for every user query:

        1. Thought: Reflect on whether you need to query `rag_tool`.
        2. Action: If you do, invoke `rag_tool["<your question>"]`.
        3. Observation: Record the result returned by `rag_tool`.
        4. (Repeat Thought/Action/Observation as needed.)
        5. Answer: Provide the final response to the user, incorporating any retrieved information.

        Attention, You must call `rag_tool` at least once before giving your Answer.
        Do NOT jump directly to the final Answer without retrieval.
        """),
    name="rag_agent",
)


if __name__ == "__main__":

    question = "What is ORNL ARMS group doing and how many members in the group?"

    input_data = {
        "messages": [
            {"role": "user", "content": f"{question}"}
        ]
    }
    # result = rag_chain.invoke({'input': question})
    answer = agent_RAG.invoke(input_data)
    print("Agent's final reply:\n", answer["messages"][-1].content)
