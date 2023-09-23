import asyncio
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.tools.code_interpreter import create_mock_client
from schema_agents.memory.longterm_memory import LongTermMemory
from schema_agents.memory.memory_storage import MemoryStorage

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

from metagpt.document_store.faiss_store import FaissStore
from metagpt.logs import logger

class MessageChunk(BaseModel):
    """Message of functions to be saved in faiss store."""
    function_name: str = Field(default="", description="Function name")
    metadata: str = Field(default="", description="original function")
    func_type: str = Field(default="", description="type of the function language")
    args: List[str] = Field(default=[], description="arguments of the function")
    

INIT_SCRIPT = """
def microscope_move(position):
    print(f"===> Moving to: {position}")

def microscope_snap(config):
    print(f"===> Snapped an image with exposure {config['exposure']} and saved to: { config['path']}")
"""

message_move = MessageChunk(function_name='microscope_move', metadata="""def microscope_move(position):
    print(f"===> Moving to: {position}")""", func_type='python', args=['position'])
message_snap = MessageChunk(function_name='microscope_snap', metadata="""def microscope_snap(config):
    print(f"===> Snapped an image with exposure {config['exposure']} and saved to: { config['path']}")""", func_type='python', args=['config'])

def test_idea_message():
    # idea = 'Write a cli snake game'
    role_id = '1'
    message = Message(role='BOSS', content=INIT_SCRIPT, cause_by=MessageChunk)

    memory_storage: MemoryStorage = MemoryStorage()
    messages = memory_storage.recover_memory(role_id)
    memory_storage.add(message)
    messages = memory_storage.recover_memory(role_id)
    # assert len(messages) == 0

    memory_storage.add(message)
    assert memory_storage.is_initialized is True

    sim_idea = 'Write a game of cli snake'
    sim_message = Message(role='BOSS', content=sim_idea, cause_by=MessageChunk)
    new_messages = memory_storage.search(sim_message)
    # assert len(new_messages) == 0   # similar, return []

    memory_storage.add(sim_message)
    search_message = memory_storage.search(sim_message)
    # assert len(search_message) == 1
    
    python_message = Message(role='BOSS', content='microsope python functions', cause_by=MicroscopeControlRequirements)
    search_python = memory_storage.search2(python_message)
    # new_idea = 'Write a 2048 web game'
    # new_message = Message(role='BOSS', content=new_idea, cause_by=MicroscopeControlRequirements)
    # new_messages = memory_storage.search(new_message)
    # assert new_messages[0].content == message.content

    memory_storage.clean()
    assert memory_storage.is_initialized is False

def test_txtloader():
    loader = TextLoader("/home/alalulu/workspace/schema-agents/.data/memory_text.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    query = "microscope functions"
    docs = db.similarity_search(query,1)
    return docs
    


def test_message_pydantic():
    role_id = 'bio'
    
    memory_store: MemoryStorage = MemoryStorage()
    messages = memory_store.recover_memory(role_id)
    memory_store.clean()
    messages = memory_store.recover_memory(role_id)
    assert len(messages) == 0
    
    
    message_pyd = Message(role='bio',content='microscope move python function',instruct_content=message_move)
    memory_store.add(message_pyd)


    message_pyd = Message(role='bio',content='microscope snap python function',instruct_content=message_snap)
    memory_store.add(message_pyd)

    query = 'load microscope control related functions'
    query_message = Message(role='age',content=query,cause_by=MessageChunk)
    search_message = memory_store.retrieve_by_query(query)
    
    return search_message

if __name__ == "__main__":
    # test_idea_message()
    print(len(test_message_pydantic()))    # print(docs[0])
    # test_faiss_store()

