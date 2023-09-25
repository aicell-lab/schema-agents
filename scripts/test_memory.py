import asyncio
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message, MemoryChunk
from schema_agents.tools.code_interpreter import create_mock_client
from schema_agents.memory.long_term_memory import LongTermMemory

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

from metagpt.document_store.faiss_store import FaissStore
from metagpt.logs import logger

class FunctionMemory(BaseModel):
    """Message of functions to be saved in faiss store."""
    function_name: str = Field(default="", description="Function name")
    code: str = Field(default="", description="original code of the function")
    lang: str = Field(default="", description="function language")
    args: List[str] = Field(default=[], description="arguments of the function")
    

INIT_SCRIPT = """
def microscope_move(position):
    print(f"===> Moving to: {position}")

def microscope_snap(config):
    print(f"===> Snapped an image with exposure {config['exposure']} and saved to: { config['path']}")
"""


def create_long_term_memory():
    memory = LongTermMemory()
    role_id = 'bio'
    memory.recover_memory(role_id)
    memory.clean()
    assert len(memory.recover_memory(role_id)) == 0

    function_move = FunctionMemory(function_name='microscope_move', code="""def microscope_move(position):
        print(f"===> Moving to: {position}")""", lang='python', args=['position'])
    function_snap = FunctionMemory(function_name='microscope_snap', code="""def microscope_snap(config):
        print(f"===> Snapped an image with exposure {config['exposure']} and saved to: { config['path']}")""", lang='python', args=['config'])

    print(memory.is_initialized)
    new_memory = MemoryChunk(index='microscope move python function',content=function_move, category='function')
    memory.add(new_memory)
    print(memory.is_initialized)
    new_memory = MemoryChunk(index='microscope snap python function',content=function_snap, category='function')
    memory.add(new_memory)
    new_error = MemoryChunk(index='Error made for microscope move function', category='error')
    memory.add(new_error)

    memories = memory.recover_memory(role_id)
    assert len(memories) == 3
    return memory


def test_role_memory():
    Microscopist = Role.create(
        name="Thomas",
        profile="test_Microscopist",
        goal="Acquire images from the microscope based on user's requests.",
        constraints=None,
        actions=[],
        long_term_memory=create_long_term_memory(),
    )
    
    ms = Microscopist()

    query = 'get microscope related functions'
    resp = ms.long_term_memory.retrieve(query, filter={"category": "error"})
    print(resp)

if __name__ == "__main__":
    # test_idea_message()
    # print(len(test_message_pydantic()))    # print(docs[0])
    test_role_memory()
    # test_faiss_store()

