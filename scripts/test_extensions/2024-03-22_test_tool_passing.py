import os
from enum import Enum
import inspect
from schema_agents.provider.openai_api import retry
import asyncio
from typing import List, Optional, Union, Type, Any, get_type_hints, Tuple, Literal
import asyncio
from schema_agents.role import Message
from schema_agents import schema_tool, Role
from pydantic import BaseModel, Field
from .graph_adder import plot_metdata
from functools import partial
import json
from langchain.schema.document import Document
from typing import Callable

from .tools.NCBI import get_geo_api_info, get_genomic_api_info, ncbi_api_call, get_pubmed_api_info, get_pubmed_central_oa
from .tools.fileIO import write_to_file, read_file, ftp_download, unzip_tar_gz, list_files_in_dir, get_current_time
from .tools.llm_web_search import search_pubmed_paper
from .tools.paperqa_tools import ask_pdf_paper
from .tools.agents import ThoughtsSchema, recruit_agent
from .serialize import dump_metadata_json
from .visualize_reasoning import visualize_reasoning_chain

import importlib.util
import sys

INSTRUCTIONS = """
- You are the overall task manager. Your job is to complete the user's task completely. 
"""
class RecruitTool(BaseModel):
    name : str = Field(description = "The name of the tool")
    # usage: str = Field(description = "A description of the tool usage")
    docstring: str = Field(description = "The docstring of the tool")
    posix_path : str = Field(description = "The posix_path of the tool")

def get_function(t : RecruitTool):
    module_name = t.posix_path.rsplit('/', 1)[-1][:-3]  # Extracts the base '.py' file name then removes the '.py'
    # Import the module
    spec = importlib.util.spec_from_file_location(module_name, t.posix_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    # Access the function
    function = getattr(module, t.name)
    return function

@schema_tool
async def recruit_agent_local(agent_name : str = Field(description="The name of the agent to recruit"),
                        agent_instructions : str = Field(description = "The role instructions to give to the agent. This is a general description of the agent's role"),
                        query : str = Field(description = "The specific task to give to the agent"),
                        tools : List[RecruitTool] = Field(description = "The tools to use for the task")):
    """Recruit an agent to perform a specific task. Give the agent a name, instructions, a query, and the tools to use."""

    modified_instructions = agent_instructions + "\n\n" + "Use the `get_tool_info` tool to get usage information about the tools you will use."
    agent = Role(name=agent_name,
                    instructions = agent_instructions,
                    constraints=None,
                    register_default_events=True,
                    )
    response, metadata = await agent.acall(query,
                                tools = [get_function(t) for t in tools],
                                return_metadata=True,
                                max_loop_count = 10,
                                thoughts_schema=ThoughtsSchema,
                                )
    return response, metadata

async def main():

    # prompt = s + "\n\n" + "User question : What is the official gene symbol of LMP10?"
    manager_instructions =  INSTRUCTIONS
    manager = Role(
        name="manager",
        instructions = manager_instructions,
        constraints=None,
        register_default_events=True,
    )

    from .tools.tool_explorer import create_tool_db, search_tools, fixed_db_tool_search
    db_path = "tool_index"
    tool_db = await create_tool_db(tool_dir="/Users/gkreder/schema-agents/scripts/test_extensions/tools",
                             save_path = db_path,)
    search_tools = fixed_db_tool_search(fixed_db_path = db_path)

    tools = [search_tools, recruit_agent_local]

    # query = """Search for tools in the tool database that can help with the task of finding the official gene symbol of LMP10. Then pass these tools to a recruited agent to complete the task."""
    query = """Search for tools in the tool database that can help with the task of figuring out the main finding of the PDF located at /Users/gkreder/schema-agents/PMC10897392/41746_2024_Article_1038.pdf. Then pass these tools to a recruited agent to complete the task."""
    
    response, metadata = await manager.acall(query,
                                #    [ask_pdf_paper, search_web],
                                    tools,
                                   return_metadata=True,
                                   max_loop_count = 10,
                                   thoughts_schema=ThoughtsSchema,
                                   )
    
    with open('metadata_complete.txt', 'w') as f:
        print(metadata, file = f)
    metadata_json_fname = "metadata.json"
    dump_metadata_json(metadata, metadata_json_fname)
    with open(metadata_json_fname) as f:
        metadata_json = json.load(f)
    visualize_reasoning_chain(metadata_json, file_path_gv='reasoning_chain_visualization_auto.gv', view = True)
    print(response)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()






