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

from .tools.NCBI import get_geo_api_info, get_genomic_api_info, ncbi_api_call, get_pubmed_api_info, get_pubmed_central_oa
from .tools.fileIO import write_to_file, read_file, ftp_download, unzip_tar_gz, list_files_in_dir, get_current_time
from .tools.llm_web_search import search_pubmed_paper
from .tools.paperqa import ask_pdf_paper
from .tools.agents import ThoughtsSchema, recruit_agent
from .serialize import dump_metadata_json
from .visualize_reasoning import visualize_reasoning_chain

INSTRUCTIONS = """
- You are the overall task manager. Your job is to complete the user's task completely. 
- Check your final response (the last `CompleteUserQuery`). If it does not provide a satisfactory answer to the original task, revise your plan and try again. 
- Recruit as many agents as necessary to complete the task and modify the team if they perform poorly. You MUST only use the tools provided to you. Same for agents you recruit. They must ONLY use tools provided to them.
- You MUST acknowledge that you understand these instructions by having your first action be a `Message` with the text "Acknowledged. I understand the instructions followed by a list of tools that end in `_info`"
- Your second step MUST be to do a preliminary tool discovery step by calling all of the tools that end in `_info` and using the results to make a plan that you execute and modify.
"""

async def main():

    # prompt = s + "\n\n" + "User question : What is the official gene symbol of LMP10?"
    manager_instructions =  INSTRUCTIONS
    manager = Role(
        name="manager",
        instructions = manager_instructions,
        constraints=None,
        register_default_events=True,
    )

    query = """Use the `get_pubmed_central_oa` tool to check if the following PubMed Central IDs correspond to open access articles: PMC1790863, PMC10500329, and PMC10669231."""
    
    
    # query = """Make a plan and recruit agents to complete the following task: `Take the PubMed Central articles with IDs PMC1790863, PMC10500329, and PMC10669231, identify the second author in each of them, and create a single tsv file listing the paper titles, journal, and second author name`. Do this by recruiting individual agents for each paper"""

    # tools = [get_geo_api_info, get_genomic_api_info, ncbi_api_call, write_to_file, read_file, get_pubmed_api_info]
    # tools = [get_pubmed_central_oa, write_to_file, read_file, ftp_download, unzip_tar_gz, search_pubmed_paper, list_files_in_dir, ask_pdf_paper]
    tools_static = [get_pubmed_central_oa, write_to_file, read_file, 
             ftp_download, unzip_tar_gz, search_pubmed_paper, 
             list_files_in_dir, ask_pdf_paper, get_geo_api_info, 
             get_genomic_api_info, ncbi_api_call, get_pubmed_api_info, get_current_time]
    
    @schema_tool
    async def recruit_agent_local(agent_name : str = Field(description="The name of the agent to recruit"),
                            agent_instructions : str = Field(description = "The role instructions to give to the agent. This is a general description of the agent's role"),
                            query : str = Field(description = "The specific task to give to the agent"),):
        """Recruit an agent to perform a specific task. Give the agent a name, instructions, a query, and the tools to use"""
        agent = Role(name=agent_name,
                        instructions = agent_instructions,
                        constraints=None,
                        register_default_events=True,
                        )
        response, metadata = await agent.acall(query,
                                    tools_static,
                                    return_metadata=True,
                                    max_loop_count = 10,
                                    thoughts_schema=ThoughtsSchema,
                                    )
        return response, metadata

    tools = tools_static + [recruit_agent_local]
    response, metadata = await manager.acall(query,
                                #    [ask_pdf_paper, search_web],
                                    tools,
                                   return_metadata=True,
                                   max_loop_count = 10,
                                   thoughts_schema=ThoughtsSchema,
                                   )
    
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






