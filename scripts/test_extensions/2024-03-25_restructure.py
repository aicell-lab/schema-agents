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
from typing import Callable, Any

from .tools.NCBI import get_geo_api_info, get_genomic_api_info, ncbi_api_call, get_pubmed_api_info, get_pubmed_central_oa
from .tools.fileIO import write_to_file, read_file, ftp_download, unzip_tar_gz, list_files_in_dir, get_current_time
from .tools.llm_web_search import search_pubmed_paper
from .tools.paperqa_tools import ask_pdf_paper
from .tools.agents import ThoughtsSchema, recruit_agent
from .serialize import dump_metadata_json
from .visualize_reasoning import visualize_reasoning_chain

import importlib.util
import sys
import yaml

# MANAGER_INSTRUCTIONS = """
# You are the general task manager. Your job is to complete the user's task completely by searching tools, making a plan, then hiring agents. Your workflow should follow this logic:
# - Search for tools in the tool database using the `search_tools` tools that will be useful or necessary to complete the task.
# - Make a plan using `StartNewPlan` for how to use the tools to complete the task by recruiting agents.
# - Execute the plan you created by doing the following:
#     - Recruit agents one-by-one using the `recruit_agent_local` tool to solve a single step in the task using the subset of tools you found that are relevant to that step. Give the agent ALL the tools it needs to complete the task. There may be more than one tool needed to complete the task.
#     - Look at the agent's response and ask yourself "does this completely finish the user's task?". If the user's task is not complete, recruit another agent to complete the task using the information gained so far. If you need to recruit another agent, do so again with the subset of tools that are relevant to the next step.
# - Keep modifying and executing your plan until the entire user task is complete.
# """

MANAGER_INSTRUCTIONS = """
You are the general task manager. Your job is to complete the user's task completely by searching tools, making a plan, then hiring agents.
"""

HIRED_AGENT_INSTRUCTIONS = """
You are an agent hired for a specific task. Your job is to complete the task given to you by the manager. Your workflow should follow this logic:
- Read the instructions given to you by the manager.
- Check the tool usage for each tool you are given by the manager by invoking the `get_tool_usage` tool on each one of them.
- Make a plan using `StartNewPlan` to use the provided tools to complete your task. This plan might be multi-step.
- Execute the plan you created by doing the following:
    - Carry out the plan's tasks
    - Check if you have completed your assigned task from the manager.
- Keep modifying and executing your plan until the entire manager's task is complete.

The manager's instructions are as follows:\n{manager_instructions}
"""


class RecruitTool(BaseModel):
    """A tool that can be used for the task."""
    name : str = Field(description = "The name of the tool")
    relevant_info: str = Field(description = "The relevant information about the tool")
    posix_path : str = Field(description = "The posix_path of the tool")
    yaml_path : str = Field(description = "The yaml_path of the tool containing the tool's usage information")
    reasoning : str = Field(description = "The reasoning behind why this tool is useful for the task")

class UsefulTools(BaseModel):
    """A list of tools that are useful for the task."""
    tools : List[RecruitTool] = Field(description = "The tools to use for the task")

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
async def get_tool_usage(tool_name : str = Field(description="The name of the tool to get usage information for"),
                        yaml_path : str = Field(description = "The path to the yaml file containing the tool's usage information")): 
    """Get the usage information for the tool."""
    if not os.path.exists(yaml_path):
        return "No usage information provided for this tool."
    with open(yaml_path, 'r', encoding = 'utf-8') as yaml_file:
            usage_strings = yaml.safe_load(yaml_file)
    s = usage_strings.get(tool_name, {'usage' : "No usage string provided"})['usage']
    return s
            
class IntermediateTaskResponse(BaseModel):
    """Holds the response from an intermediate sub-task assigned to a recruited agent."""
    response : Any = Field(description = "The intermediate response from the agent")

# @schema_tool
# async def fire_agent

async def main():

    # prompt = s + "\n\n" + "User question : What is the official gene symbol of LMP10?"
    manager_instructions =  MANAGER_INSTRUCTIONS
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

    # tools = [search_tools, recruit_agent_local]

    # query = """Search for tools in the tool database that can help with the task of finding the official gene symbol of LMP10. Then pass these tools to a recruited agent to complete the task."""
    # query = """Search for tools in the tool database that can help with the task of figuring out the main finding of the PDF located at /Users/gkreder/schema-agents/PMC10897392/41746_2024_Article_1038.pdf. Then pass these tools to a recruited agent to complete the task."""
    # query = """Search for tools in the tool database that can help with the task of downloading and unzipping the PubMed Central article with ID 1790863. Then pass these tools to a recruited agent to complete the task."""
    query = """Download and unzip the PubMed Central article with ID 1790863"""

    task_responses = []
    class IsTaskComplete(str, Enum):
        user_task_complete = "User task complete"
        not_complete = "The user task is not complete."
    
    class ResponsePackage(BaseModel):
        """The response to the check if the user task is completed"""
        response : IsTaskComplete = Field(description = "A check of whether or not the user's task is complete.")
        last_response : str = Field(description = "The response from the most recently recruited agent")
        agent_response_chain : List[str] = Field(description = "The chain of responses from all the agents that have been recruited so far (ordered sequentially)")
        original_query : str = Field(description = "The original user task")
    
    # Wei : loop structure starts here
    max_manager_loops = 5
    current_loops = 0
    original_query = query
    response_package = ResponsePackage(response = IsTaskComplete.not_complete, 
                                       last_response = "No agents have been recruited yet.", 
                                       agent_response_chain = [],
                                       original_query = original_query)
    while True and current_loops < max_manager_loops:
        # force a tool search
        tool_prompt = f"""Your current task is to search for tools in the tool database that can help with this stage of solving the user's task. You will pass these tools to a recruited agent to complete the next step in solving the user task.

The original user task is this : `{response_package.original_query}`

The previous agent response was the following : `{response_package.last_response}`

The current chain of agent responses (ordered sequentially) is the following : `{response_package.agent_response_chain}`

Reason through the steps required to complete the user task at this stage and find some tools to give the next agent.
"""

        tool_response, tool_metadata = await manager.acall(tool_prompt,
                                                        [search_tools],
                                                        output_schema = UsefulTools,
                                                        return_metadata=True,
                                                        max_loop_count = 10,
                                                        thoughts_schema=ThoughtsSchema,
                                                        )
        @schema_tool
        async def recruit_agent_local(agent_name : str = Field(description="The name of the agent to recruit"),
                                agent_instructions : str = Field(description = "The role instructions to give to the agent. This is a general description of the agent's role"),
                                query : str = Field(description = "The specific task to give to the agent")):
            """Recruit an agent to perform a specific task. Give the agent a name, instructions, a query, and the tools to use."""
            tools = tool_response.tools
            agent_instructions = HIRED_AGENT_INSTRUCTIONS.format(manager_instructions = agent_instructions)
            query = query + "\n\n" + "Tool usage yamls are located at the following paths:\n" + "\n".join([f"{t.name} : {t.yaml_path}" for t in tools])

            agent = Role(name=agent_name,
                            instructions = agent_instructions,
                            constraints=None,
                            register_default_events=True,
                            )

            response, metadata = await agent.acall(query,
                                        tools = [get_function(t) for t in tools] + [get_tool_usage],
                                        return_metadata=True,
                                        max_loop_count = 10,
                                        thoughts_schema=ThoughtsSchema,
                                        output_schema = IntermediateTaskResponse,
                                        )
            return response, metadata
        
        local_agent_query = f"""Recruit an agent to perform the next step in the overall user task. Keep your instructions to the local agent specific to the next step.

The original user task is the following : `{response_package.original_query}`

The current chain of agent responses (ordered sequentially) is the following : `{response_package.agent_response_chain}`

The previous agent response was the following : `{response_package.last_response}`

The tools that you found to be useful for this next step are the following : `{tool_response.tools}`
"""
        response, metadata = await manager.acall(local_agent_query,
                                    [recruit_agent_local],
                                    return_metadata=True,
                                    max_loop_count = 10,
                                    thoughts_schema=ThoughtsSchema,
                                    )
        task_responses.append(response)
        check_query = f"""
The original user task was the following : `{response_package.original_query}`

The last recruited agent gave the following response : `{response}`

Is the entire user task complete or do you need to recruit another agent?
"""
        response_check = await manager.aask(check_query, ResponsePackage, use_tool_calls=False)
        if response_check.response == IsTaskComplete.user_task_complete:
            break
        # query = f"""The original user task is the following:\n{original_query}\n\nThe current response chain is the following:\n{task_responses}\n\nThe user task is not complete. Continue recruiting agents to complete the task, search for tools relevant to the last response before recruiting another agent."""
        response_package = ResponsePackage(response = response_check.response,
                                             last_response = response,
                                             agent_response_chain = task_responses,
                                             original_query = original_query)
        current_loops += 1
        # query = response
    # response, metadata = await manager.acall(query,
    #                             #    [ask_pdf_paper, search_web],
    #                                 tools,
    #                                return_metadata=True,
    #                                max_loop_count = 10,
    #                                thoughts_schema=ThoughtsSchema,
    #                                )
    
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






