import os
from enum import Enum
from schema_agents.provider.openai_api import retry
import asyncio
from typing import List, Optional, Union, Type, Any, get_type_hints, Tuple, Literal
import asyncio
from schema_agents.role import Message
from schema_agents import schema_tool, Role
from pydantic import BaseModel, Field
import json
from langchain.schema.document import Document

from .tools.agents import ThoughtsSchema
from .serialize import dump_metadata_json
from .visualize_reasoning import visualize_reasoning_chain
from .tools.tool_explorer import create_tool_db, search_tools, fixed_db_tool_search, list_schema_tools

import importlib.util
import sys
import yaml


tool_dir = "/Users/gkreder/schema-agents/scripts/test_extensions/tools"
AGENT_TOOLS = asyncio.run(list_schema_tools(tool_dir))
db_path = "tool_index"
tool_db = asyncio.run(create_tool_db(tool_dir=tool_dir,
                            save_path = db_path,))
agents = {}
search_tools = fixed_db_tool_search(fixed_db_path = db_path)

@schema_tool
async def hire_agent(agent_name : str = Field(description = "A name for the agent"),
                    agent_description : str = Field(description = "The description of the agent to hire phrased in the 'you' tense. E.g. `You are an autonomous software agent specialized in X`")) -> str:
    """Hire an agent with a certain specialization for completing sub-tasks"""

    agent_description = agent_description + "\n\n" + """You are a localized agent so DO NOT comment on the capabilities of the system or other agents. DO NOT make any suggestions or recommendations. This is CRUCIAL. You must ONLY report what you did and the results."""

    agent = Role(name = agent_name,
                    instructions = agent_description,
                    constraints=None,
                    register_default_events=True,
                )

    agents[agent_name] = agent
    return f"Agent {agent_name} hired successfully"

class UsefulTools(BaseModel):
    """A list of tools that are useful for the task."""
    tools : List[str] = Field(description = "The names of the tools to use for the task")

class AgentsSteps(BaseModel):
    """The steps that you, the hired agent, used to complete the task given to you"""
    steps : List[str] = Field(description = "All the steps that you took including the tools you used and the reasoning behind each step")

@schema_tool
async def use_hired_agent(agent_name : str = Field(description = "The name of the agent to use"),
                        agent_task : str = Field(description = "The specific task to give to the agent")) -> str:
    """Use a hired agent to complete a specific task"""

    agent = agents[agent_name]
    
    # Find the tools that might be interesting to use for this task
    tool_query = f"""You have been given the following task : `{agent_task}`

Reason about what this task might involve and use the `search_tools` tool to find tools that will be useful for this task.
    """
    tool_response, tool_metadata = await agent.acall(tool_query,
                                        tools = [search_tools],
                                        output_schema = UsefulTools,
                                        return_metadata=True,
                                        thoughts_schema = ThoughtsSchema,
                                        max_loop_count = 10,
                                        )
    found_tool_names = tool_response.tools

    with open(f'tool_steps_{agent_name}.txt', 'w') as f:
        print(tool_metadata, file = f)

    action_response, action_metadata = await agent.acall(agent_task,
                                                         tools = [AGENT_TOOLS[t]['func_ref'] for t in found_tool_names],
                                                            return_metadata=True,
                                                            thoughts_schema=ThoughtsSchema,
                                                            max_loop_count = 10,
                                                            )
    
    with open(f'action_steps_{agent_name}.txt', 'w') as f:
        print(action_metadata, file = f)

    return action_response, action_metadata

class CheckCompletion(BaseModel):
    """A check of whether or not the overall project is complete."""
    project_complete : bool = Field(description = "True if the project is complete, False otherwise")
    summary : str = Field(description = "A summary of the completion status of the project")

async def check_completion(project_goal : str, current_response : str, manager : Role):
    
    check_query = f"""The overall project goal is the following : `{project_goal}`
    
The current state of the project is the following : `{current_response}`

Has the current state of the project COMPLETELY satified the project goal? If yes, then summarize why the project is complete. If no, give a complete description of how the current state falls short and what needs to be done to finish the project.
    """
    completion_res = await manager.aask(check_query, CheckCompletion)
    
    return completion_res

class ProjectState(BaseModel):
    """The current state of the project and the ultimate project goal. Use this to make and execute a plan to complete the project"""
    project_goal : str = Field(description = "The ultimate project goal")
    current_state : str = Field(description = "A summary of the the current state of the project")

async def main():
    manager = Role(
        name="manager",
        instructions = "You are the project manager. Your job is to take the user's input and break it down into sub-tasks for the agents to complete.",
        constraints=None,
        register_default_events=True,
    )
    project_task = "Download and unzip the PubMed Central article with ID 1790863"
    project_response, project_metadata = await manager.acall(project_task, [hire_agent, use_hired_agent], return_metadata=True)
    completion_res = await check_completion(project_goal = project_task, current_response = project_response, manager = manager)

    manager_iterations = 0
    max_manager_iterations = 5
    
    while not completion_res.project_complete and manager_iterations < max_manager_iterations:
        project_state = ProjectState(project_goal = project_task, current_state = completion_res.summary)
        project_response, project_metadata = await manager.acall(project_state, [hire_agent, use_hired_agent], return_metadata=True)
        completion_res = await check_completion(project_goal = project_task, current_response = project_response, manager = manager)
        manager_iterations += 1


    with open('metadata_complete.txt', 'w') as f:
        print(project_metadata, file = f)
    metadata_json_fname = "metadata.json"
    dump_metadata_json(project_metadata, metadata_json_fname)
    with open(metadata_json_fname) as f:
        metadata_json = json.load(f)
    visualize_reasoning_chain(metadata_json, file_path_gv='reasoning_chain_visualization_auto.gv', view = True)
    print(project_response)


if __name__ == "__main__":
    asyncio.run(main())
    # loop = asyncio.get_event_loop()
    # loop.create_task(main())
    # loop.run_forever()

# agent_tools = []
# from .tools.tool_explorer import create_tool_db, search_tools, fixed_db_tool_search
# db_path = "tool_index"
# tool_db = asyncio.run(create_tool_db(tool_dir="/Users/gkreder/schema-agents/scripts/test_extensions/tools",
#                             save_path = db_path,))
# search_tools = fixed_db_tool_search(fixed_db_path = db_path)




# MANAGER_INSTRUCTIONS = """
# You are the general task manager. Your job is to complete the user's task completely by searching tools, making a plan, then hiring agents.
# """

# HIRED_AGENT_INSTRUCTIONS = """
# You are an agent hired for a specific task. Your job is to complete the task given to you by the manager. Your workflow should follow this logic:
# - Read the instructions given to you by the manager.
# - Check the tool usage for each tool you are given by the manager by invoking the `get_tool_usage` tool on each one of them.
# - Make a plan using `StartNewPlan` to use the provided tools to complete your task. This plan might be multi-step.
# - Execute the plan you created by doing the following:
#     - Carry out the plan's tasks
#     - Check if you have completed your assigned task from the manager.
# - Keep modifying and executing your plan until the entire manager's task is complete.

# The manager's instructions are as follows:\n{manager_instructions}
# """

# agents = {}
# agent_tools = []


# class RecruitTool(BaseModel):
#     """A tool that can be used for the task."""
#     name : str = Field(description = "The name of the tool")
#     relevant_info: str = Field(description = "The relevant information about the tool")
#     posix_path : str = Field(description = "The posix_path of the tool")
#     yaml_path : str = Field(description = "The yaml_path of the tool containing the tool's usage information")
#     reasoning : str = Field(description = "The reasoning behind why this tool is useful for the task")

# # class UsefulTools(BaseModel):
# #     """A list of tools that are useful for the task."""
# #     tools : List[RecruitTool] = Field(description = "The tools to use for the task")

# def get_function(t : RecruitTool):
#     module_name = t.posix_path.rsplit('/', 1)[-1][:-3]  # Extracts the base '.py' file name then removes the '.py'
#     # Import the module
#     spec = importlib.util.spec_from_file_location(module_name, t.posix_path)
#     module = importlib.util.module_from_spec(spec)
#     sys.modules[module_name] = module
#     spec.loader.exec_module(module)
#     # Access the function
#     function = getattr(module, t.name)
#     return function

# @schema_tool
# async def get_tool_usage(tool_name : str = Field(description="The name of the tool to get usage information for"),
#                         yaml_path : str = Field(description = "The path to the yaml file containing the tool's usage information")): 
#     """Get the usage information for the tool."""
#     if not os.path.exists(yaml_path):
#         return "No usage information provided for this tool."
#     with open(yaml_path, 'r', encoding = 'utf-8') as yaml_file:
#             usage_strings = yaml.safe_load(yaml_file)
#     s = usage_strings.get(tool_name, {'usage' : "No usage string provided"})['usage']
#     return s
            
# @schema_tool
# async def recruit_agent_local(agent_name : str = Field(description="The name of the agent to recruit"),
#                         agent_instructions : str = Field(description = "The role instructions to give to the agent. This is a general description of the agent's role"),
#                         query : str = Field(description = "The specific task to give to the agent"),):
#     """Recruit an agent to perform a specific task. Give the agent a name, instructions, a query, and the tools to use."""

#     # agent_instructions = agent_instructions + "\n\n" + "Before using any other tool, you MUST use the `get_tool_usage` tool to get usage information about the tool you will use."
#     agent_instructions = HIRED_AGENT_INSTRUCTIONS.format(manager_instructions = agent_instructions)
#     query = query + "\n\n" + "Tool usage yamls are located at the following paths:\n" + "\n".join([f"{t.name} : {t.yaml_path}" for t in tools])

#     # Search tool DB
#     # reason over tools
#     # Equip agent with tools
#     # Tool creation (future)

#     agent = Role(name=agent_name,
#                     instructions = agent_instructions,
#                     constraints=None,
#                     register_default_events=True,
#                     )

#     tq = f"""You are a software agent that has been recruited for a specific task from the manager. The manager's overall instructions are as follows:\n`{agent_instructions}`. Your specific task is the following:\n`{query}`.

# Use the `search_tools` tool to find tools that might be relevant to your specific task at hand.
#     """
#     Role.acall(tq, [search_tools], output_schema = UsefulTools)

    
#     agents[agent_name] = agent
    
#     async def agent_tool(query : str = Field(description = "The specific task to give to the agent")):
#         response, metadata = await agent.acall(query,
#                                 tools = [get_function(t) for t in tools] + [get_tool_usage],
#                                 return_metadata=True,
#                                 max_loop_count = 10,
#                                 thoughts_schema=ThoughtsSchema,
#                                 output_schema = IntermediateTaskResponse,
#                                 )
#         return response, metadata
#     agent_tool.__name__ = agent_name
#     agent_tool.__doc__ = agent_instructions
#     agent_tools.append(schema_tool(agent_tool))
#     return f"Agent {agent_name} recruited successfully"

# async def main():

#     # prompt = s + "\n\n" + "User question : What is the official gene symbol of LMP10?"
#     manager_instructions =  MANAGER_INSTRUCTIONS
#     manager = Role(
#         name="manager",
#         instructions = manager_instructions,
#         constraints=None,
#         register_default_events=True,
#     )

#     from .tools.tool_explorer import create_tool_db, search_tools, fixed_db_tool_search
#     db_path = "tool_index"
#     tool_db = await create_tool_db(tool_dir="/Users/gkreder/schema-agents/scripts/test_extensions/tools",
#                              save_path = db_path,)
#     search_tools = fixed_db_tool_search(fixed_db_path = db_path)

#     # tools = [search_tools, recruit_agent_local]

#     # query = """Search for tools in the tool database that can help with the task of finding the official gene symbol of LMP10. Then pass these tools to a recruited agent to complete the task."""
#     # query = """Search for tools in the tool database that can help with the task of figuring out the main finding of the PDF located at /Users/gkreder/schema-agents/PMC10897392/41746_2024_Article_1038.pdf. Then pass these tools to a recruited agent to complete the task."""
#     # query = """Search for tools in the tool database that can help with the task of downloading and unzipping the PubMed Central article with ID 1790863. Then pass these tools to a recruited agent to complete the task."""
#     query = """Download and unzip the PubMed Central article with ID 1790863"""

#     task_responses = []
#     class IsTaskComplete(str, Enum):
#         master_task_complete = "Master task complete"
#         not_complete = "The user task is not complete."
    
#     class ResponsePackage(BaseModel):
#         """The response to the check if the master task is completed"""
#         response : IsTaskComplete = Field(description = "A check of whether or not the master task is complete.")
#         last_response : str = Field(description = "The response from the most recently recruited agent's sub-task")
#         agent_response_chain : List[str] = Field(description = "The chain of responses from all the agents that have been recruited so far to solve sub-tasks (ordered sequentially where the first item is the first agent's response and the last item is the most recent agent's response)")
#         master_task : str = Field(description = "The high-level master task that the agents are trying to complete")
    
#     class UsefulTools(BaseModel):
#         """A list of tools that are useful for the task."""
#         tools : List[RecruitTool] = Field(description = "The tools to use for the task")
    
#     class CurrentTaskSummary(BaseModel):
#         """A summary of the current state of master task completion including what has been tried so far and what the next steps need to be."""
#         master_task : str = Field(description = "The high-level master task that the agents are trying to complete")
#         summary_of_agent_work : str = Field(description = "A summary of the work that the agents have done so far in bullet point form")
#         suggested_next_step : str = Field(description = "A suggestion for the next step to take in completing the user task.")
#         suggested_tools : List[RecruitTool] = Field(description = "A list of tools that are suggested for the next step in completing the user task.")
#         overall_plan : str = Field(description = "A high-level plan for how to complete the user task. This should be a summary of the steps that have been taken so far and the steps that need to be taken in the future.")
    
#     # Wei : loop structure starts here
#     max_manager_loops = 5
#     current_loops = 0
#     original_query = query
#     response_package = ResponsePackage(response = IsTaskComplete.not_complete, 
#                                        last_response = "No agents have been recruited yet. Come up with a first step to complete the master task.", 
#                                        agent_response_chain = [],
#                                        master_task = original_query)
    
#     while True and current_loops < max_manager_loops:

#         # if current_loops == 0:
#             # tool_prompt = f"Come up with a first step to complete the master task. No agents have been recruited to solve sub-tasks yet. The master task original given was the following: `{original_query}`"
#         # else:
#             # tool_prompt = f"""Read the current chain of agent responses and compare them to the original master task. Suggest a specific next step and use the `search_tools` tool to find tools that will help with it"""
#         tool_prompt = response_package

#         tool_response, tool_metadata = await manager.acall(tool_prompt,
#                                                         [search_tools],
#                                                         output_schema = CurrentTaskSummary,
#                                                         return_metadata=True,
#                                                         max_loop_count = 10,
#                                                         thoughts_schema=ThoughtsSchema,
#                                                         )
#         @schema_tool
#         async def recruit_agent_local(agent_name : str = Field(description="The name of the agent to recruit"),
#                                 agent_instructions : str = Field(description = "The role instructions to give to the agent. This is a general description of the agent's role. MAKE SURE TO TELL THE AGENT TO SIMPLY REPORT BACK ON ITS SUBTASK AND ONLY ITS SUBTASK"),
#                                 query : str = Field(description = "The specific task to give to the agent")):
#             """Recruit an agent to perform the next specific task in the overall task completion chain."""
#             tools = tool_response.suggested_tools
#             agent_instructions = HIRED_AGENT_INSTRUCTIONS.format(manager_instructions = agent_instructions)
#             query = query + "\n\n" + "Tool usage yamls are located at the following paths:\n" + "\n".join([f"{t.name} : {t.yaml_path}" for t in tools])

#             agent = Role(name=agent_name,
#                             instructions = agent_instructions,
#                             constraints=None,
#                             register_default_events=True,
#                             )

#             response, metadata = await agent.acall(query,
#                                         tools = [get_function(t) for t in tools] + [get_tool_usage],
#                                         return_metadata=True,
#                                         max_loop_count = 10,
#                                         thoughts_schema=ThoughtsSchema,
#                                         output_schema = IntermediateTaskResponse,
#                                         )
#             return response, metadata
        
#         local_agent_query = f"""Recruit an agent to perform the next step in the overall master task using the information you have already figured out which is summarized below.
        
# The overall master task is the following : `{tool_response.master_task}`
        
# The next step you have chosen in the overall master task is the following : `{tool_response.suggested_next_step}`
# """
        
#         response, metadata = await manager.acall(local_agent_query,
#                                     [recruit_agent_local],
#                                     return_metadata=True,
#                                     max_loop_count = 10,
#                                     thoughts_schema=ThoughtsSchema,
#                                     )
#         task_responses.append(response)
#         check_query = f"""
# The high-level master task is the following : `{response_package.master_task}`

# The last recruited agent gave the following response : `{response}`

# Is the entire master task complete or do you need to recruit another agent?
# """
#         response_check = await manager.aask(check_query, ResponsePackage, use_tool_calls=False)
#         if response_check.response == IsTaskComplete.master_task_complete:
#             break
#         # query = f"""The original user task is the following:\n{original_query}\n\nThe current response chain is the following:\n{task_responses}\n\nThe user task is not complete. Continue recruiting agents to complete the task, search for tools relevant to the last response before recruiting another agent."""
#         response_package = ResponsePackage(response = response_check.response,
#                                              last_response = response,
#                                              agent_response_chain = task_responses,
#                                              master_task = original_query)
#         current_loops += 1
#         # query = response
#     # response, metadata = await manager.acall(query,
#     #                             #    [ask_pdf_paper, search_web],
#     #                                 tools,
#     #                                return_metadata=True,
#     #                                max_loop_count = 10,
#     #                                thoughts_schema=ThoughtsSchema,
#     #                                )
    
#     with open('metadata_complete.txt', 'w') as f:
#         print(metadata, file = f)
#     metadata_json_fname = "metadata.json"
#     dump_metadata_json(metadata, metadata_json_fname)
#     with open(metadata_json_fname) as f:
#         metadata_json = json.load(f)
#     visualize_reasoning_chain(metadata_json, file_path_gv='reasoning_chain_visualization_auto.gv', view = True)
#     print(response)

# if __name__ == "__main__":
#     loop = asyncio.get_event_loop()
#     loop.create_task(main())
#     loop.run_forever()






