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
from .tools.tool_explorer import create_tool_db, fixed_db_tool_search, list_schema_tools

import argparse

def initialize_tools(tool_dir, db_path):
    global AGENT_TOOLS, tool_db, agents
    AGENT_TOOLS = asyncio.run(list_schema_tools(tool_dir))
    tool_db = asyncio.run(create_tool_db(tool_dir=tool_dir,
                                save_path = db_path,))
    agents = {}
    # search_tools = fixed_db_tool_search(fixed_db_path = db_path)

@schema_tool
async def search_tools(query : str = Field(description="The query to search against the tool database")) -> str:
    """Search for tool to solve tasks given a tool database"""
    tool_chunk ="""({i}) Tool name : {name}\n- Relevant tool description : {docstring}\n- posix path : {posix_path}\n- yaml path : {yaml_path}"""
    results_string = """I found {l} tools that might be helpful to give a hired agent to solve the task:\n{tool_chunks}"""
    results = tool_db.similarity_search(query)
    tool_chunks = '\n'.join([tool_chunk.format(i = i+1, name = x.metadata['name'], docstring = x.page_content, posix_path = x.metadata['posix_path'], yaml_path = x.metadata['yaml_path']) for i, x in enumerate(results)])
    rs = results_string.format(l = len(results), tool_chunks = tool_chunks)
    return rs

@schema_tool
async def get_tool_usage(tool_name : str = Field(description = "The name of the tool in the tool database")) -> str:
    """Gets more detailed tool usage on a specific tool in the tool database"""
    u = AGENT_TOOLS[tool_name]['usage']
    return u


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
    """A list of tools that are useful for the task or fixing the current state of the project"""
    tools : List[str] = Field(description = "The names of the tools to use for the task")

@schema_tool
async def use_hired_agent(agent_name : str = Field(description = "The name of the agent to use"),
                        agent_task : str = Field(description = "The specific task to give to the agent"),
                        suggested_tools : Optional[str] = Field(description = "Tools suggested by the manager that might be helpful")) -> str:
    """Use a hired agent to complete a specific task"""

    agent = agents[agent_name]
    
    # Find the tools that might be interesting to use for this task
    tool_query = f"""You have been given the following task : `{agent_task}`

Reason about what this task might involve and use the `search_tools` tool to find tools that will be useful for this task.

The manager that hired you has suggested that the following tools might be helpful: `{suggested_tools}`
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
    
    action_query = f"""You have been given the following task : `{agent_task}`

Use the tools that you have already found to complete the task. If you unsure about how to use the tools, use the `get_tool_usage` tool to get information about any other tool.
"""

    action_response, action_metadata = await agent.acall(agent_task,
                                                         tools = [AGENT_TOOLS[t]['func_ref'] for t in found_tool_names] + [get_tool_usage],
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

Has the current state of the project COMPLETELY satified the project goal? 
If yes, then summarize why the project is complete.
If no, give a complete description of how the current state falls short and what needs to be done to finish the project. In this description throw away any comments about the system's capabilities or the capabilities of other agents. Only focus on the project completion status.
    """
    completion_res, _ = await manager.acall(check_query, [search_tools], output_schema = CheckCompletion, return_metadata=True)
    
    return completion_res

async def get_feedback(query : str, troubleshooter : Role) -> str:
    """Check the troubleshooting knowledge base for helpful tips on completing the project"""
    feedback = await troubleshooter.aask(query)
    return query

class SuggestedTools(BaseModel):
    """A list of tools that are suggested for completing the project"""
    tools : List[str] = Field(description = "The names of the tools to use for the task")

class ManagerSuggestions(BaseModel):
    """The manager's suggestions for the next steps in the project completion process. DO NOT carry out any actual steps, only make suggestions"""
    feedback : str = Field(description = "The manager's feedback on the current state of the project and how to proceed")
    suggested_tools : SuggestedTools = Field(description = "A list of tools from the tool database that are suggested for completing the project at this point in time")

class ProjectState(BaseModel):
    """The current state of the project and the ultimate project goal. Use this to make and execute a plan to complete the project"""
    project_goal : str = Field(description = "The ultimate project goal")
    current_state : str = Field(description = "A summary of the the current state of the project")
    manager_suggestions : ManagerSuggestions = Field(description = "The manager's suggestions for the next steps in the project completion process")

async def get_manager_suggestions(project_task : str, current_state : str, manager : Role, troubleshooter : Role) -> ManagerSuggestions:
    """Get the manager's suggestions for the next steps in the project completion process"""

    suggestion_query = f"""The overall project goal is to complete the following : `{project_task}`

The current state of the project is the following : `{current_state}`

Based on the current state of the project, please check for any feedback to give the worker agents.
"""

    feedback = await get_feedback(suggestion_query, troubleshooter)
    tool_db_hits = await search_tools(current_state)
    tool_query = f"""Your summary of the current project state involves the following : `{current_state}`

After using searching our database of tools, with these steps, you received the following response: `{tool_db_hits}`

Please use this information to provide a list of tool names that might be helpful for the next step in the project
"""
    suggested_tools = await manager.aask(tool_query, SuggestedTools)
    manager_suggestions = ManagerSuggestions(feedback = feedback,
                                             suggested_tools = suggested_tools)
    return manager_suggestions
    

async def main(project_task : str):
    manager = Role(
        name="manager",
        instructions = "You are the project manager. Your job is to take the user's input and break it down into sub-tasks for the agents to complete.",
        constraints=None,
        register_default_events=True,
    )

    troubleshooter = Role(
        name="troubleshooter",
        instructions = "You are the troubleshooter, an expert in reasoning and critical thinking. Your job is to check all the steps that have been tried so far and to look for any logical issues or alternative approaches that will help the manager with project planning.",
        constraints = None,
        register_default_events=True,
    )


    initial_state = "No agents have been recruited yet. The project has not been started yet."    
    # manager_suggestions, _ = await manager.acall(first_prompt, [search_tools], output_schema = ManagerSuggestions, return_metadata=True)
    # manager_suggestions = await get_manager_suggestions(project_task, initial_state, manager, troubleshooter)
    # manager_suggestions = ManagerSuggestions(feedback = "No feedback yet", suggested_tools = SuggestedTools(tools = []))
    # project_state = ProjectState(project_goal = project_task,
    #                              current_state = initial_state,
    #                              manager_suggestions = manager_suggestions)
    # project_response, project_metadata = await manager.acall(project_state, [hire_agent, use_hired_agent], return_metadata=True)
    project_response, project_metadata = await manager.acall(project_task, [hire_agent, use_hired_agent], return_metadata=True)
    completion_res = await check_completion(project_goal = project_task, current_response = project_response, manager = manager)

    manager_iterations = 0
    max_manager_iterations = 5
    
    while not completion_res.project_complete and manager_iterations < max_manager_iterations:
        # manager_suggestions, _ = await manager.acall(project_task, [search_tools], output_schema = ManagerSuggestions, return_metadata=True)
        manager_suggestions = await get_manager_suggestions(project_task, completion_res.summary, manager, troubleshooter)
        project_state = ProjectState(project_goal = project_task, current_state = completion_res.summary, manager_suggestions = manager_suggestions)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run the manager team project")
    parser.add_argument("--tool_dir", type=str, help="The directory where the tools are stored", required=True)
    parser.add_argument("--db_path", type=str, help="The path to the tool database", required=True)
    parser.add_argument("--query", type=str, help="The query to run", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    tool_dir = "/Users/gkreder/schema-agents/scripts/test_extensions/tools"
    db_path = "tool_index"
    args = parse_args()
    initialize_tools(tool_dir, db_path)
    asyncio.run(main(project_task = args.query))
    # loop = asyncio.get_event_loop()
    # loop.create_task(main())
    # loop.run_forever()

