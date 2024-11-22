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

from test_extensions.tools.agents import ThoughtsSchema
from test_extensions.serialize import dump_metadata_json
from test_extensions.visualize_reasoning import visualize_reasoning_chain
from test_extensions.tools.tool_explorer import create_tool_db, list_schema_tools

import argparse

def initialize_tools(tool_dir, db_path):
    global AGENT_TOOLS, tool_db, agents, toolsets
    AGENT_TOOLS = asyncio.run(list_schema_tools(tool_dir))
    toolsets = {}
    for tool_name, tool in AGENT_TOOLS.items():
        if tool['posix_path'] not in toolsets:
            toolsets[tool['posix_path']] = tool['toolset_usage']
    tool_db = asyncio.run(create_tool_db(tool_dir=tool_dir,
                                save_path = db_path,))
    agents = {}

@schema_tool
async def get_tool_usage(tool_name : str = Field(description = "The name of the tool in the tool database")) -> str:
    """Gets more detailed tool usage on a specific tool in the tool database"""
    tu = AGENT_TOOLS[tool_name]['usage']
    u = f"""This tool's is part of the toolset `{AGENT_TOOLS[tool_name]['posix_path']}`\n\nThis specific tool's usage documentation is the following : `{tu}`"""
    return u

async def main(project_task : str):
    manager = Role(
        name="manager",
        instructions = "You are the project manager. Your job is to complete the task. You must ALWAYS use `get_tool_usage` before calling a tool",
        constraints=None,
        register_default_events=True,
    )

    prompt = f"Find PubMed Central papers from the last two years related to the following user task:`{project_task}`"
    tool_names = ["pmc_search"]
    project_response, project_metadata = await manager.acall(prompt, [AGENT_TOOLS[x]['func_ref'] for x in tool_names] + [get_tool_usage], return_metadata=True)
    # project_response, project_metadata = await manager.acall(project_task, [hire_agent, use_hired_agent, check_hired_agents], return_metadata=True)
    with open(f'project_metadata_initial.txt', 'w') as f:
        print(project_metadata, file = f)
    
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
    print(AGENT_TOOLS)
    asyncio.run(main(project_task = args.query))
    # loop = asyncio.get_event_loop()
    # loop.create_task(main())
    # loop.run_forever()

