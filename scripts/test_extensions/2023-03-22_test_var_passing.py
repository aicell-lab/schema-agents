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
import yaml


async def main():
    manager_instructions =  "You are a helpful assistant, answer the query"
    manager = Role(
        name="manager",
        instructions = manager_instructions,
        constraints=None,
        register_default_events=True,
    )

    # query = """Search for tools in the tool database that can help with the task of finding the official gene symbol of LMP10. Then pass these tools to a recruited agent to complete the task."""
    query = """What are the contents of the variable `test_var`?"""
    test_var = {'a' : 'hello', 'b' : 1}
    # m = Message(content = query, role = 'user', data = {'test_var' : test_var})
    m = {'content' : query, 'role' : 'user'}
    tool_calls, metadata = await manager.aask(m, return_metadata=True, use_tool_calls=True)
    
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






