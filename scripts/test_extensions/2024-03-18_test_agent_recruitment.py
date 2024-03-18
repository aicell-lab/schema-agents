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

from .tools.NCBI import get_geo_api_info, get_genomic_api_info, ncbi_api_call, get_pubmed_api_info, get_pubmed_central_oa
from .tools.fileIO import write_to_file, read_file, ftp_download, unzip_tar_gz, list_files_in_dir
from .tools.llm_web_search import search_pubmed_paper
from .tools.paperqa import ask_pdf_paper
from .tools.agents import ThoughtsSchema, recruit_agent
from .serialize import dump_metadata_json


async def main():

    # prompt = s + "\n\n" + "User question : What is the official gene symbol of LMP10?"

    manager = Role(
        name="manager",
        instructions = "You are the manager. Your job is to complete the user's task completely. If it fails, revise your plan and keep trying until it's done",
        constraints=None,
        register_default_events=True,
    )

    # query = "Figure out how to download the data associated with the paper located at /Users/gkreder/gdrive/exponential-chain/GSE254364/405.pdf, open the data and give me a complete summary of the files, their formats, and their contents"
    # query = "What is the official gene symbol of LMP10?"
    # query = "Tell me all the proteins associated with muscular dystrophy. Keep using the NCBI Web APIs until you get a final list. Use the write_to_file and read_file tools to store and retrieve intermediate results."
    # query = "Please find a paper associated with muscular dystrophy whose full text is available on PubMed. Then write the content of that article to a local text file."
    # query = "Who are some authors who have written lots of articles recently about synthetic biology?"
    # query = "Give me a non-redundant list of genes associated with 'Malignant breast neoplasm'"
    # query = """Answer the following question. Do thorough research then answer yes/no/maybe and give justification: 'Does repeated hyperbaric exposure to 4 atmosphere absolute cause hearing impairment?' If you are 70 percent certain of an answer (yes/no) give that as your final answer."""
    # query = "What specific methods were used in the paper with PubMed Central ID PMC1790863?"
    # query = "Is the PubMed Central article with ID PMC1790863 open access? If so, download it, unzip it if necessary, and tell me the final path to the PDF"
    # query = "Who are the authors of the paper with PubMed Central ID PMC1790863?"
    # query = """Make a plan and execute it to answer the following question using all research methods necessary: 'Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?'"""
    # query = "Recruit an agent to perform the task of creating a text file and writing the phrase `hello world` to the file. Give the agent a name, instructions, a query, and the tools to use"
    query = "Make a plan and execute it to have agents write a text file with the contents `hello world`. Then have another agent read that file and print the contents to another file with the current and and time appended to the contents of the first file."
    # query = """Make a plan and execute it to answer the following question using all research methods necessary: 'Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?'. Recruit as many agents as necessary to complete the task and modify the team if they perform poorly."""

    # tools = [get_geo_api_info, get_genomic_api_info, ncbi_api_call, write_to_file, read_file, get_pubmed_api_info]
    # tools = [get_pubmed_central_oa, write_to_file, read_file, ftp_download, unzip_tar_gz, search_pubmed_paper, list_files_in_dir, ask_pdf_paper]
    tools_static = [get_pubmed_central_oa, write_to_file, read_file, 
             ftp_download, unzip_tar_gz, search_pubmed_paper, 
             list_files_in_dir, ask_pdf_paper, get_geo_api_info, 
             get_genomic_api_info, ncbi_api_call, get_pubmed_api_info]
    
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
    
    dump_metadata_json(metadata, "metadata.json")
    print(response)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()






