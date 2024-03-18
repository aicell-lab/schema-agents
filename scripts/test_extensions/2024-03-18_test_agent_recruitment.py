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
    # query = "Make a plan and execute it to have agents write a text file whose contents are the current year and time. Then have another agent read that file and print the contents to another file containing a poem about the time in the file contents. Use as many agents as necessary and keep looping until the job is complete"
    # query = """Make a plan and execute it to answer the following question using all research methods necessary: 'Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?'. Recruit as many agents as necessary to complete the task and modify the team if they perform poorly."""
    query = """Make a plan and execute it to answer the following question using all research methods necessary. Make sure to recruit agents to perform sub-tasks and modify the agent team if they perform poorly. You MUST limit your paper search to open access papers: 'List signaling molecules (ligands) that interact with the receptor EGFR?' Do not use an api key for the NCBI api"""
    # query = """Construct an NCBI query url to answer the following question. You are ONLY allowed to use papers from before the year 2002. You MUST limit your paper search to open access papers: 'Is leptin involved in phagocytic NADPH oxidase overactivity in obesity?'"""
    
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






