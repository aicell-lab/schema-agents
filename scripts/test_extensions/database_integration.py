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

from .tools.NCBI import get_geo_api_info, get_genomic_api_info, ncbi_api_call, get_pubmed_api_info
from .tools.fileIO import write_to_file, read_file
tools = [get_geo_api_info, get_genomic_api_info, ncbi_api_call, write_to_file, read_file, get_pubmed_api_info]

class ThoughtsSchema(BaseModel):
    """Details about the thoughts"""
    reasoning: str = Field(..., description="reasoning and constructive self-criticism; make it short and concise in less than 20 words")


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
    query = """Answer the following question. Do thorough research then answer yes/no/maybe and give justification: 'Does repeated hyperbaric exposure to 4 atmosphere absolute cause hearing impairment?' If you are 70 percent certain of an answer (yes/no) give that as your final answer."""

    response, metadata = await manager.acall(query,
                                #    [ask_pdf_paper, search_web],
                                    tools,
                                   return_metadata=True,
                                   max_loop_count = 10,
                                   thoughts_schema=ThoughtsSchema,
                                   )
    # plot_metdata(metadata)
    print(response)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()






