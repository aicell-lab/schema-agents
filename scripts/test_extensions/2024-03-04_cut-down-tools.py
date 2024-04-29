import os
from enum import Enum
import inspect
from schema_agents.provider.openai_api import retry
import asyncio
from paperqa import Docs
from typing import List, Optional, Union, Type, Any, get_type_hints, Tuple, Literal
from pydantic import BaseModel, Field, validator, create_model
from bioimageio_chatbot.utils import ChatbotExtension
import asyncio
from schema_agents.role import Message
from schema_agents import schema_tool, Role
import json
from .web_search_extension_with_decorator.llm_web_search import search_duckduckgo
from .web_search_extension_with_decorator.langchain_websearch import LangchainCompressor

langchain_compressor = None
@schema_tool
async def search_web(query: str):
    """Search the web for information using duckduckgo."""
    global langchain_compressor
    langchain_compressor = langchain_compressor or LangchainCompressor(device="cpu")
    content = await search_duckduckgo(query, langchain_compressor, max_results=5, similarity_threshold=0.5, instant_answers=True, chunk_size=500, num_results_to_process=5)
    return content

class LocationType(str, Enum):
    file = "file"
    url = "url"
        
@schema_tool
async def ask_pdf_paper(location : str = Field(description="The location of the paper, either a file path or a url"),
                        location_type : LocationType = Field(description="The type of file location it MUST be either 'file' or 'url'"),
                        question :  str = Field(description="The question to ask about the paper")) -> str:
    """Query a paper for information"""
    docs = Docs()
    if location_type == LocationType.file:
        await docs.aadd(location)
        # docs.add(location)
    elif location_type == LocationType.url:
        await docs.add_url(location)
        # docs.add_url(location)
    else:
        raise ValueError(f"Invalid location type: {location_type}")
    complete_answer = await docs.aquery(question)
    simple_answer = complete_answer.answer
    answer = simple_answer
    with open(os.devnull, 'w') as devnull:
        print(answer, file=devnull)
    return answer

class ThoughtsSchema(BaseModel):
    """Details about the thoughts"""
    reasoning: str = Field(..., description="reasoning and constructive self-criticism; make it short and concise in less than 20 words")

async def main():
    manager = Role(
        name="manager",
        instructions = "You are the manager. Your job is to complete the user's task completely. If it fails, revise your plan and keep trying until it's done",
        constraints=None,
        register_default_events=True,
    )

    response, metadata = await manager.acall("Figure out how to download the data associated with the paper located at /Users/gkreder/gdrive/exponential-chain/GSE254364/405.pdf, open the data and give me a complete summary of the files, their formats, and their contents",
                                   [ask_pdf_paper, search_web],
                                   return_metadata=True,
                                   max_loop_count = 10,
                                   thoughts_schema=ThoughtsSchema
                                   )
    print(response)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()






