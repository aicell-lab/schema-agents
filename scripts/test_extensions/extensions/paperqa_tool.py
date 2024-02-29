import os
import sys
import time
from enum import Enum
import asyncio
import requests
from bs4 import BeautifulSoup
import pandas as pd
from paperqa import Docs
from typing import List, Optional
from pydantic import BaseModel, Field
from bioimageio_chatbot.utils import ChatbotExtension
    
class LocationType(str, Enum):
    file = "file"
    url = "url"
        
class Paper(BaseModel):
    """A scientific journal or conference article to study"""
    location : str = Field(description="The location of the paper, either a file path or a url")
    location_type : LocationType = Field(description="The type of file location it MUST be either 'file' or 'url'")
    question :  Optional[str] = Field(None, description="The question to ask about the paper")

async def aux_query(question, docs):
    answer = await docs.aquery(question)
    return answer.formatted_answer    

async def ask_pdf_paper(location : str = Field(description="The location of the paper, either a file path or a url"),
                        location_type : LocationType = Field(description="The type of file location it MUST be either 'file' or 'url'"),
                        question :  str = Field(description="The question to ask about the paper")) -> str:
    """Query a paper for information"""
    docs = Docs()
    if location_type == LocationType.file:
        await docs.aadd(location)
    elif location_type == LocationType.url:
        await docs.add_url(location)
    else:
        raise ValueError(f"Invalid location type: {location_type}")
    answer = await aux_query(question, docs)
    with open(os.devnull, 'w') as devnull:
        print(answer, file=devnull)
    return answer

def get_extensions():
    return [
        ChatbotExtension(
            name="PaperQA",
            description="Use paper-qa to retrieve information about a paper.",
            execute=run_extension,
        )
    ]

