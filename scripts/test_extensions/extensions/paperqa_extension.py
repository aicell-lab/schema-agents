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

async def run_extension(paper: Paper) -> str:

    docs = Docs()
    if paper.location_type == LocationType.file:
        await docs.aadd(paper.location)
        # docs.add(paper.location)
    elif paper.location_type == LocationType.url:
        await docs.add_url(paper.location)
        # docs.add_url(paper.location)
    else:
        raise ValueError(f"Invalid location type: {paper.location_type}")
    
    # questions = [
    #     "What are the key topics of the paper?",
    #     "Does the paper involve informatic, computational, or data analysis that involves coding?",
    #     "What are the key findings of the paper?",
    #     "What methods were used in the paper?"
    # ]
    # answers = [f"Question {i} :\n{q}\n\nAnswer:\n{await aux_query(q, docs)}" for i, q in enumerate(questions)]
    # with open(os.devnull, 'w') as devnull:
            # print(answers, file=devnull)
    # return "\n------------------------\n".join(answers)
    answer = await aux_query(paper.question, docs)
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

