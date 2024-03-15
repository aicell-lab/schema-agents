import os
import xml.etree.ElementTree as ET
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
import time
import urllib.request
from paperqa import Docs

@schema_tool
async def ask_pdf_paper(file_location : str = Field(description="The location of the paper's PDF file"),
                        question :  str = Field(description="The question to ask about the paper")) -> str:
    """Query a paper for information"""
    docs = Docs()
    await docs.aadd(file_location)
    answer = await docs.aquery(question)
    formatted_answer = answer.formatted_answer
    return formatted_answer