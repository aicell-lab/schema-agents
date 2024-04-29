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

# @schema_tool
# async def ask_pdf_paper_info() -> str:
#     """The `ask_pdf_paper` tool lets you query a PDF paper for information with natural language queries.
    
#     Here is an example:

#     Question: What are the paper's significant findings?
#     [(file_location="PMC10912389/main.pdf", question="What are the paper's significant findings?")]->["Question: What are the paper's significant findings?

#         The paper Chen2024 presents significant findings on heart failure treatment and management. It emphasizes the importance of maintaining autophagy balance and mitochondrial stability, highlighting the role of mitochondrial dynamics and the AMPK/PPAR-α pathway in myocardial stability (Chen2024 pages 5-6). The therapeutic potential of natural compounds such as chlorogenic acid, rutin, quercetin, liguzinediol, tetramethylpyrazine, and berberine in heart-related conditions is discussed, with mechanisms involving modulation of immunity and inhibition of ferroptosis, apoptosis, and various signaling pathways (Chen2024 pages 29-29). Additionally, the paper covers the management of HFpEF and HFmrEF, suggesting interventions targeting the NO-sGC-cGMP pathway and addressing comorbidities (Chen2024 pages 3-4). The efficacy of Shenfu Qiangxin pill combined with sacubitril valsartan sodium tablet and other traditional Chinese medicine products in improving heart function is also reported (Chen2024 pages 22-22).

#         References

#         1. (Chen2024 pages 5-6): Chen, Xing-Juan, et al. "The Recent Advance and Prospect of Natural Source Compounds for the Treatment of Heart Failure." Heliyon, vol. 10, 2024, e27110. Elsevier Ltd, doi:10.1016/j.heliyon.2024.e27110. Accessed 25 Feb. 2024.

#         2. (Chen2024 pages 29-29): Chen, Xing-Juan, et al. "The Recent Advance and Prospect of Natural Source Compounds for the Treatment of Heart Failure." Heliyon, vol. 10, 2024, e27110. Elsevier Ltd, doi:10.1016/j.heliyon.2024.e27110. Accessed 25 Feb. 2024.

#         3. (Chen2024 pages 3-4): Chen, Xing-Juan, et al. "The Recent Advance and Prospect of Natural Source Compounds for the Treatment of Heart Failure." Heliyon, vol. 10, 2024, e27110. Elsevier Ltd, doi:10.1016/j.heliyon.2024.e27110. Accessed 25 Feb. 2024.

#         4. (Chen2024 pages 22-22): Chen, Xing-Juan, et al. "The Recent Advance and Prospect of Natural Source Compounds for the Treatment of Heart Failure." Heliyon, vol. 10, 2024, e27110. Elsevier Ltd, doi:10.1016/j.heliyon.2024.e27110. Accessed 25 Feb. 2024."]

#     Answer: The paper Chen2024 presents significant findings on heart failure treatment and management. It emphasizes the importance of maintaining autophagy balance and mitochondrial stability, highlighting the role of mitochondrial dynamics and the AMPK/PPAR-α pathway in myocardial stability (Chen2024 pages 5-6). The therapeutic potential of natural compounds such as chlorogenic acid, rutin, quercetin, liguzinediol, tetramethylpyrazine, and berberine in heart-related conditions is discussed, with mechanisms involving modulation of immunity and inhibition of ferroptosis, apoptosis, and various signaling pathways (Chen2024 pages 29-29). Additionally, the paper covers the management of HFpEF and HFmrEF, suggesting interventions targeting the NO-sGC-cGMP pathway and addressing comorbidities (Chen2024 pages 3-4). The efficacy of Shenfu Qiangxin pill combined with sacubitril valsartan sodium tablet and other traditional Chinese medicine products in improving heart function is also reported (Chen2024 pages 22-22).
#     """
#     return "Query a PDF paper for information"


async def ask_pdf_paper_info() -> str:
    """Get information about the `ask_pdf_paper` tool"""
    return """The `ask_pdf_paper` tool lets you query a PDF paper for information with natural language queries."""

@schema_tool
async def ask_pdf_paper(file_location : str = Field(description="The location of the paper's PDF file"),
                        question :  str = Field(description="The question to ask about the paper")) -> str:
    """Query a PDF paper for information"""
    docs = Docs()
    await docs.aadd(file_location)
    answer = await docs.aquery(question)
    formatted_answer = answer.formatted_answer
    return formatted_answer