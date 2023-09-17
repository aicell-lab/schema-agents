#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 19:26
@Author  : alexanderwu
@File    : test_design_api.py
"""
import pytest
import json

from schema_agents.action import Action
from metagpt.logs import logger
from pydantic import BaseModel, Field
from metagpt.schema import Message

class APIDesign(BaseModel):
    """API Design.""" 
    api_functions: str = Field(description="The list of functions that the API needs to implement")
    file_name: str  = Field(description="Concise and clear file name, characters only use a combination of all lowercase and underscores")

class CodeFile(BaseModel):
    """Code File.""" 
    file_name: str = Field(description="Concise and clear file name, characters only use a combination of all lowercase and underscores")
    file_content: str = Field(description="The content of the file, comply with PEP8 standards")
    

@pytest.mark.asyncio
async def test_schema_action():
    sa = Action(
        name="Coder",
        desc="Write code in python",
        system_prompt="You are a helpful assistant."
    )
    ret = await sa.run("Hello, which programming language do you use? reply in 1 word")
    logger.debug(ret)
    assert "python" in ret.content.lower()

    sa = Action(
        name="Coder",
        desc="Write code in python",
        system_prompt="You are a helpful assistant.",
        input_schemas=[APIDesign],
        output_schemas=[CodeFile],
    )
    design = APIDesign(api_functions="add(a, b): add two numbers\nmultiply(a): multiple a number by 16;", file_name="calc.py")
    ret = await sa.run([
        Message(content=design.json(), instruct_content=design,
                          role="BOSS", cause_by="someone")
    ])
    logger.debug(ret)
    assert ret.instruct_content.file_name == "calc.py"
    assert "def add(a, b):" in ret.instruct_content.file_content
    assert "def multiply(a):" in ret.instruct_content.file_content
    assert "def add(a, b):" in ret.content
    assert json.loads(ret.content)["file_name"] == "calc.py"
    
    
