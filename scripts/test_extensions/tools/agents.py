import os
import xml.etree.ElementTree as ET
from enum import Enum
import inspect
from schema_agents.provider.openai_api import retry
import asyncio
from typing import List, Optional, Union, Type, Any, get_type_hints, Tuple, Literal, Callable
from pydantic import BaseModel, Field, validator, create_model
import asyncio
from schema_agents.role import Message
from schema_agents import schema_tool, Role

class ThoughtsSchema(BaseModel):
    """Details about the thoughts"""
    reasoning: str = Field(..., description="reasoning and constructive self-criticism; make it short and concise in less than 20 words")


async def recruit_agent(agent_tools : List[Callable] = Field(description = "The tools to equip this specific agent with. It should be a subset of all the available tools and only the ones that would be useful for this agent's task"),
                        agent_name : str = Field(description="The name of the agent to recruit"),
                        agent_instructions : str = Field(description = "The role instructions to give to the agent. This is a general description of the agent's role"),
                        query : str = Field(description = "The specific task to give to the agent"),):
    """Recruit an agent to perform a specific task. Give the agent a name, instructions, a query, and the tools to use"""
    agent = Role(name=agent_name,
                    instructions = agent_instructions,
                    constraints=None,
                    register_default_events=True,
                    )
    response, metadata = await agent.acall(query,
                                agent_tools,
                                return_metadata=True,
                                max_loop_count = 10,
                                thoughts_schema=ThoughtsSchema,
                                )
    return response, metadata