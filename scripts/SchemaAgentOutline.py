# generated by datamodel-codegen:
#   filename:  SchemaAgentOutline.json
#   timestamp: 2023-11-30T08:32:39+00:00

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class SchemaActionOutline(BaseModel):
    """
    A detailed description of a python function containing type annotations, a function signature, and pseudocode for the function body
    """

    content: str = Field(
        ...,
        description='The python function for performing the desired task. \n                          This should always contain a keyword argument called `role` of type `Role` that defaults to None. ',
        title='Content',
    )


class SchemaAgentOutline(BaseModel):
    """
    A autonomous agent capable of carrying out actions according to input and output schema
    """

    name: str = Field(
        ...,
        description="The agent's name. It should refer to the agents desired function",
        title='Name',
    )
    profile: str = Field(
        ...,
        description="The agent's profile. It should give a brief description of the agent's desired function. This will be used as input to an LLM model",
        title='Profile',
    )
    goal: str = Field(
        ...,
        description="The agent's goal. This will be used as input to an LLM model",
        title='Goal',
    )
    actions: List[SchemaActionOutline] = Field(
        ..., description="The agent's list of actions", title='Actions'
    )
