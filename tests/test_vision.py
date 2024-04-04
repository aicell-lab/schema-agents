#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from schema_agents.role import Role
from schema_agents.schema import Message
from pydantic import BaseModel, Field
from typing import List


class Recipe(BaseModel):
    """A recipe."""
    name: str = Field(description="The name of the recipe.")
    ingredients: List[str] = Field(description="The list of ingredients.")
    instructions: str = Field(description="The instructions for making the recipe.")
    rating: float = Field(description="The rating of the recipe.")


class CookBook(BaseModel):
    """Creating a recipe book with a list of recipes based on the user's query."""
    name: str = Field(description="The name of the recipe book.")
    recipes: List[Recipe] = Field(description="The list of recipes in the book.")


async def respond_to_user(query: str, role: Role = None) -> CookBook:
    """Respond to user's request by recipe book."""
    response = await role.aask([{"role": "user", "content": [
        {"type": "text", "text": "Cook something like this"},
        {
            "type": "image_url",
            "image_url": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
            },
        },
    ]}], str)
    return response
    
@pytest.mark.asyncio
async def test_vision():
    alice = Role(
        name="Alice",
        profile="Cooker",
        goal="Your goal is to listen to user's request and propose recipes for making the most delicious meal for thanksgiving.",
        constraints=None,
        actions=[respond_to_user],
        model="gpt-4-vision-preview",
    )
    event_bus = alice.get_event_bus()
    event_bus.register_default_events()
    responses = await alice.handle(Message(content="make something to surprise our guest from Stockholm.", role="User"))
    print(responses)