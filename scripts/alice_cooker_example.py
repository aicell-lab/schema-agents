"""A simple demo for creating an agent for generating a recipe book based on user's query."""
import asyncio
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
    response = await role.aask(query, CookBook)
    return response
    
    
async def main():
    alice = Role(
        name="Alice",
        profile="Cooker",
        goal="Your goal is to listen to user's request and propose recipes for making the most delicious meal for thanksgiving.",
        constraints=None,
        actions=[respond_to_user],
    )
    event_bus = alice.get_event_bus()
    event_bus.register_default_events()
    responses = await alice.handle(Message(content="make something to surprise our guest from Stockholm.", role="User"))
    print(responses)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()
