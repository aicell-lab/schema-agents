"""A simple demo for creating an agent for generating a recipe book based on user's query."""
import asyncio
from schema_agents import Role, schema_tool
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

    
async def main():
    alice = Role(
        name="Alice",
        profile="Cooker",
        goal="Your goal is to listen to user's request and propose recipes for making the most delicious meal for thanksgiving.",
        constraints=None,
        register_default_events=True,
    )
    
    # Let's define a recipe book for Alice to use
    recipies = await alice.aask("make something to surprise our guest from Stockholm.", CookBook)
    print(recipies)
    
    # Let's define some tools for Alice to use
    
    @schema_tool
    def go_shopping(ingredients: List[str]) -> str:
        """Go shopping for the ingredients."""
        return "I have bought all the ingredients."
    
    @schema_tool
    def cook(recipe: Recipe) -> str:
        """Cook the recipe."""
        return f"I have cooked {recipe.name}."
    
    # Now let's call Alice to prepare a dinner for our guest
    response = await alice.acall(["Let's prepare a dinner for our guest!", recipies], [go_shopping, cook])
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
