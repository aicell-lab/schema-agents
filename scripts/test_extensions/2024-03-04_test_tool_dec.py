from schema_agents import Role, tool

alice = Role(
    name="Alice",
    profile="Cooker",
    goal="Your goal is to listen to user's request and propose recipes for making the most delicious meal for thanksgiving.",
    constraints=None,
    register_default_events=True,
)

@tool
def go_shopping(ingredients: List[str]) -> str:
    """Go shopping for the ingredients."""
    return "I have bought all the ingredients."

@tool
def cook(recipe: Recipe) -> str:
    """Cook the recipe."""
    return f"I have cooked {recipe.name}."

async def main():
    # Now let's call Alice to prepare a dinner for our guest
    response = await alice.acall("Let's prepare a dinner for our guest from stockholm", [go_shopping, cook])
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())