import asyncio
from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.teams import Team
from typing import List, Dict
from xml.etree import ElementTree as ET

class Type1(BaseModel):
    """Test type 1"""
    a: str = Field(description="A string")

class Type2(BaseModel):
    """Test type 2"""
    b: str = Field(description="A string")

class Type3(BaseModel):
    """Test type 3"""
    c: Type1 = Field(description="A field of type1")

async def fun_1(user_input: str, role: Role = None) -> Type1:
    """Test function 1"""
    result = Type1(a = "a")
    return(result)

async def fun_2(type_1: Type1, role: Role = None) -> Type2:
    """Test function 2"""
    result = Type2(b = "b")
    return(result)

async def fun_3(input_1 : Type1, role: Role = None) -> Type3:
    """Test function 3"""
    result = await role.aask(input_1, Type3)
    return(result)

def make_team():
    agents = []
    input_interpreter = Role(
        name="Tester",
        profile="A test agent",
        goal="To test stuff",
        constraints=None,
        actions=[fun_1, fun_2],
    )
    agents.append(input_interpreter)

    agent_2 = Role(
        name="Tester2",
        profile="A second test agent",
        goal="To test more stuff",
        constraints=None,
        actions=[fun_3],
    )
    agents.append(agent_2)

    team = Team(name="Testers", profile="A team of testers", investment=0.7)

    team.hire(agents)
    event_bus = team.get_event_bus()
    event_bus.register_default_events()
    return(team)

# Main function
async def main():
    team = make_team()
    user_request = "Test this"
    responses = await team.handle(Message(content=user_request, role="User"))
    print(responses)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()