import asyncio
from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.teams import Team
from typing import List, Dict
import requests
from xml.etree import ElementTree as ET
import re
from collections import defaultdict
from typing import Union

class Type1(BaseModel):
    """Test type 1"""
    a: str = Field(description="A string")

class Type2(BaseModel):
    """Test type 2"""
    b: str = Field(description="A string")

class Type3(BaseModel):
    """Test type 3"""
    a: str = Field(description="A string")
    b: str = Field(description="A string")

async def fun_1(user_input: str, role: Role = None) -> Type1:
    """Test function 1"""
    result = Type1(a = "a")
    return(result)

async def fun_2(type_1: Type1, role: Role = None) -> Type2:
    """Test function 2"""
    result = Type2(b = "b")
    return(result)

async def fun_3(input : Union[Type1, Type2], role: Role = None) -> Type3:
    """Test function 3"""
    print(input)
    return(Type3(a = "placeholder_a", b = "placeholder_b"))


# Main function
async def main():
    agents = []
    input_interpreter = Role(
        name="Tester",
        profile="A test agent",
        goal="To test stuff",
        constraints=None,
        actions=[fun_1, fun_2, fun_3],
    )
    agents.append(input_interpreter)

    team = Team(name="Testers", profile="A team of testers", investment=0.7)

    team.hire(agents)
    event_bus = team.get_event_bus()
    event_bus.register_default_events()
    user_request = "Test this"
    responses = await team.handle(Message(content=user_request, role="User"))
    print(responses)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()