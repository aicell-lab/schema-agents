import asyncio
from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.teams import Team
from typing import List, Dict
from xml.etree import ElementTree as ET

class OBIDescription(BaseModel):
    """The sample description stated purely in terms of OBI ids"""
    desc : str = Field(..., description="The description of each sample purely in terms of OBI ids")

async def obi_convert(miniml_path : str, role : Role = None) -> OBIDescription:
    """Converts the miniml xml file to OBI ids"""
    with open(miniml_path, "r") as file:
        miniml_contents = file.read()
    result = await role.aask(miniml_contents, OBIDescription)
    return(result)


def make_team():
    agents = []
    data_scanner = Role(
        name="Data Scanner",
        profile="An agent that scans for available data",
        goal="To comprehensively scan for available data",
        constraints=None,
        actions=[obi_convert],
    )
    agents.append(data_scanner)

    team = Team(name="NCBI GEO analyzers", profile="A team of agents meant to comprehensively understand and analyze an NCBI GEO repository", investment=0.7)

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