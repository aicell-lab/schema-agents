import os
import asyncio
from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.teams import Team
from typing import List, Dict
from xml.etree import ElementTree as ET
from agents.database_scanners import GEOScanner


class GEOTeam(Team):
    def __init__(self):
        agents = [GEOScanner()]
        name = "GEO Team"
        profile = "A team of agents that scan the GEO database"
        investment = 0.7
        super().__init__(name = name, profile = profile, investment = investment)
        self.agents = []
        self.hire(agents)