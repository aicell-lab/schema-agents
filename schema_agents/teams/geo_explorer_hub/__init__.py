import asyncio
from functools import partial
from typing import Union
from schema_agents.role import Role
from schema_agents.teams import Team
import os
import sys
import json
import xml.etree.ElementTree as ET
import re
import urllib
import urllib.request
import urllib.parse
import matplotlib.pyplot as plt
from collections import Counter
import itertools
import random
from schema_agents.teams.geo_explorer_hub.geo_explorer import create_geo_querier



def create_geo_explorer_hub(client, investment, path_to_json):
    """Create a team for NCBI Geo exploration"""
    team = Team(name = "Geo Explorer Hub", profile = "A team consisting of a single role for querying the NCBI GEO database", goal = "Query the NCBI GEO database and process results", investment = investment)
    geo_querier = create_geo_querier(path_to_json)
    team.hire([geo_querier])
    return(team)