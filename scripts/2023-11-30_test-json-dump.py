import asyncio
from pydantic import BaseModel, Field
from schema_agents.role import Role
from schema_agents.schema import Message
from schema_agents.teams import Team
from typing import List, Dict
import json



class StructuredQuery(BaseModel):
    search_terms: List[str] = Field(description = "A list of key terms extracted from the user input that are relevant to the search.")
    query_parameters: Dict[str, str] = Field(description = "A dictionary of parameters that will be used to structure the query for the database.")


if __name__ == "__main__":
    # json_str = """{}"""
    sq = StructuredQuery(search_terms = ['terms'], query_parameters = {'q1' : 'a1'})
    with open('StructuredQuery.json', 'w') as f:
        print(json.dumps(StructuredQuery.model_json_schema()), file = f)

    print(1)          


