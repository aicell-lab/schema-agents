from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import asyncio
from schema_agents.role import Role, Action
from schema_agents.schema import Message

import os
import urllib.request
import pandas as pd

class SearchQuery(BaseModel):
    """React UI Requirement
    The aim is to create a web UI for the main script in python to obtain user input, display results, and allow for interaction. 
    The exported imjoy plugin api function will be called inside the main function to interact with the user."""
    query: str = Field(..., description="Name of the React plugin for main function referencing.")
    database: str = Field(..., description="name of the databse, including positioning of elements.")


def gene_id_convert(gene_id: str) -> str:   
    """Input: database (columns), source type, source value. 
    Generate python code, execute python code, return another str (gene ensemble id type)."""
    return gene_id

def hpa_explorer(query: str) -> List[Union[str, Dict[str, str]]]:
    # check if umap_results_fit_all_transform_all.csv exists
    # if not, download from https://dl.dropbox.com/s/s4m2iysupy8gwj0/umap_results_fit_all_transform_all.csv
    # and save it to the current folder
    
    if not os.path.exists('umap_results_fit_all_transform_all.csv'):
        url = 'https://dl.dropbox.com/s/s4m2iysupy8gwj0/umap_results_fit_all_transform_all.csv'
        urllib.request.urlretrieve(url, 'umap_results_fit_all_transform_all.csv')
    df = pd.read_csv('umap_results_fit_all_transform_all.csv')

    return df.iloc[0]
 
# def pdb_explorer(query: str) -> List[Union[str, Dict[str, str]]]:
#     return ['456']


async def search(req: SearchQuery, role: Role) -> List[Union[str, Dict[str, str]]]:
    """Search for a gene in the HPA Cell Atlas and return the results."""
    
    if req.database == 'HPA Cell Atlas':
        check = await role.aask('if conversion is needed, please provide the gene id type.')
        if check:
            new_query = gene_id_convert(req.query)
        else:
            new_query = req.query
        return hpa_explorer(new_query)
    elif req.database == 'PDB':
        check = await role.aask('if conversion is needed, please provide the gene id type.')
        if check:
            new_query = gene_id_convert(req.query)
        else:
            new_query = req.query
        return pdb_explorer(new_query)
    

SearchManager = Role.create(name="Alice",
                                profile="Search Manager",
                                goal="Search for a gene in the HPA Cell Atlas and return the results.",
                                constraints=None,
                                actions=[search])

class HPADatabseExplorer(Team):
    """
    ImageAnalysisHub: a team of roles to create software for image analysis.
    """
    def recruit(self, client):
        """recruit roles to cooperate"""
        UXManager = Role.create(name="Luisa",
            profile="UX Manager",
            goal="Focus on understanding the user's needs and experience. Understand the user needs by interacting with user and communicate these findings to the project manager by calling `UserRequirements`.",
            constraints=None,
            actions=[partial(clarify_user_request, client), create_user_requirements])

        ProjectManager = Role.create(name="Alice",
                    profile="Project Manager",
                    goal="Efficiently communicate with the user and translate the user's needs into software requirements",
                    constraints=None,
                    actions=[create_software_requirements])

        WebDeveloper  = create_web_developer(client=client)
        DataEngineer = create_data_engineer(client=client)
        DevOps = Role.create(name="Bruce",
                    profile="DevOps",
                    goal="Deploy the software to the cloud and make it available to the user.",
                    constraints=None,
                    actions=[deploy_app])  
        self.environment.add_roles([UXManager(), ProjectManager(), DataEngineer(), WebDeveloper(), DevOps()]) 

