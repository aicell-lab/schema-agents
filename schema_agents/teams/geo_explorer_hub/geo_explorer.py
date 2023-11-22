# import asyncio
# import json
# import traceback
# from typing import List, Union

# # from schema_agents.role import Role
# # from schema_agents.schema import Message
# # from schema_agents.tools.code_interpreter import create_mock_client

# # from schema_agents.teams.image_analysis_hub.schemas import (PythonFunctionScript, PythonFunctionScriptChanges, Change,
# #                       PythonFunctionScriptWithLineNumber, SoftwareRequirement)

from pydantic import BaseModel, Field
import os
import sys
import asyncio
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
from schema_agents.role import Role
from bioimageio_chatbot.chatbot import QuestionWithHistory
# from bioimageio_chatbot.chatbot import *

def load_geo_search_terms(fpath):
    with open(fpath, 'r') as f:
        search_terms = json.load(f)
    return(search_terms)

def python_geo_query_function(input_query:str)->[list[str],str]:
    import urllib.request
    import urllib.parse
    from xml.etree import ElementTree as ET

    base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
    search_url = f'{base_url}esearch.fcgi?db=gds&term={urllib.parse.quote(input_query)}&usehistory=y'
    with urllib.request.urlopen(search_url) as response:
        search_results = response.read()

    root = ET.fromstring(search_results)
    id_list = [id_tag.text for id_tag in root.findall('.//IdList/Id')]
    web_env = root.find('.//WebEnv').text

    return id_list, web_env

def fetch_significant_genes(geo_acc):
    base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
    db = 'geoprofiles'  # GEO Profiles database
    query = f'{geo_acc}[ACCN] AND "up down genes"[filter]'
    # Encoding the query parameters
    params = {
        'db': db,
        'term': query,
        'retmode': 'xml',  # You can change this depending on the desired response format,
        'retmax' : 100
    }
    query_string = urllib.parse.urlencode(params)
    # Constructing the complete URL
    url = f'{base_url}?{query_string}'
    # Making the request
    with urllib.request.urlopen(url) as response:
        result = response.read()
    root = ET.fromstring(result)
    id_list = [id_tag.text for id_tag in root.findall('.//IdList/Id')]
    return(id_list)


def fetch_gene_from_geoprofiles(geoprofiles_gene_id):
    base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi'
    params = {
        'dbfrom': 'geoprofiles',
        'db': 'gene',
        'id': geoprofiles_gene_id,
        'linkname': 'geoprofiles_gene',
        'retmode': 'json'  # This can be changed to 'json' if you prefer JSON format
    }
    query_string = urllib.parse.urlencode(params)
    url = f"{base_url}?{query_string}"

    with urllib.request.urlopen(url) as response:
        result = response.read()

    data = json.loads(result)
    gene_id_hits = []
    for linkset in data.get('linksets', []):
        for linksetdb in linkset.get('linksetdbs', []):
            ids = linksetdb.get('links', [])
            gene_id_hits.extend(ids)
    
    # return result.decode('utf-8')
    return(gene_id_hits)

def fetch_name_from_gene_ids(gene_ids):
    term = ','.join(gene_ids)
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=gene&id={term}&retmode=json"
    with urllib.request.urlopen(url) as response:
        result = response.read()
    content = result.decode('utf-8')
    content_chunks =content.split('\n\n')
    gene_names = []
    for chunk in content_chunks:
        gene_name = re.search(r"\. (\S*)", chunk)
        if gene_name:
            gene_names.append(gene_name.group(1))
    return(gene_names)

def plot_id_counts(gene_hits, message_text = None, out_plot_file = None):
    num_studies = len(gene_hits)
    # Flatten the list of lists
    all_ids = list(itertools.chain.from_iterable(gene_hits))

    # Count the occurrences of each ID
    id_counts = Counter(all_ids)

    # Filter IDs that appear more than once
    filtered_id_counts = {gene_id: count for gene_id, count in id_counts.items() if count > 1}

    # Sort by count and select top 10 (randomly select in case of ties)
    top_10_ids = sorted(filtered_id_counts, key=lambda x: (filtered_id_counts[x], random.random()))[-10:]
    # top_10_ids.reverse()

    top_10_counts = [filtered_id_counts[id] for id in top_10_ids]

    # Create a horizontal bar plot
    plt.figure(figsize=(10, 6))  # Adjust the size as needed
    plt.barh(top_10_ids, top_10_counts, color='skyblue')
    plt.ylabel('Gene IDs')
    plt.xlabel('Counts')
    title_text = f'Top Gene IDs by Count\n{num_studies} Total Studies'
    if message_text:
        title_text += f'\nUser: "{message_text}"'
    plt.title(title_text)
    plt.tight_layout()  # Adjusts the plot to ensure everything fits
    if out_plot_file:
        plt.savefig(out_plot_file, bbox_inches = 'tight')
    plt.show()
    

class geneHits(BaseModel):
    """A list of gene hit IDs from GEO queries"""
    request: str = Field(description="User's request in details")
    user_info: str = Field(description="Brief user info summary for personalized response, including name, backhground etc.")
    gene_list: list = Field(description="List of gene hits from GEO query")





def create_geo_querier(path_to_json):
    search_terms = load_geo_search_terms(path_to_json)
    search_terms_stats = f"""The following search terms are available: {[x['Field full name'] for x in search_terms]}.
    The description for each of these terms respectively is: {[x['Description'] for x in search_terms]}.
    The values and rules for each of these terms respectively is: {[x['Search term values and rules'] for x in search_terms]}.
    """

    class GEOQuery(BaseModel):
        """An NCBI GEO database query for obtaining information relevant to the user's request. The query MUST include a term for Entry Type = gds. The query must use the search term syntax outlined here:```{search_terms_stats}```"""
        request: str = Field(description="User's request in details")
        user_info: str = Field(description="Brief user info summary for personalized response, including name, background etc.")
        ncbi_geo_query: str = Field(description=f"""An NCBI GEO Datasets database Entrez query conforming to the user's request. The query MUST include a term for Entry Type = gds. Available terms (in JSON format) are: {search_terms_stats}""")

    async def create_geo_query(question_with_history: QuestionWithHistory = None, role: Role = None) -> str:
        """Answers the user's query by creating an ENTREZ query url for getting the relevant information from the GEO database"""
        inputs = [question_with_history.user_profile] + list(question_with_history.chat_history) + [question_with_history.question]
        req = await role.aask(inputs, GEOQuery)
        ncbi_geo_query = req.ncbi_geo_query
        print(f"ncbi_query:\n{ncbi_geo_query}\n")
        loop = asyncio.get_running_loop()
        ids, web_env = python_geo_query_function(ncbi_geo_query)
        geo_accs = [f"GDS{i}" for i in ids]
        print(f"GEO accesions: {geo_accs}")
        sig_gene_ids = [fetch_significant_genes(geo_acc) for geo_acc in geo_accs]
        gene_db_ids = [fetch_gene_from_geoprofiles(sg_id) for sg_id in sig_gene_ids]
        gene_names = [fetch_name_from_gene_ids(gids) for gids in gene_db_ids]
        sr = geneHits(request = req.request, user_info = req.user_info, gene_list = gene_names)
        return(sr)
    
    geo_querier = Role(
        name = "Dave",
        profile = "GEO database interactor",
        goal = "Your goal as Dave, the GEO database interactor, is to assist users in efectively utilizing the NCBI GEO database. You are responsible for answering user questions by writing GEO database queries in the form of Etrez eutils url queries.",
        constraints = None,
        actions = [create_geo_query]
    )

    return(geo_querier)