import os
from enum import Enum
import inspect
from schema_agents.provider.openai_api import retry
import asyncio
from paperqa import Docs
from typing import List, Optional, Union, Type, Any, get_type_hints, Tuple, Literal
from pydantic import BaseModel, Field, validator, create_model
from bioimageio_chatbot.utils import ChatbotExtension
import asyncio
from schema_agents.role import Message
from schema_agents import schema_tool, Role
import json

import re
import sys
import time
import urllib.request


# @schema_tool
# async def use_ncbi(ncbi_query_url : str = Field(description = "A query url to use the NCBI GEO database via the web API")) -> str:
#     """
# Uses the NCBI APIs to answer genomic questions. There are two types of Web APIs you can use: Eutils and BLAST.
# # Documentation 1
# You can call Eutils by:
# "[https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{esearch|efetch|esummary}.fcgi?db={gene|snp|onim}&retmax={}&tersm|id={}]" {API_DOC}

# # Documentation 2
# For DNA sequences, you can use BLAST by:

# """
#     response = await retry(ncbi_query_url)
#     return response

def call_api(url):
	time.sleep(1)
	url = url.replace(' ', '+')
	print(url)

	req = urllib.request.Request(url) 
	with urllib.request.urlopen(req) as response:
		call = response.read()

	return call

def get_prompt_header(mask):
    '''
    mask: [1/0 x 6], denotes whether each prompt component is used

    output: prompt
    '''
    url_1 = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&retmax=5&retmode=json&sort=relevance&term=LMP10'
    call_1 = call_api(url_1)

    url_2 = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=gene&retmax=5&retmode=json&id=19171,5699,8138'
    call_2 = call_api(url_2)

    url_3 = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=snp&retmax=10&retmode=json&id=1217074595' 
    call_3 = call_api(url_3)

    url_4 = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=omim&retmax=20&retmode=json&sort=relevance&term=Meesmann+corneal+dystrophy'
    call_4 = call_api(url_4)

    url_5 = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=omim&retmax=20&retmode=json&id=618767,601687,300778,148043,122100'
    call_5 = call_api(url_5)

    url_6 = 'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD=Put&PROGRAM=blastn&MEGABLAST=on&DATABASE=nt&FORMAT_TYPE=XML&QUERY=ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACCCTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGAAAAAAAAAAAAGTATTTCTCT&HITLIST_SIZE=5'
    call_6 = call_api(url_6)
    rid = re.search('RID = (.*)\n', call_6.decode('utf-8')).group(1)

    url_7 = f'https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD=Get&FORMAT_TYPE=Text&RID={rid}'
    time.sleep(30)
    call_7 = call_api(url_7)

    prompt = ''
    prompt += 'Hello. Your task is to use NCBI Web APIs to answer genomic questions.\n'
    #prompt += 'There are two types of Web APIs you can use: Eutils and BLAST.\n\n'

    if mask[0]:
        # Doc 0 is about Eutils
        prompt += 'You can call Eutils by: "[https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{esearch|efetch|esummary}.fcgi?db={gene|snp|omim}&retmax={}&{term|id}={term|id}]".\n'
        prompt += 'esearch: input is a search term and output is database id(s).\n'
        prompt += 'efectch/esummary: input is database id(s) and output is full records or summaries that contain name, chromosome location, and other information.\n'
        prompt += 'Normally, you need to first call esearch to get the database id(s) of the search term, and then call efectch/esummary to get the information with the database id(s).\n'
        prompt += 'Database: gene is for genes, snp is for SNPs, and omim is for genetic diseases.\n\n'

    if mask[1]:
        # Doc 1 is about BLAST
        prompt += 'For DNA sequences, you can use BLAST by: "[https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD={Put|Get}&PROGRAM=blastn&MEGABLAST=on&DATABASE=nt&FORMAT_TYPE={XML|Text}&QUERY={sequence}&HITLIST_SIZE={max_hit_size}]".\n'
        prompt += 'BLAST maps a specific DNA {sequence} to its chromosome location among different specices.\n'
        prompt += 'You need to first PUT the BLAST request and then GET the results using the RID returned by PUT.\n\n'

    if any(mask[2:]):
        prompt += 'Here are some examples:\n\n'

    if mask[2]:
        # Example 1 is from gene alias task 
        prompt += f'Question: What is the official gene symbol of LMP10?\n'
        prompt += f'[{url_1}]->[{call_1}]\n' 
        prompt += f'[{url_2}]->[{call_2}]\n'
        prompt += f'Answer: PSMB10\n\n'

    if mask[3]:
        # Example 2 is from SNP gene task
        prompt += f'Question: Which gene is SNP rs1217074595 associated with?\n'
        prompt += f'[{url_3}]->[{call_3}]\n'
        prompt += f'Answer: LINC01270\n\n'

    if mask[4]:
        # Example 3 is from gene disease association
        prompt += f'Question: What are genes related to Meesmann corneal dystrophy?\n'
        prompt += f'[{url_4}]->[{call_4}]\n'
        prompt += f'[{url_5}]->[{call_5}]\n'
        prompt += f'Answer: KRT12, KRT3\n\n'

    if mask[5]:
        # Example 4 is for BLAST
        prompt += f'Question: Align the DNA sequence to the human genome:ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACCCTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGAAAAAAAAAAAAGTATTTCTCT\n'
        prompt += f'[{url_6}]->[{rid}]\n'
        prompt += f'[{url_7}]->[{call_7}]\n'
        prompt += f'Answer: chr15:91950805-91950932\n\n'

    return prompt




async def main():
    manager = Role(
        name="manager",
        instructions = "You are the manager. Your job is to complete the user's task completely. If it fails, revise your plan and keep trying until it's done",
        constraints=None,
        register_default_events=True,
    )

    response, metadata = await manager.acall("Figure out how to download the data associated with the paper located at /Users/gkreder/gdrive/exponential-chain/GSE254364/405.pdf, open the data and give me a complete summary of the files, their formats, and their contents",
                                   [ask_pdf_paper, search_web],
                                   return_metadata=True,
                                   max_loop_count = 10,
                                   thoughts_schema=ThoughtsSchema
                                   )
    print(response)

s = """Your task is to use NCBI Web APIs to answer genomic questions.
You can call Eutils by: "[https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{esearch|efetch|esummary}.fcgi?db={gene|snp|omim}&retmax={}&{term|id}={term|id}]".
esearch: input is a search term and output is database id(s).
efectch/esummary: input is database id(s) and output is full records or summaries that contain name, chromosome location, and other information.
Normally, you need to first call esearch to get the database id(s) of the search term, and then call efectch/esummary to get the information with the database id(s).
Database: gene is for genes, snp is for SNPs, and omim is for genetic diseases.

For DNA sequences, you can use BLAST by: "[https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD={Put|Get}&PROGRAM=blastn&MEGABLAST=on&DATABASE=nt&FORMAT_TYPE={XML|Text}&QUERY={sequence}&HITLIST_SIZE={max_hit_size}]".
BLAST maps a specific DNA {sequence} to its chromosome location among different specices.
You need to first PUT the BLAST request and then GET the results using the RID returned by PUT.

Here are some examples:

Question: What is the official gene symbol of LMP10?
[https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&retmax=5&retmode=json&sort=relevance&term=LMP10]->[b'{"header":{"type":"esearch","version":"0.3"},"esearchresult":{"count":"3","retmax":"3","retstart":"0","idlist":["5699","8138","19171"],"translationset":[],"translationstack":[{"term":"LMP10[All Fields]","field":"All Fields","count":"3","explode":"N"},"GROUP"],"querytranslation":"LMP10[All Fields]"}}\n']
[https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=gene&retmax=5&retmode=json&id=19171,5699,8138]->[b'\n1. Psmb10\nOfficial Symbol: Psmb10 and Name: proteasome (prosome, macropain) subunit, beta type 10 [Mus musculus (house mouse)]\nOther Aliases: Mecl-1, Mecl1\nOther Designations: proteasome subunit beta type-10; low molecular mass protein 10; macropain subunit MECl-1; multicatalytic endopeptidase complex subunit MECl-1; prosome Mecl1; proteasome (prosomome, macropain) subunit, beta type 10; proteasome MECl-1; proteasome subunit MECL1; proteasome subunit beta-2i\nChromosome: 8; Location: 8 53.06 cM\nAnnotation: Chromosome 8 NC_000074.7 (106662360..106665024, complement)\nID: 19171\n\n2. PSMB10\nOfficial Symbol: PSMB10 and Name: proteasome 20S subunit beta 10 [Homo sapiens (human)]\nOther Aliases: LMP10, MECL1, PRAAS5, beta2i\nOther Designations: proteasome subunit beta type-10; low molecular mass protein 10; macropain subunit MECl-1; multicatalytic endopeptidase complex subunit MECl-1; proteasome (prosome, macropain) subunit, beta type, 10; proteasome MECl-1; proteasome catalytic subunit 2i; proteasome subunit MECL1; proteasome subunit beta 10; proteasome subunit beta 7i; proteasome subunit beta-2i; proteasome subunit beta2i\nChromosome: 16; Location: 16q22.1\nAnnotation: Chromosome 16 NC_000016.10 (67934506..67936850, complement)\nMIM: 176847\nID: 5699\n\n3. MECL1\nProteosome subunit MECL1 [Homo sapiens (human)]\nOther Aliases: LMP10, PSMB10\nThis record was replaced with GeneID: 5699\nID: 8138\n\n']
Answer: PSMB10

Question: Which gene is SNP rs1217074595 associated with?
[https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=snp&retmax=10&retmode=json&id=1217074595]->[b'{"header":{"type":"esummary","version":"0.3"},"result":{"uids":["1217074595"],"1217074595":{"uid":"1217074595","snp_id":1217074595,"allele_origin":"","global_mafs":[{"study":"GnomAD","freq":"A=0.000007/1"},{"study":"TOPMED","freq":"A=0.000004/1"},{"study":"ALFA","freq":"A=0./0"}],"global_population":"","global_samplesize":"","suspected":"","clinical_significance":"","genes":[{"name":"LINC01270","gene_id":"284751"}],"acc":"NC_000020.11","chr":"20","handle":"GNOMAD,TOPMED","spdi":"NC_000020.11:50298394:G:A","fxn_class":"non_coding_transcript_variant","validated":"by-frequency,by-alfa,by-cluster","docsum":"HGVS=NC_000020.11:g.50298395G>A,NC_000020.10:g.48914932G>A,NR_034124.1:n.351G>A,NM_001025463.1:c.*4G>A|SEQ=[G/A]|LEN=1|GENE=LINC01270:284751","tax_id":9606,"orig_build":155,"upd_build":156,"createdate":"2017/11/09 09:55","updatedate":"2022/10/13 17:11","ss":"4354715686,5091242333","allele":"R","snp_class":"snv","chrpos":"20:50298395","chrpos_prev_assm":"20:48914932","text":"","snp_id_sort":"1217074595","clinical_sort":"0","cited_sort":"","chrpos_sort":"0050298395","merged_sort":"0"}}}\n']
Answer: LINC01270

Question: What are genes related to Meesmann corneal dystrophy?
[https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=omim&retmax=20&retmode=json&sort=relevance&term=Meesmann+corneal+dystrophy]->[b'{"header":{"type":"esearch","version":"0.3"},"esearchresult":{"count":"5","retmax":"5","retstart":"0","idlist":["122100","618767","300778","601687","148043"],"translationset":[],"translationstack":[{"term":"Meesmann[All Fields]","field":"All Fields","count":"5","explode":"N"},{"term":"corneal[All Fields]","field":"All Fields","count":"565","explode":"N"},"AND",{"term":"dystrophy[All Fields]","field":"All Fields","count":"1355","explode":"N"},"AND","GROUP"],"querytranslation":"Meesmann[All Fields] AND corneal[All Fields] AND dystrophy[All Fields]"}}\n']
[https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=omim&retmax=20&retmode=json&id=618767,601687,300778,148043,122100]->[b'{"header":{"type":"esummary","version":"0.3"},"result":{"uids":["618767","601687","300778","148043","122100"],"618767":{"uid":"618767","oid":"#618767","title":"CORNEAL DYSTROPHY, MEESMANN, 2; MECD2","alttitles":"","locus":"602767"},"601687":{"uid":"601687","oid":"*601687","title":"KERATIN 12, TYPE I; KRT12","alttitles":"","locus":""},"300778":{"uid":"300778","oid":"%300778","title":"CORNEAL DYSTROPHY, LISCH EPITHELIAL; LECD","alttitles":"","locus":"#6####"},"148043":{"uid":"148043","oid":"*148043","title":"KERATIN 3, TYPE II; KRT3","alttitles":"","locus":"mutation"},"122100":{"uid":"122100","oid":"#122100","title":"CORNEAL DYSTROPHY, MEESMANN, 1; MECD1","alttitles":"","locus":"P#####"}}}\n']
Answer: KRT12, KRT3

Question: Align the DNA sequence to the human genome:ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACCCTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGAAAAAAAAAAAAGTATTTCTCT
[https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD=Put&PROGRAM=blastn&MEGABLAST=on&DATABASE=nt&FORMAT_TYPE=XML&QUERY=ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACCCTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGAAAAAAAAAAAAGTATTTCTCT&HITLIST_SIZE=5]->[YBWG5AYT013]
[https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD=Get&FORMAT_TYPE=Text&RID=YBWG5AYT013]->[b'<p><!--\nQBlastInfoBegin\n\tStatus=READY\nQBlastInfoEnd\n--><p>\n<PRE>\nBLASTN 2.15.0+\nReference: Zheng Zhang, Scott Schwartz, Lukas Wagner, and\nWebb Miller (2000), "A greedy algorithm for aligning DNA\nsequences", J Comput Biol 2000; 7(1-2):203-14.\n\n\nReference for database indexing: Aleksandr Morgulis, George\nCoulouris, Yan Raytselis, Thomas L. Madden, Richa Agarwala,\nAlejandro A. Schaffer (2008), "Database Indexing for\nProduction MegaBLAST Searches", Bioinformatics 24:1757-1764.\n\n\nRID: YBWG5AYT013\n\n\nDatabase: Nucleotide collection (nt)\n           103,965,835 sequences; 1,574,905,710,618 total letters\nQuery= \nLength=128\n\n\n                                                                   Score     E     Max\nSequences producing significant alignments:                       (Bits)  Value  Ident\n\nCP034493.1 Eukaryotic synthetic construct chromosome 15            237     5e-58  100%      \nCP139551.1 Homo sapiens isolate NA24385 chromosome 15              237     5e-58  100%      \nNG_132175.1 Homo sapiens H3K27ac-H3K4me1 hESC enhancer GRCh37_ 237     5e-58  100%      \nCP068263.2 Homo sapiens isolate CHM13 chromosome 15                237     5e-58  100%      \nAP023475.1 Homo sapiens DNA, chromosome 15, nearly complete ge 237     5e-58  100%      \n\nALIGNMENTS\n>CP034493.1 Eukaryotic synthetic construct chromosome 15\n CP034518.1 Eukaryotic synthetic construct chromosome 15\nLength=82521392\n\n Score = 237 bits (128),  Expect = 5e-58\n Identities = 128/128 (100%), Gaps = 0/128 (0%)\n Strand=Plus/Plus\n\nQuery  1         ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACC  60\n                 ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nSbjct  72494035  ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACC  72494094\n\nQuery  61        CTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGaaaaaaaaaaaaGT  120\n                 ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nSbjct  72494095  CTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGAAAAAAAAAAAAGT  72494154\n\nQuery  121       ATTTCTCT  128\n                 ||||||||\nSbjct  72494155  ATTTCTCT  72494162\n\n\n>CP139551.1 Homo sapiens isolate NA24385 chromosome 15\nLength=96257017\n\n Score = 237 bits (128),  Expect = 5e-58\n Identities = 128/128 (100%), Gaps = 0/128 (0%)\n Strand=Plus/Plus\n\nQuery  1         ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACC  60\n                 ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nSbjct  86191593  ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACC  86191652\n\nQuery  61        CTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGaaaaaaaaaaaaGT  120\n                 ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nSbjct  86191653  CTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGAAAAAAAAAAAAGT  86191712\n\nQuery  121       ATTTCTCT  128\n                 ||||||||\nSbjct  86191713  ATTTCTCT  86191720\n\n\n>NG_132175.1 Homo sapiens H3K27ac-H3K4me1 hESC enhancer GRCh37_chr15:92493309-92494181 \n(LOC127830695) on chromosome 15\nLength=1073\n\n Score = 237 bits (128),  Expect = 5e-58\n Identities = 128/128 (100%), Gaps = 0/128 (0%)\n Strand=Plus/Plus\n\nQuery  1    ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACC  60\n            ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nSbjct  827  ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACC  886\n\nQuery  61   CTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGaaaaaaaaaaaaGT  120\n            ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nSbjct  887  CTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGAAAAAAAAAAAAGT  946\n\nQuery  121  ATTTCTCT  128\n            ||||||||\nSbjct  947  ATTTCTCT  954\n\n\n>CP068263.2 Homo sapiens isolate CHM13 chromosome 15\nLength=99753195\n\n Score = 237 bits (128),  Expect = 5e-58\n Identities = 128/128 (100%), Gaps = 0/128 (0%)\n Strand=Plus/Plus\n\nQuery  1         ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACC  60\n                 ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nSbjct  89712558  ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACC  89712617\n\nQuery  61        CTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGaaaaaaaaaaaaGT  120\n                 ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nSbjct  89712618  CTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGAAAAAAAAAAAAGT  89712677\n\nQuery  121       ATTTCTCT  128\n                 ||||||||\nSbjct  89712678  ATTTCTCT  89712685\n\n\n>AP023475.1 Homo sapiens DNA, chromosome 15, nearly complete genome\nLength=95537968\n\n Score = 237 bits (128),  Expect = 5e-58\n Identities = 128/128 (100%), Gaps = 0/128 (0%)\n Strand=Plus/Plus\n\nQuery  1         ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACC  60\n                 ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nSbjct  85572367  ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACC  85572426\n\nQuery  61        CTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGaaaaaaaaaaaaGT  120\n                 ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nSbjct  85572427  CTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGAAAAAAAAAAAAGT  85572486\n\nQuery  121       ATTTCTCT  128\n                 ||||||||\nSbjct  85572487  ATTTCTCT  85572494\n\n\n  Database: Nucleotide collection (nt)\n    Posted date:  Mar 1, 2024  12:40 PM\n  Number of letters in database: 1,574,905,710,618\n  Number of sequences in database:  103,965,835\n\nLambda      K        H\n    1.33    0.621     1.12 \nGapped\nLambda      K        H\n    1.28    0.460    0.850 \nMatrix: blastn matrix:1 -2\nGap Penalties: Existence: 0, Extension: 0\nNumber of Sequences: 103965835\nNumber of Hits to DB: 0\nNumber of extensions: 0\nNumber of successful extensions: 0\nNumber of sequences better than 10: 1\nNumber of HSP\'s better than 10 without gapping: 0\nNumber of HSP\'s gapped: 1\nNumber of HSP\'s successfully gapped: 1\nLength of query: 128\nLength of database: 1574905710618\nLength adjustment: 35\nEffective length of query: 93\nEffective length of database: 1571266906393\nEffective search space: 146127822294549\nEffective search space used: 146127822294549\nA: 0\nX1: 13 (25.0 bits)\nX2: 32 (59.1 bits)\nX3: 54 (99.7 bits)\nS1: 13 (25.1 bits)\nS2: 24 (45.4 bits)\n\n\n']
Answer: chr15:91950805-91950932"""

if __name__ == "__main__":
    # loop = asyncio.get_event_loop()
    # loop.create_task(main())
    # loop.run_forever()
    s = get_prompt_header([1, 1, 1, 1, 1, 1])

    print(s)






