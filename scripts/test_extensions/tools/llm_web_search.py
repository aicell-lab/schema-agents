from xml.etree import cElementTree as ET
import os
import httpx
from bs4 import BeautifulSoup
from langchain.schema import Document
from pydantic import Field
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.document_compressors.embeddings_filter import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from .langchain_websearch import LangchainCompressor
from schema_agents import schema_tool, Role
import re

def preprocess_text(text: str) -> str:
    text = text.replace("\n", " \n")
    spaces_regex = re.compile(r" {3,}")
    text = spaces_regex.sub(" ", text)
    text = text.strip()
    return text

def extract_text(element):
    text = ""
    if element.text:
        text += element.text
    for child in element:
        text += extract_text(child)
        if child.tail:
            text += child.tail
    return text

def pubmed_xml_to_text(pubmed_response : str):
    tree = ET.fromstring(pubmed_response)
    article_body = tree.find(".//body")
    article_text = extract_text(article_body).strip()
    return article_text

@schema_tool
async def search_pubmed_paper(query : str = Field(description = "The query to run on the paper"),
                        pmc_id : str = Field(description = "The PMC ID of the paper to query in the correct format (e.g. PMC)"),
                        chunk_size : int = Field(description = "The chunk size to use when breaking up the paper text into smaller chunks for processing. Defaults to 500"),
                        num_results : int = Field(description = "The number of results to return. Defaults to 5"),
                        similarity_threshold : float = Field(description="The similarity threshold to use when filtering out query search results. Defaults to 0.5")) -> list[Document]:
# chunk_size : int = Field(descriotion = )500,
#                         num_results : int = 5,
#                         similarity_threshold : float = 0.5):
    """Take a pubmed paper ID and run a query on it. The query is a plaintext string. The function returns the top `num_results` most relevant text chunks from the paper."""
    from .NCBI import call_api
    documents = []
    pmc_fname = f"{pmc_id}.txt"
    pubmed_query_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmc_id}"
    if not os.path.exists(pmc_fname):
        pubmed_response = call_api(pubmed_query_url).decode()
        pubmed_text = pubmed_xml_to_text(pubmed_response)
        with open(pmc_fname, "w") as f:
            f.write(pubmed_text)
    else:
        with open(pmc_fname, "r") as f:
            pubmed_text = f.read()
    documents = [Document(page_content=pubmed_text, metadata={"source": pmc_id})]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=10,
                                                       separators=["\n\n", "\n", ".", ", ", " ", ""])
    split_docs = text_splitter.split_documents(documents)
    embedding = OpenAIEmbeddings()
    faiss_retriever = FAISS.from_documents(split_docs, embedding).as_retriever(search_kwargs={"k": 5})
    bm25_retriever = BM25Retriever.from_documents(split_docs, preprocess_func=preprocess_text)
    bm25_retriever.k = num_results
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embedding)
    embeddings_filter = EmbeddingsFilter(embeddings=embedding, k=None, similarity_threshold=similarity_threshold)
    pipeline_compressor = DocumentCompressorPipeline(transformers=[redundant_filter, embeddings_filter])
    compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=faiss_retriever)
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, compression_retriever], weights=[0.5, 0.5])
    compressed_docs = await ensemble_retriever.aget_relevant_documents(query)
    res = compressed_docs[:num_results]
    return res


async def search_duckduckgo(query: str, langchain_compressor: LangchainCompressor,
                                max_results: int, similarity_threshold: float, instant_answers: bool,
                                chunk_size: int, num_results_to_process: int):
    
    from duckduckgo_search import AsyncDDGS
    documents = []
    query = query.strip("\"'")
    async with AsyncDDGS() as ddgs:
        if instant_answers:
            answer_list = []
            async for answer in ddgs.answers(query):
                answer_list.append(answer)
            if answer_list:
                max_results -= 1  # We already have 1 result now
                answer_dict = answer_list[0]
                instant_answer_doc = Document(page_content=answer_dict["text"],
                                              metadata={"source": answer_dict["url"]})
                documents.append(instant_answer_doc)

        results = []
        result_urls = []
        async for result in ddgs.text(query, region='wt-wt', safesearch='moderate', timelimit=None,
                                max_results=num_results_to_process):
            results.append(result)
            result_urls.append(result["href"])

    documents.extend(await langchain_compressor.faiss_embedding_query_urls(query, result_urls,
                                                               num_results=num_results_to_process,
                                                               similarity_threshold=similarity_threshold,
                                                               chunk_size=chunk_size))
    
    if not documents:    # Fall back to old simple search rather than returning nothing
        print("LLM_Web_search | Could not find any page content "
              "similar enough to be extracted, using basic search fallback...")
        data = results[:max_results]
        docs = []
        for d in data:
            docs.append({"title": d['title'], "body": d['body'], "href": d['href']})
        return docs
    
    docs = []
    for doc in documents[:max_results]:
        docs.append({"content": doc.page_content, "url": doc.metadata["source"]})
    return docs


async def get_webpage_content(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)

    soup = BeautifulSoup(response.content, features="lxml")
    for script in soup(["script", "style"]):
        script.extract()

    strings = soup.stripped_strings
    return '\n'.join([s.strip() for s in strings])
