import os
from typing import List

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer

from langchain_api.embeddings.openai_tools import OpenAIEmbeds
from langchain_api.vector_search import FAISSVectorStore
from langchain_api.chains.qa_toolchain import setup_qa_chain
from langchain_api.models import OpenAIModel

loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()
faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())

qa_chain = setup_qa_chain(OpenAIModel(), chain_variant="basic")
question = "What are the main findings of the paper?"
relevant_docs = vector_store.find_similar_texts(question)
response = qa_chain.execute(input_docs=relevant_docs, query_text=question)
print(response)