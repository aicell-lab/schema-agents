# from langchain_community.document_loaders import AsyncChromiumLoader
# from langchain_community.document_transformers import BeautifulSoupTransformer

# # Load HTML
# loader = AsyncChromiumLoader(["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE244452"])
# html = loader.load()
# with open("/Users/gkreder/Downloads/test_ncbi.html", 'w') as f:
#     f.write(html[0].page_content)

# # print(html[0].page_content)

from langchain.docstore.document import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.utilities import ApifyWrapper

apify = ApifyWrapper()
# Call the Actor to obtain text from the crawled webpages
loader = apify.call_actor(
    actor_id="apify/website-content-crawler",
    run_input={
        "startUrls": [{"url": "https://python.langchain.com/docs/integrations/chat/"}]
    },
    dataset_mapping_function=lambda item: Document(
        page_content=item["text"] or "", metadata={"source": item["url"]}
    ),
)

# Create a vector store based on the crawled data
index = VectorstoreIndexCreator().from_loaders([loader])

# Query the vector store
query = "Are any OpenAI chat models integrated in LangChain?"
result = index.query(query)
print(result)