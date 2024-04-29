import asyncio
import pprint

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import AsyncHtmlLoader


# TESTING
if __name__ == "__main__":
    # # url = "https://www.patagonia.ca/shop/new-arrivals"
    # url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE244452"

    # async def scrape_playwright():
    #     results = await ascrape_playwright(url)
    #     print(results)

    # pprint.pprint(asyncio.run(scrape_playwright()))
    

    # Load HTML
    # loader = AsyncChromiumLoader(["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE244452"])
    # html = loader.load()
    # print(html[0])



    urls = ["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE244452"]
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()
    content = docs[0]
    print(content)