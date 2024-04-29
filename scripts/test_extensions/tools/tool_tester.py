import asyncio

async def test_ask_pdf_paper():
    from paperqa_tools import ask_pdf_paper
    res = await ask_pdf_paper(file_location = "PMC10912389/main.pdf",
                            question = "What are the paper's significant findings?")
    print(res)

async def test_pubmed_oa(pmc_id):
    from NCBI import get_pubmed_central_oa
    res = await get_pubmed_central_oa(pmc_id = pmc_id)
    print(res)

async def main():
    await test_pubmed_oa("PMC10912389")
    await test_pubmed_oa("7615674")
    await test_pubmed_oa("123456789")
    
    

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()
    


