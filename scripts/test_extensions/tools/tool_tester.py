import asyncio

async def main():
    from paperqa_tools import ask_pdf_paper
    res = await ask_pdf_paper(file_location = "PMC10912389/main.pdf",
                         question = "What are the paper's significant findings?")
    print(res)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()
    


