import asyncio
import ollama

client = ollama.AsyncClient(host="http://localhost:11434")


# async def chat():
#     # await client.pull('nexusraven')
#     async for part in await client.chat(model='nexusraven', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}], stream=True):
#         print(part['message']['content'], end='', flush=True)


# asyncio.run(chat())


async def chat():
  message = {'role': 'user', 'content': 'Why is the sky blue?'}
  async for part in await client.chat(model='nexusraven', messages=[message], stream=True):
    print(part['message']['content'], end='', flush=True)

asyncio.run(chat())