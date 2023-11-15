import json
import time
import uuid
import asyncio
import secrets
from pydantic import BaseModel
from imjoy_rpc.hypha import connect_to_server, login
from schema_agents.utils import dict_to_md
from schema_agents.config import CONFIG
from schema_agents.teams.image_analysis_hub import create_image_analysis_hub
import logging
from schema_agents.tools.code_interpreter import create_mock_client
logging.basicConfig(level=logging.DEBUG)

SERVER_URL = "https://ai.imjoy.io"

async def chat(msg, client, context=None):
    """Hypha bot chat to help you with various tasks"""
    # if context["user"]['id'] != "github|478667":
    #     raise Exception(f"Sorry, you (user: {context['user']}) are not authorized to use this service.")
    conversation_id = str(uuid.uuid4())
    session_id = conversation_id or secrets.token_hex(8)
    message_context = {"parent_id": msg.messageId}
    if client:
        await client.initialize({"conversationId": conversation_id })
        await client.showMessage("Hey, I am on it! 👩‍💻")

    async def create_new_message(name, query_id):
        message_context[query_id] = str(uuid.uuid4())
        promise = client.newMessage({
            "messageId": message_context[query_id],
            "parentMessageId": message_context["parent_id"],
            "sender": "ChatGPT",
            "text": f"\n\n## Generating response for {name}\n```json\n",
            "submitting": True,
        })
        # update parent
        message_context["parent_id"] = message_context[query_id]
        await promise

    async def stream_callback(message):
        if message["session_id"] == session_id or not client:
            return
        if message["type"] == "function_call":
            if message["status"] == "start":
                await create_new_message(message["name"], message["query_id"])
                # await client.appendText({"messageId": message_context[message["query_id"]], "text": f"\n\n## Generating response for {message['name']}\n```json\n"})
            elif message["status"] == "in_progress":
                await client.appendText({"messageId": message_context[message["query_id"]], "text": message["arguments"].replace("\\n", "\n")})
                # print(message["arguments"], end="")
            elif message["status"] == "finished":
                await client.appendText({"messageId": message_context[message["query_id"]], "text": "\n```\n\n", "submitting": False})
            # else:
                # print(f'\nGenerating {message["name"]} ({message["status"]}): {message["arguments"]}')
        elif message["type"] == "text":
            if not message_context.get(message["query_id"]):
                await create_new_message(message["name"], message["query_id"])
            # print(message["content"], end="")
            await client.appendText({"messageId": message_context[message["query_id"]], "text": message["arguments"]})
            

    hub = create_image_analysis_hub(client=client, investment=0.5)
    event_bus = hub.get_event_bus()
    event_bus.register_default_events()
    # event_bus.on("message", message_callback)
    event_bus.on("stream", stream_callback)
    await hub.handle(msg.text)

async def test_chat():
    client = create_mock_client(form_data={
        "microscopy_image_type": "widefield",
        "image_format": "png",
        "cell_shape": "round",
        "goal": "segment the image in green channel, the background is black",
    })
    await chat("create a tool for counting cells in microscopy images", client, {"user": {"id": "github|478667"}})

async def test_microscope():
    client=create_mock_client(form_data={
        "microscopy_image_type": "widefield",
        "path": "./images",
        "exposure": 0.1,
    })
    class Message(BaseModel):
        messageId: str
        text: str
    
    await chat(Message(text="acquire an image every 1 second for 3.5 seconds", messageId="123"), client, {"user": {"id": "github|478667"}})
    loop = asyncio.get_event_loop()
    loop.stop()
    
async def main():
    # client_id = str(uuid.uuid3(uuid.NAMESPACE_DNS, f'urn:node:{hex(uuid.getnode())}'))
    # token = login({"server_url": SERVER_URL})
    print("Connecting to server...", flush=True)
    token = await login({"server_url": SERVER_URL})
    server = await connect_to_server(
        {"name": "hypha bot", "server_url": SERVER_URL, "token": token}
    )
    # write token into .hypha-token file and save the current time as json file
    with open(".hypha-token", "w") as f:
        f.write(json.dumps({"token": token, "time": time.time()}))

    print("Connected to server.", flush=True)

    service_id = "image-analysis-bot"

    await server.register_service({
        "name": "Hypha bot",
        "description": "A hypha bot for code interpreter services",
        "id": service_id,
        "icon": "https://imjoy.io/static/img/imjoy-icon.png",
        "type": "hypha-bot",
        "config": {
            "visibility": "public",
            "require_context": True,
            "run_in_executor": True,
        },
        "chat": chat,
        "ping": lambda context: "pong",
    }, overwrite=True)
    svc = await server.get_service(service_id)
    url = f"http://localhost:3090/#/chat/new?service-id={svc.id}"
    print(f"Hypha bot is ready!\nYou can connect via {url}")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()