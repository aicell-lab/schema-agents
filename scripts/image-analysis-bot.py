import json
import time
import os
import uuid
import asyncio
from functools import partial
from imjoy_rpc.hypha import connect_to_server, login
from schema_agents.utils import dict_to_md
from schema_agents.config import CONFIG
from schema_agents.teams.image_analysis_hub import ImageAnalysisHub
import logging
from schema_agents.tools.code_interpreter import create_mock_client
logging.basicConfig(level=logging.DEBUG)

SERVER_URL = "https://ai.imjoy.io"

async def chat(msg, client, context=None):
    """Hypha bot chat to help you with various tasks"""
    if context["user"]['id'] != "github|478667":
        raise Exception(f"Sorry, you (user: {context['user']}) are not authorized to use this service.")
    conversation_id = str(uuid.uuid4())
    if client:
        await client.initialize({"conversationId": conversation_id })
        await client.showMessage("hypha bot joined the workspace")
        # await client.executeScript({"script": query})
        # dialog = await client.showDialog(src="https://kaibu.org/#/app")
        # await dialog.view_image("https://images.proteinatlas.org/61448/1319_C10_2_blue_red_green.jpg")
        await client.newMessage({
            "messageId": str(uuid.uuid4()),
            "parentMessageId": msg.messageId,
            "sender": "ChatGPT",
            "text": "",
            "submitting": False,
        })

    async def message_callback(message):
        if message.instruct_content:
            content = dict_to_md(message.instruct_content.dict())
        else:
            content = message.content
        text = f"# 🧑{message.role}\n\n{content}\n **💰{CONFIG.total_cost:.4f} / {CONFIG.max_budget:.4f}**\n"
        if client:
            await client.appendText({"messageId": msg.messageId, "text": text})
    
    hub = ImageAnalysisHub()
    hub.invest(0.5)
    hub.recruit(client)
    hub.start(msg.text, message_callback=message_callback)
    await hub.run(n_round=10)

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
  
    await chat("acquire an image every 1 second for 3.5 seconds", client, {"user": {"id": "github|478667"}})

async def main():
    client_id = str(uuid.uuid3(uuid.NAMESPACE_DNS, f'urn:node:{hex(uuid.getnode())}'))
    # token = login({"server_url": SERVER_URL})
    print("Connecting to server...", flush=True)
    # try to load token from .hypha-token file
    try:
        with open(".hypha-token", "r") as f:
            token_info = json.loads(f.read())
            token = token_info["token"]
        # check if the token is expired
        if time.time() - token_info["time"] > 3600 * 12:
            raise Exception("Token expired")
        try:
            server = await connect_to_server(
                {"name": "hypha bot", "server_url": SERVER_URL, "client_id": client_id, "token": token, "method_timeout": 60, "sync_max_workers": 2, "webrtc": True}
            )
        except PermissionError:
            raise Exception("Failed to login expired")
    except Exception:
        if os.path.exists(".hypha-token"):
            os.remove(".hypha-token")
        token = await login({"server_url": SERVER_URL})
        server = await connect_to_server(
            {"name": "hypha bot", "client_id": client_id, "server_url": SERVER_URL, "token": token, "method_timeout": 60, "sync_max_workers": 2, "webrtc": True}
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
    url = f"https://chat.aicell.io/#/chat/new?service-id={server.config['workspace']}/*:{service_id}"
    print(f"Hypha bot is ready!\nYou can connect via {url}")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()