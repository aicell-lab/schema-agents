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
logging.basicConfig(level=logging.DEBUG)

SERVER_URL = "https://ai.imjoy.io"

accumulated_text = ""


async def chat(query, client, context=None):
    """Hypha bot chat to help you with various tasks"""
    if context["user"]['id'] != "github|478667":
        raise Exception(f"Sorry, you (user: {context['user']}) are not authorized to use this service.")
    conversation_id = str(uuid.uuid4())
    if client:
        await client.init(conversation_id)
        await client.show_message("hypha bot joined the workspace")
        # await client.execute_code(query)
        # dialog = await client.show_dialog(src="https://kaibu.org/#/app")
        # await dialog.view_image("https://images.proteinatlas.org/61448/1319_C10_2_blue_red_green.jpg")
        await client.set_output("", None)
    async def message_callback(message):
        global accumulated_text
        if message.instruct_content:
            content = dict_to_md(message.instruct_content.dict())
        else:
            content = message.content
        text = f"# 🧑{message.role}\n\n{content}\n **💰{CONFIG.total_cost:.4f} / {CONFIG.max_budget:.4f}**\n"
        accumulated_text += text
        if client:
            await client.stream_output(text)
    
    hub = ImageAnalysisHub()
    hub.invest(0.5)
    hub.recruit(client)
    hub.start(query, message_callback=message_callback)
    await hub.run(n_round=10)
    return {'text': accumulated_text, 'plugin': None, 'conversation_id': conversation_id}

async def test_chat():
    from schema_agents.tools.code_interpreter import create_mock_client
    client = create_mock_client()
    await chat("create a tool for counting cells in microscopy images", client, {"user": {"id": "github|478667"}})

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

    await server.register_service({
        "name": "Hypha bot",
        "description": "A hypha bot for code interpreter services",
        "id": "image-analysis-bot",
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

    print("Hypha bot is ready!")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()