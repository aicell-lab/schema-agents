import aiohttp
import asyncio
from aiohttp_sse_client import client as sse_client
import json
import uuid
import os
from jhub_client.api import JupyterKernelAPI, JupyterAPI

# Example code for jhub_client: https://github.com/Quansight/jhub-client/tree/master/tests

class KernelAPI(JupyterKernelAPI):
    async def execute_code(self, username, code, wait=True, timeout=None):
        msg_id = str(uuid.uuid4())

        await self.websocket.send_json(
            self.request_execute_code(msg_id, username, code)
        )

        if not wait:
            return None

        async for msg_text in self.websocket:
            if msg_text.type != aiohttp.WSMsgType.TEXT:
                return False

            # TODO: timeout is ignored

            msg = msg_text.json()

            if "parent_header" in msg and msg["parent_header"].get("msg_id") == msg_id:
                # These are responses to our request
                if msg["channel"] == "iopub":
                    if msg["msg_type"] == "execute_result":
                        return msg["content"]["data"]["text/plain"]
                    elif msg["msg_type"] == "stream":
                        return msg["content"]["text"]
                    # cell did not produce output
                    elif msg["content"].get("execution_state") == "idle":
                        return ""

class BinderServer():
    def __init__(self, repo='imjoy-team/imjoy-binder-image', branch='master', config=None):
        self.repo = repo
        self.branch = branch
        self.jhub_config = config

    async def start(self, load_existing=True):
        if load_existing:
            if os.path.exists('.jhub_config.json'):
                try:
                    with open('.jhub_config.json', 'r') as f:
                        jhub_config = json.load(f)
                    jupyter = JupyterAPI(jhub_config['url'], jhub_config['token'])
                    async with jupyter:
                        specs = await jupyter.list_kernel_specs()
                        if len(specs) > 0:
                            self.jhub_config = jhub_config
                            print("Existing Binder server found.")
                            return self.jhub_config
                except Exception as e:
                    print("Failed to reuse existing Binder server config.")

        binder_api_url = f"https://notebooks.gesis.org/binder/build/gh/{self.repo}/{self.branch}"
        # Start Binder build
        async with aiohttp.ClientSession() as session:
            async with session.post(binder_api_url) as response:
                if response.status == 200:
                    print("Binder build starting...")
                else:
                    print(f"Failed to start Binder build: {response.status}")
        
        # Listen to SSE events for build status
        async with sse_client.EventSource(binder_api_url) as event_source:
            jhub_config = None
            async for event in event_source:
                data = json.loads(event.data)
                if data['phase'] == 'ready':
                    # example: MessageEvent(type=None, message='', data='{"phase": "ready", "message": "server running at https://notebooks.gesis.org/binder/jupyter/user/imjoy-team-imjoy-binder-image-9fc2zw8r/\\n", "image": "gesiscss/binder-r2d-g5b5b759-imjoy-2dteam-2dimjoy-2dbinder-2dimage-560b33:83282e916f96e5cc28e76775965c1601331ace18", "repo_url": "https://github.com/imjoy-team/imjoy-binder-image", "token": "vXFZ0kV6S3auR2nbUCYXbA", "binder_ref_url": "https://github.com/imjoy-team/imjoy-binder-image/tree/83282e916f96e5cc28e76775965c1601331ace18", "binder_launch_host": "https://mybinder.org/", "binder_request": "v2/gh/imjoy-team/imjoy-binder-image/master", "binder_persistent_request": "v2/gh/imjoy-team/imjoy-binder-image/83282e916f96e5cc28e76775965c1601331ace18", "url": "https://notebooks.gesis.org/binder/jupyter/user/imjoy-team-imjoy-binder-image-9fc2zw8r/"}', origin='https://notebooks.gesis.org', last_event_id='')
                    # Usage: jupyter = JupyterAPI(hub.hub_url / "user" / username, hub.api_token)
                    self.username = self.jhub_config['url'].split('/')[-2]
                    jhub_config = data
                    print("Binder build finished.", data['message'])
                    break
                if data['phase'] == 'failed':
                    print("Binder build failed.", data['message'])
                    break
                else:
                    print(f"Binder {data['phase']}: {data['message']}")
        if jhub_config is None:
            raise Exception(f"Failed to start Binder server: {data['message']}")
        with open('.jhub_config.json', 'w') as f:
            json.dump(self.jhub_config, f)
        return self.jhub_config

        
    async def start_kernel(self):
        jupyter = JupyterAPI(self.jhub_config['url'], self.jhub_config['token'])
        async with jupyter:
            kernel_id = (await jupyter.create_kernel())["id"]
            kernel = KernelAPI(
                jupyter.api_url / "kernels" / kernel_id, jupyter.api_token
            )
            return kernel
    
    async def stop_kernel(self, kernel_id):
        jupyter = JupyterAPI(self.jhub_config['url'], self.jhub_config['token'])
        async with jupyter:
            await jupyter.delete_kernel(kernel_id)

    async def execute_code(self, kernel, code):
        async with kernel:
            return await kernel.execute_code(
                self.username,
                code
            )

async def test_binder():
    binder = BinderServer()
    await binder.start()
    kernel = await binder.start_kernel()
    print(await binder.execute_code(kernel, "print('hello world')"))
    await binder.stop_kernel(kernel.kernel_id)
    
if __name__ == '__main__':
    asyncio.run(test_binder())
