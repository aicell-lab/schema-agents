import ray
import asyncio
import argparse
from imjoy_rpc.hypha import connect_to_server
import cloudpickle

async def register_ray(server, ray_address=None):
    if ray_address:
        ray.init(address=ray_address)

    print(server.config)
    server.register_codec({
      "name": "RayObjectRef",
      "type": ray.ObjectRef,
      "encoder": lambda x: {"id": x.hex()},
      "decoder": lambda x: ray.ObjectRef(x["id"]),
    })

    def encode_ray_function(x):
        pass

    def decode_ray_function(x):
        return ray.remote(x)

    server.register_codec({
        "name": "RayRemoteFunction",
        "type": ray.remote_function.RemoteFunction,
        "encoder": encode_ray_function,
        "decoder": decode_ray_function,
    })

    async def remote(f):
        func = cloudpickle.loads(f)
        return ray.remote(func)

    await server.register_service(
        {
            "name": "Ray",
            "id": "ray",
            "config": {
                "visibility": "public",
                "require_context": False,
                "run_in_executor": True,
            },
            "put": lambda x: ray.put(x),
            "get": lambda x: ray.get(x),
            "remote": remote,
        }
    )
    print("Ray service registered.")
    
async def test_ray(server):
    svc = await server.get_service("ray")
    f = cloudpickle.dumps(lambda x: x + 1)
    func = await svc.remote(f)
    print(await func(1))
    print("Ray service is working.")

async def start_function_launcher(server_url, workspace, token, ray_address):
    server = await connect_to_server(
        {
            "name": "function client",
            "server_url": server_url,
            "token": token,
            "workspace": workspace,
        }
    )
    
    print("Registering Ray service...")
    await register_ray(server, ray_address=ray_address)
    
    await test_ray(server)
    
    print("Ray service is ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="https://ai.imjoy.io")
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--ray-server", default="auto")
    opts = parser.parse_args()

    loop = asyncio.get_event_loop()
    task = loop.create_task(
        start_function_launcher(
            ray_address=opts.ray_server,
            server_url=opts.server_url,
            token=opts.token,
            workspace=opts.workspace,
        )
    )
    loop.run_forever()