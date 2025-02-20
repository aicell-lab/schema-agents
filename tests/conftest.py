"""Provide common pytest fixtures."""
import os
import subprocess
import sys
import time
import uuid
import pytest
import pytest_asyncio
import requests
from requests import RequestException
from pydantic_ai.models.openai import OpenAIModel
from hypha_rpc import connect_to_server
import asyncio

# Test server configuration
WS_PORT = 9527
JWT_SECRET = str(uuid.uuid4())
os.environ["JWT_SECRET"] = JWT_SECRET
test_env = os.environ.copy()

@pytest.fixture(scope="session")
def openai_model():
    """Create a mock OpenAI model for testing."""
    return OpenAIModel(
        'gpt-4o-mini',
        api_key=os.getenv('OPENAI_API_KEY', 'test-key')
    )

@pytest_asyncio.fixture(scope="session")
async def hypha_server_process():
    """Start and manage the Hypha server process."""
    # Start server process
    proc = subprocess.Popen(
        [sys.executable, "-m", "hypha.server", f"--port={WS_PORT}"],
        env=test_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    timeout = 30  # Increase timeout to 30 seconds
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://127.0.0.1:{WS_PORT}/health/liveness")
            if response.ok:
                break
        except RequestException:
            # Check if process is still running
            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                raise RuntimeError(
                    f"Server process exited with code {proc.returncode}.\n"
                    f"stdout: {stdout.decode()}\n"
                    f"stderr: {stderr.decode()}"
                )
        time.sleep(0.5)  # Longer sleep to reduce CPU usage
    else:
        proc.kill()
        proc.terminate()
        stdout, stderr = proc.communicate()
        raise RuntimeError(
            f"Server failed to start within {timeout} seconds.\n"
            f"stdout: {stdout.decode()}\n"
            f"stderr: {stderr.decode()}"
        )
    
    yield proc
    
    # Cleanup
    proc.kill()
    proc.terminate()
    proc.wait(timeout=5)  # Wait for process to terminate

@pytest_asyncio.fixture
async def hypha_server(hypha_server_process):
    """Fixture that provides a connected Hypha server instance."""
    # Try to connect multiple times
    max_retries = 3
    retry_delay = 1
    last_error = None
    
    for i in range(max_retries):
        try:
            server = await connect_to_server({"server_url": f"ws://127.0.0.1:{WS_PORT}/ws"})
            try:
                yield server
            finally:
                await server.disconnect()
            return
        except Exception as e:
            last_error = e
            if i < max_retries - 1:
                await asyncio.sleep(retry_delay)
    
    raise RuntimeError(f"Failed to connect to server after {max_retries} attempts: {last_error}")

@pytest.fixture
def hypha_server_url():
    """Get the Hypha server URL."""
    return f"http://localhost:{WS_PORT}"
