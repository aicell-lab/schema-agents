"""Provide common pytest fixtures."""
import os
import subprocess
import sys
import time
import uuid
import pytest
import pytest_asyncio
import requests
import json
from pydantic_ai.models.openai import OpenAIModel
import asyncio
from fastapi import FastAPI, Request, HTTPException
from hypha_rpc.utils.serve import create_openai_chat_server
import uvicorn
import httpx
import socket
from fastapi.responses import StreamingResponse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
from pydantic_ai import RunContext, models, result
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolReturnPart, ToolCallPart
from pydantic_ai.usage import Usage

# Test server configuration
WS_PORT = 9573
OPENAI_PORT = 9528
JWT_SECRET = str(uuid.uuid4())
os.environ["JWT_SECRET"] = JWT_SECRET
test_env = os.environ.copy()

def find_free_port():
    """Find a free port to use for the mock server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

async def mock_chat_generator(body: dict):
    messages = body.get("messages", [])
    tools = body.get("tools", [])
    
    # Check if this is a ReAct flow with calculation and Wikipedia tools
    if tools and any("calculate" in str(tool) for tool in tools):
        # Check if this is a follow-up after calculation
        if any("Previous observation: 1200" in str(msg.get("content")) for msg in messages):
            # Return Wikipedia search tool call
            yield "data: " + json.dumps({
                "id": "mock-completion-id",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": body.get("model", "gpt-3.5-turbo"),
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "search_wikipedia",
                                    "arguments": json.dumps({
                                        "query": "1200"
                                    })
                                }
                            }]
                        }
                    }
                ]
            }) + "\n\n"
        # Check if this is a follow-up after Wikipedia search
        elif any("Wikipedia results for: 1200" in str(msg.get("content")) for msg in messages):
            # Return final response
            yield "data: " + json.dumps({
                "id": "mock-completion-id",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": body.get("model", "gpt-3.5-turbo"),
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": "Final Response: The calculation of 25 * 48 equals 1200. According to Wikipedia, 1200 is a significant number with various historical and mathematical properties."
                        }
                    }
                ]
            }) + "\n\n"
        else:
            # Initial calculation request
            yield "data: " + json.dumps({
                "id": "mock-completion-id",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": body.get("model", "gpt-3.5-turbo"),
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "calculate",
                                    "arguments": json.dumps({
                                        "expression": "25 * 48"
                                    })
                                }
                            }]
                        }
                    }
                ]
            }) + "\n\n"
    else:
        # Initial response with model info
        yield "data: " + json.dumps({
            "id": "mock-completion-id",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": body.get("model", "gpt-3.5-turbo"),
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant"
                    }
                }
            ]
        }) + "\n\n"

        # Content response
        yield "data: " + json.dumps({
            "id": "mock-completion-id",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": body.get("model", "gpt-3.5-turbo"),
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "The calculation of 25 * 48 equals 1200. According to Wikipedia, 1200 is a significant number with various historical and mathematical properties."
                    }
                }
            ]
        }) + "\n\n"

    # Final response with finish reason
    yield "data: " + json.dumps({
        "id": "mock-completion-id",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": body.get("model", "gpt-3.5-turbo"),
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }) + "\n\n"

    # End of stream
    yield "data: [DONE]\n\n"

@pytest_asyncio.fixture
async def mock_openai_server():
    app = FastAPI()
    
    # Add routes
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        print(f"Request path: {request.url.path}")
        print(f"Request method: {request.method}")
        print(f"Request headers: {request.headers}")
        body = await request.json()
        print(f"Request body: {body}")
        
        # Check API key
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        # Check if streaming is requested
        stream = body.get("stream", False)
        
        # Check if function call is required
        tool_choice = body.get("tool_choice")
        tools = body.get("tools", [])
        messages = body.get("messages", [])
        
        # Check for tool returns in messages
        has_tool_return = any("function_call" in str(msg.get("content")) or "tool_call" in str(msg.get("content")) for msg in messages)
        
        # Handle ReAct reasoning
        if tools and any("calculate" in str(tool) for tool in tools):
            # Check if this is a follow-up after Wikipedia search
            if any("Wikipedia results for: 1200" in str(msg.get("content")) for msg in messages):
                # Return final response
                response = {
                    "id": "mock-completion-id",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": body.get("model", "gpt-3.5-turbo"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Final Response: The calculation of 25 * 48 equals 1200. According to Wikipedia, 1200 is a significant number with various historical and mathematical properties."
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
                }
                return response
            # Check if this is a follow-up after calculation
            elif any("Previous observation: 1200" in str(msg.get("content")) for msg in messages):
                # Return Wikipedia search
                response = {
                    "id": "mock-completion-id",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": body.get("model", "gpt-3.5-turbo"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "search_wikipedia",
                                        "arguments": json.dumps({
                                            "query": "1200"
                                        })
                                    }
                                }]
                            },
                            "finish_reason": "tool_call"
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
                }
                return response
            else:
                # Initial calculation request
                response = {
                    "id": "mock-completion-id",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": body.get("model", "gpt-3.5-turbo"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "calculate",
                                        "arguments": json.dumps({
                                            "expression": "25 * 48"
                                        })
                                    }
                                }]
                            },
                            "finish_reason": "tool_call"
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
                }
                return response
        elif tool_choice == "required" and tools:
            # Handle structured output
            if has_tool_return:
                # Return final response after tool call
                response = {
                    "id": "mock-completion-id",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": body.get("model", "gpt-3.5-turbo"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps({
                                    "message": "This is a mock analysis",
                                    "confidence": 0.9,
                                    "tags": ["mock", "test"],
                                    "metadata": {"source": "mock"}
                                })
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
                }
                return response
            else:
                # Return a function call for structured output
                tool = tools[0]["function"]
                response = {
                    "id": "mock-completion-id",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": body.get("model", "gpt-3.5-turbo"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "final_result",
                                        "arguments": json.dumps({
                                            "message": "This is a mock analysis",
                                            "confidence": 0.9,
                                            "tags": ["mock", "test"],
                                            "metadata": {"source": "mock"}
                                        })
                                    }
                                }]
                            },
                            "finish_reason": "tool_call"
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
                }
                return response
        
        if stream:
            return StreamingResponse(mock_chat_generator(body), media_type="text/event-stream")
        
        # Handle memory test
        if any("Remember the number 42" in str(msg.get("content")) for msg in messages):
            response = {
                "id": "mock-completion-id",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "gpt-3.5-turbo"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "I will remember the number 42."
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            }
            return response
        elif any("What number did I ask you to remember?" in str(msg.get("content")) for msg in messages):
            response = {
                "id": "mock-completion-id",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "gpt-3.5-turbo"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "You asked me to remember the number 42."
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            }
            return response
        
        # Default response
        response = {
            "id": "mock-completion-id",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "gpt-3.5-turbo"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a mock response"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        }
        return response

    @app.get("/models")
    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": "gpt-3.5-turbo",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "openai"
                }
            ]
        }
    
    # Find a free port
    port = find_free_port()
    
    config = uvicorn.Config(app, host="127.0.0.1", port=port)
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    
    # Wait for server to start
    base_url = f"http://127.0.0.1:{port}"
    max_retries = 30
    retry_interval = 1.0
    
    for _ in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}/v1/models")
                if response.status_code == 200:  # We now expect 200 since we implemented the endpoint
                    break
        except Exception:
            await asyncio.sleep(retry_interval)
    else:
        raise Exception("Failed to start mock server")

    await asyncio.sleep(1)  # Additional wait to ensure server is ready
    
    try:
        return base_url + "/v1"  # Return the string URL directly instead of yielding it
    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

class TestOpenAIModel(OpenAIModel):
    """Test OpenAI model implementation."""
    
    def __init__(self, base_url: str = None, api_key: str = "test-key", model_name: str = "test-chat-model"):
        super().__init__(base_url=base_url, api_key=api_key)
        self._model_name = model_name
        self._step = 0
    
    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name
    
    @property
    def system(self) -> str:
        """Return the system prompt."""
        return "You are a helpful test assistant."
    
    async def request(
        self,
        message_history: List[ModelMessage],
        model_settings: Dict[str, Any] | None = None,
        request_parameters: Dict[str, Any] | None = None,
    ) -> Tuple[ModelResponse, Usage]:
        """Process a request and return a mock response."""
        self._step += 1
        
        # Get the last request
        last_message = message_history[-1]
        if not isinstance(last_message, ModelRequest):
            return ModelResponse(
                parts=[TextPart(content="I don't understand")],
                model_name=self.model_name
            ), Usage()
        
        content = last_message.parts[0].content
        
        # Handle tool-based interactions
        if "calculate" in content.lower():
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_calls=[{
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "calculate",
                                "arguments": json.dumps({"expression": "2 + 2"})
                            }
                        }]
                    )
                ],
                model_name=self.model_name
            ), Usage()
        elif "search" in content.lower() or "wikipedia" in content.lower():
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_calls=[{
                            "id": "call-2",
                            "type": "function",
                            "function": {
                                "name": "search_wikipedia",
                                "arguments": json.dumps({"query": "test query"})
                            }
                        }]
                    )
                ],
                model_name=self.model_name
            ), Usage()
        elif "tool return" in content.lower():
            return ModelResponse(
                parts=[
                    ToolReturnPart(
                        tool_returns=[{
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "calculate",
                                "arguments": json.dumps({"result": 4})
                            }
                        }]
                    )
                ],
                model_name=self.model_name
            ), Usage()
        
        # Default text response
        return ModelResponse(
            parts=[TextPart(content=f"Test response {self._step}")],
            model_name=self.model_name
        ), Usage()

@dataclass
class TestDependencies:
    """Test dependencies for tracking tool usage."""
    memory_entries: List[Dict[str, Any]] = None
    tool_calls: List[Dict[str, Any]] = None
    
    def __init__(self):
        self.memory_entries = []
        self.tool_calls = []
        
    def record_memory(self, entry: Dict[str, Any]):
        """Record memory entry."""
        self.memory_entries.append(entry)
        
    def record_tool_call(self, tool: str, args: Dict[str, Any]):
        """Record tool call."""
        self.tool_calls.append({
            "tool": tool,
            "args": args
        })

@pytest.fixture(scope="session")
def openai_model():
    """Create a mock OpenAI model for testing."""
    return OpenAIModel(
        'gpt-4o-mini',
        api_key=os.getenv('OPENAI_API_KEY')
    )
@pytest.fixture(scope="session")
def hypha_server_url():
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
        except requests.RequestException:
            pass
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
    
    yield f"http://127.0.0.1:{WS_PORT}"
    
    # Cleanup
    proc.kill()
    proc.terminate()
    proc.wait(timeout=5)  # Wait for process to terminate
