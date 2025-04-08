# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from schema_agents import (
    AgentError,
    AgentImage,
    CodeAgent,
    ToolCallingAgent,
    stream_to_gradio,
)
from schema_agents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallDefinition,
)
from schema_agents.monitoring import AgentLogger, LogLevel


class FakeLLMModel:
    def __init__(self):
        self.last_input_token_count = 10
        self.last_output_token_count = 20

    async def __call__(self, prompt, tools_to_call_from=None, **kwargs):
        if tools_to_call_from is not None:
            return ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ChatMessageToolCall(
                        id="fake_id",
                        type="function",
                        function=ChatMessageToolCallDefinition(name="final_answer", arguments={"answer": "image"}),
                    )
                ],
            )
        return ChatMessage(
            role="assistant",
            content="""
Thought: I will return a final answer
Code:
```py
final_answer("Final answer")
```<end_code>
""",
        )


@pytest.mark.asyncio
async def test_code_agent_metrics():
    agent = CodeAgent(
        tools=[],
        model=FakeLLMModel(),
        max_steps=1,
    )
    await agent.run("Fake task")

    assert agent.monitor.total_input_token_count == 10
    assert agent.monitor.total_output_token_count == 20


@pytest.mark.asyncio
async def test_toolcalling_agent_metrics():
    agent = ToolCallingAgent(
        tools=[],
        model=FakeLLMModel(),
        max_steps=1,
    )

    await agent.run("Fake task")

    assert agent.monitor.total_input_token_count == 10
    assert agent.monitor.total_output_token_count == 20


@pytest.mark.asyncio
async def test_code_agent_metrics_max_steps():
    class FakeLLMModelMalformedAnswer:
        def __init__(self):
            self.last_input_token_count = 10
            self.last_output_token_count = 20

        async def __call__(self, prompt, **kwargs):
            return ChatMessage(role="assistant", content="Malformed answer")

    agent = CodeAgent(
        tools=[],
        model=FakeLLMModelMalformedAnswer(),
        max_steps=1,
    )

    await agent.run("Fake task")

    assert agent.monitor.total_input_token_count == 20
    assert agent.monitor.total_output_token_count == 40


@pytest.mark.asyncio
async def test_code_agent_metrics_generation_error():
    class FakeLLMModelGenerationException:
        def __init__(self):
            self.last_input_token_count = 10
            self.last_output_token_count = 20

        async def __call__(self, prompt, **kwargs):
            self.last_input_token_count = 10
            self.last_output_token_count = 0
            raise Exception("Cannot generate")

    agent = CodeAgent(
        tools=[],
        model=FakeLLMModelGenerationException(),
        max_steps=1,
    )
    await agent.run("Fake task")

    assert agent.monitor.total_input_token_count == 20  # Should have done two monitoring callbacks
    assert agent.monitor.total_output_token_count == 0


@pytest.mark.asyncio
async def test_streaming_agent_text_output():
    agent = CodeAgent(
        tools=[],
        model=FakeLLMModel(),
        max_steps=1,
    )

    # Use stream_to_gradio to capture the output
    outputs = []
    async for message in stream_to_gradio(agent, task="Test task"):
        outputs.append(message)

    # Check that we have the expected final output
    assert len(outputs) > 0  # We should have at least one output
    
    # Find messages with "Final answer" in the content
    final_answer_messages = [msg for msg in outputs if "Final answer" in str(msg.content)]
    assert len(final_answer_messages) > 0  # We should have at least one message with "Final answer"


@pytest.mark.asyncio
async def test_streaming_agent_image_output():
    agent = ToolCallingAgent(
        tools=[],
        model=FakeLLMModel(),
        max_steps=1,
    )

    # Use stream_to_gradio to capture the output
    outputs = []
    async for message in stream_to_gradio(
        agent,
        task="Test task",
        additional_args=dict(image=AgentImage(value="path.png")),
    ):
        outputs.append(message)

    # Check that we have the expected final output
    assert len(outputs) > 0  # We should have at least one output
    assert hasattr(outputs[-1], 'type')  # The last message should have a type attribute
    assert outputs[-1].type == "final_answer"  # The last message should be a final answer
    assert outputs[-1].content == "image"  # The content should match


@pytest.mark.asyncio
async def test_streaming_with_agent_error():
    logger = AgentLogger(level=LogLevel.INFO)

    async def dummy_model(prompt, **kwargs):
        raise AgentError("Simulated agent error", logger)

    agent = CodeAgent(
        tools=[],
        model=dummy_model,
        max_steps=1,
    )

    # Use stream_to_gradio to capture the output
    outputs = []
    async for message in stream_to_gradio(agent, task="Test task"):
        outputs.append(message)

    # Check that we have the expected final output
    assert len(outputs) > 0  # We should have at least one output
    
    # Find messages with "Simulated agent error" in the content
    error_messages = [msg for msg in outputs if "Simulated agent error" in str(msg.content)]
    assert len(error_messages) > 0  # We should have at least one message with the error
