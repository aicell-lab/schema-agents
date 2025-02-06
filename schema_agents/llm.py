import random
import string
from typing import List, Dict, Any
from schema_agents.utils.common import EventBus, current_session
from schema_agents.schema import StreamEvent
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import (
    ChatCompletion,
    Choice,
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from schema_agents.role import create_session_context
from schema_agents.schema import StreamEvent
from contextvars import copy_context


def generate_query_id() -> str:
    """Generate a random query ID."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


async def _process_stream_response(response):
    """
    Process the stream response from OpenAI to handle tool calls and text responses,
    and return a complete `ChatCompletion` object.
    """
    accumulated_content = ""
    tool_calls = {}
    query_id = generate_query_id()
    session = current_session.get() if current_session in copy_context() else None
    event_bus = session.event_bus if session else None
    choices = []

    first_choice = None
    first_chunk = None
    usage_chunk = None

    if isinstance(response, ChatCompletion):
        return response  # Return directly if already a complete response

    async for chunk in response:
        if session and session.stop:
            raise InterruptedError("Session interrupted")
        if len(chunk.choices) == 0 and chunk.usage:
            # usage chunk
            usage_chunk = chunk
            continue

        choice = chunk.choices[0]
        chunk_message = choice.delta
        # Update metadata based on the first chunk
        if first_choice is None:
            first_choice = choice
            first_chunk = chunk

        # Handle tool calls
        if chunk_message.tool_calls:
            for _tool_call in chunk_message.tool_calls:
                if _tool_call.id is None and _tool_call.index is not None:
                    # find id by index in tool_calls
                    for tool_call in tool_calls.values():
                        if tool_call.index == _tool_call.index:
                            _tool_call.id = tool_call.id
                            break
                if _tool_call.id is not None:
                    func_call = _tool_call.function
                    # Accumulate tool call arguments if the same function is continued
                    if _tool_call.id not in tool_calls:
                        _tool_call.index = len(tool_calls)
                        tool_calls[_tool_call.id] = _tool_call
                        # Emit 'start' event for new function calls
                        if event_bus:
                            event_bus.emit(
                                "stream",
                                StreamEvent(
                                    type="function_call",
                                    query_id=query_id,
                                    session=session,
                                    name=func_call.name,
                                    arguments=func_call.arguments or "",
                                    status="start",
                                ),
                            )
                    else:
                        existing_func_call = tool_calls[_tool_call.id].function
                        existing_func_call.arguments += func_call.arguments or ""

                        # Emit 'in_progress' event for continuing function calls
                        if event_bus:
                            event_bus.emit(
                                "stream",
                                StreamEvent(
                                    type="function_call",
                                    query_id=query_id,
                                    session=session,
                                    name=existing_func_call.name,
                                    arguments=func_call.arguments or "",
                                    status="in_progress",
                                ),
                            )

        # Handle text content
        if chunk_message.content:
            accumulated_content += chunk_message.content
            if event_bus:
                event_bus.emit(
                    "stream",
                    StreamEvent(
                        type="text",
                        query_id=query_id,
                        content=chunk_message.content,
                        status="progress",
                    ),
                )

    # Finalize events for tool calls
    if tool_calls:
        if event_bus:
            for tool_call in tool_calls.values():
                event_bus.emit(
                    "stream",
                    StreamEvent(
                        type="function_call",
                        query_id=query_id,
                        session=session,
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                        status="finished",
                    ),
                )
        # Finalize chunk into a choice object
        choices.append(
            Choice(
                finish_reason=choice.finish_reason,
                index=choice.index,
                logprobs=choice.logprobs,
                message=ChatCompletionMessage(
                    content=accumulated_content,
                    role=first_choice.delta.role,
                    function_call=None,  # Function calls are aggregated below
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id=tool_call.id,
                            function=Function(
                                name=tool_call.function.name,
                                arguments=tool_call.function.arguments,
                            ),
                            type=tool_call.type,
                        )
                        for tool_call in tool_calls.values()
                    ],
                    refusal=first_choice.delta.refusal,
                ),
            )
        )
    else:
        # Finalize chunk into a choice object
        choices.append(
            Choice(
                finish_reason=choice.finish_reason,
                index=choice.index,
                logprobs=choice.logprobs,
                message=ChatCompletionMessage(
                    content=accumulated_content,
                    role=first_choice.delta.role,
                    function_call=None,
                    tool_calls=None,
                    refusal=first_choice.delta.refusal,
                ),
            )
        )

    # Build the complete ChatCompletion object
    response = ChatCompletion(
        id=first_chunk.id,
        choices=choices,
        created=first_chunk.created,
        model=first_chunk.model,
        object="chat.completion",
        system_fingerprint=first_chunk.system_fingerprint,
        usage=usage_chunk.usage if usage_chunk else None,
    )

    return response


async def chat_completion(
    messages: List[Dict[str, Any]],
    client=None,
    model="gpt-4o-mini",
    stream=True,
    client_config=None,
    **kwargs,
) -> ChatCompletion:
    """
    Main entry point to interact with OpenAI's chat completion with tool support.
    """
    client_config = client_config or {}
    client = client or AsyncOpenAI(**client_config)
    try:
        response = await client.chat.completions.create(
            messages=messages,
            stream=stream,
            model=model,
            stream_options={"include_usage": True} if stream else None,
            **kwargs,
        )

        return await _process_stream_response(response)

    except Exception as e:
        raise e


# Example Usage

async def _main():
    event_bus = EventBus("test")
    event_bus.register_default_events()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What’s the weather in New York?"},
    ]

    tools = [
        dict(
            type="function",
            function={
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {
                            "type": "string",
                            "enum": ["c", "f"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["location", "unit"],
                    "additionalProperties": False,
                },
            },
        )
    ]

    async with create_session_context(
        event_bus=event_bus
    ):
        response = await chat_completion(
            messages,
            tools=tools,
            tool_choice="required",
            stream=False,
            model="gpt-4o-2024-11-20",
            temperature=0.7,
            max_tokens=150,
        )
        print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(_main())
