#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/5 23:08
@Author  : alexanderwu
@File    : openai.py
"""
import asyncio
import time
import random
import string
import os
import json
from functools import wraps
import textwrap
from typing import NamedTuple, Union, List, Dict, Any

import httpx
import openai
from openai import AsyncOpenAI, OpenAI

from schema_agents.config import CONFIG
from schema_agents.logs import logger
from schema_agents.provider.base_gpt_api import BaseGPTAPI
from schema_agents.utils.singleton import Singleton
from schema_agents.utils.token_counter import (
    TOKEN_COSTS,
    count_message_tokens,
    count_string_tokens,
    get_max_completion_tokens,
)
from schema_agents.utils.common import EventBus
from schema_agents.provider.openai_api import RateLimiter
from contextvars import copy_context
import google.generativeai as genai
from google.ai import generativelanguage as glm
from google.generativeai.generative_models import GenerativeModel
from google.generativeai.types import content_types
from google.generativeai.types.generation_types import (
    AsyncGenerateContentResponse,
    BlockedPromptException,
    GenerateContentResponse,
    GenerationConfig,
)



def create_schema(entity):
    """
    Recursive function to create a glm.Schema object from a dictionary describing an entity.
    Handles nested objects and arrays.
    """
    # Handle array types with nested items
    if entity.get('type') == 'array':
        item_schema = create_schema(entity['items'])
        return glm.Schema(type=glm.Type.ARRAY, items=item_schema)

    # Handle object types with properties
    elif entity.get('type') == 'object':
        properties = {key: create_schema(value) for key, value in entity['properties'].items()}
        required_fields = entity.get('required', [])
        return glm.Schema(type=glm.Type.OBJECT, properties=properties, required=required_fields)

    # Handle basic types with potential default values
    else:
        schema = glm.Schema(type=getattr(glm.Type, entity['type'].upper()))
        # if 'default' in entity:
        #     schema.default = entity['default']
        return schema

def json_schema_to_glm_function(schema_dict):
    """
    Converts a JSON schema dictionary to a glm.FunctionDeclaration.
    """
    function_name = schema_dict['name']
    function_description = schema_dict['description']
    # The parameters are nested under "parameters" key
    parameters_schema = create_schema(schema_dict['parameters'])

    function_declaration = glm.FunctionDeclaration(
        name=function_name,
        description=textwrap.dedent(function_description),
        parameters=parameters_schema
    )

    return function_declaration


class GeminiGenerativeModel(GenerativeModel):
    """
    Due to `https://github.com/google/generative-ai-python/pull/123`, inherit a new class.
    Will use default GenerativeModel if it fixed.
    """

    def count_tokens(self, contents: content_types.ContentsType) -> glm.CountTokensResponse:
        contents = content_types.to_contents(contents)
        return self._client.count_tokens(model=self.model_name, contents=contents)

    async def count_tokens_async(self, contents: content_types.ContentsType) -> glm.CountTokensResponse:
        contents = content_types.to_contents(contents)
        return await self._async_client.count_tokens(model=self.model_name, contents=contents)


class GeminiAPI(BaseGPTAPI, RateLimiter):
    """
    Gemini API client.
    """
    def __init__(self, model=None, seed=None, temperature=None, timeout=None, rpm=None):
        self.__init_gemini(CONFIG)
        self.model = model or CONFIG.gemini_api_model
        self.auto_max_tokens = False
        self.llm = GeminiGenerativeModel(model_name=self.model)
        self.rpm = rpm or CONFIG.gemini_api_rpm
        RateLimiter.__init__(self, rpm=self.rpm)
        logger.info(f"Gemini API model: {self.model}")

    def __init_gemini(self, config):
        if config.gemini_proxy:
            logger.info(f"Use proxy: {config.gemini_proxy}")
            os.environ["http_proxy"] = config.gemini_proxy
            os.environ["https_proxy"] = config.gemini_proxy
        genai.configure(api_key=config.gemini_api_key)

    def _const_kwargs(self, messages: list[dict], tools: list, stream: bool = False) -> dict:
        kwargs = {"contents": messages, "generation_config": GenerationConfig(temperature=0.3), "tools": tools, 
                  "stream": stream}
        return kwargs

    def format_msg(self, messages: Union[str, list[dict], list[str]]) -> list[dict]:
        """convert messages to list[dict]."""

        if not isinstance(messages, list):
            messages = [messages]

        # REF: https://ai.google.dev/tutorials/python_quickstart
        # As a dictionary, the message requires `role` and `parts` keys.
        # The role in a conversation can either be the `user`, which provides the prompts,
        # or `model`, which provides the responses.
        processed_messages = []
        for msg in messages:
            if isinstance(msg, str):
                processed_messages.append({"role": "user", "parts": [msg]})
            elif isinstance(msg, dict):
                role = msg.get("role")
                if role == "system":
                    role = "user"
                if "parts" not in msg.keys():
                    content = msg.get("content")
                    msg = {"role": role, "parts": [content]}
                assert set(msg.keys()) == set(["role", "parts"])
                processed_messages.append(msg)
            else:
                raise ValueError(
                    f"Only support message type are: str, Message, dict, but got {type(messages).__name__}!"
                )
        # combine the parts if two consecutive messages have the same role
        # for example, {"role": "user", "parts": ["hello"]}, {"role": "user", "parts": ["world"]} should be combined
        # as {"role": "user", "parts": ["hello", "world"]}
        combined_messages = []
        for msg in processed_messages:
            if combined_messages and combined_messages[-1]["role"] == msg["role"]:
                combined_messages[-1]["parts"].extend(msg["parts"])
            else:
                combined_messages.append(msg)
        
        return combined_messages

    async def aget_usage(self, messages: list[dict], resp_text: str) -> dict:
        req_text = messages[-1]["parts"][0] if messages else ""
        prompt_resp = await self.llm.count_tokens_async(contents={"role": "user", "parts": [{"text": req_text}]})
        completion_resp = await self.llm.count_tokens_async(contents={"role": "model", "parts": [{"text": resp_text}]})
        usage = {"prompt_tokens": prompt_resp.total_tokens, "completion_tokens": completion_resp.total_tokens}
        return usage

    def _update_costs(self, usage: dict):
        prompt_tokens = int(usage['prompt_tokens'])
        completion_tokens = int(usage['completion_tokens'])
        print(f"prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}")
        # self._cost_manager.update_cost(prompt_tokens, completion_tokens, self.model)
        
    async def _achat_completion_stream(self, messages: list[dict], functions: List[Dict[str, Any]]=None, function_call: Union[str, Dict[str, str]]=None, event_bus: EventBus=None):
        """Stream the completion of messages."""
        messages = self.format_msg(messages)
        func_call = {}
        function_call_detected = False
        tools=[]
        # Specify a function declaration and parameters for an API request
        for function in functions:
            function_declaration = json_schema_to_glm_function(function)
            tools.append(function_declaration)

        if function_call == 'auto':
            function_call = 'AUTO'
        elif function_call == 'none':
            function_call = 'NONE'
        elif function_call == 'required' or isinstance(function_call, dict):
            function_call = 'ANY'
        else:
            assert False, f"function_call must be 'auto', 'none', 'required' or dict, but got {function_call}"
            
        resp: AsyncGenerateContentResponse = await self.llm.generate_content_async(
            **self._const_kwargs(messages,tools=tools, stream=True), tool_config={'function_calling_config': function_call}
        )
        collected_content = []
        async for chunk in resp:
            try:
                if 'function_call' in chunk.candidates[0].content.parts[0]:
                    fc = chunk.candidates[0].content.parts[0].function_call
                    fc_content = type(fc).to_dict(fc)
                    function_call_detected = True
                    if "name" in fc_content and fc_content["name"]:
                        func_call["name"] = fc_content["name"]
                    if "args" in fc_content:
                        if "arguments" not in func_call:
                            func_call["arguments"] = ""
                        func_call["arguments"] += json.dumps(fc_content['args'])
                    content = json.dumps(type(fc).to_dict(fc), indent=4)
                elif 'text' in chunk.candidates[0].content.parts[0]:
                    content = chunk.text
            except Exception as e:
                logger.warning(f"messages: {messages}\nerrors: {e}\n{BlockedPromptException(str(chunk))}")
                raise BlockedPromptException(str(chunk))
            print(content)
            collected_content.append(content)
            print("\n")

        if function_call_detected:
            full_reply_content = {"type": "function_call", "function_call": func_call}
            # if chunk.system_fingerprint:
            #     full_reply_content['system_fingerprint'] = chunk.system_fingerprint
            # usage = self._calc_usage(messages, f"{func_call['name']}({func_call['arguments']})", functions=functions)
        # elif tool_call_detected:
        #     full_reply_content = {"type": "tool_calls", "tool_calls": [tool_calls[k] for k in sorted(list(tool_calls.keys()))]}
        #     if raw_chunk.system_fingerprint:
        #         full_reply_content['system_fingerprint'] = raw_chunk.system_fingerprint
        #     _msg = ""
        #     for _tool_call in tool_calls.values():
        #         func_call = _tool_call['function']
        #         _msg += f"{func_call['name']}({func_call['arguments']})\n"
        #     tools = kwargs["tools"]
        #     functions = [tool['function'] for tool in tools]
        #     usage = self._calc_usage(messages, _msg, functions=functions)
        else:
            full_reply_content = {"type": "text", "content": ''.join([m.get('content', '') for m in collected_content if m.get('content')])}
            # if chunk.system_fingerprint:
            #     full_reply_content['system_fingerprint'] = chunk.system_fingerprint
            # usage = self._calc_usage(messages, full_reply_content["content"])
            
        # full_content = "".join(collected_content)
        # usage = await self.aget_usage(messages, full_reply_content)
        # self._update_costs(usage)
        return full_reply_content

    def completion(self, messages: list[dict], event_bus: EventBus=None) -> dict:
        # if isinstance(messages[0], Message):
        #     messages = self.messages_to_dict(messages)
        rsp = self._chat_completion(messages, event_bus=event_bus)
        return rsp

    async def acompletion(self, messages: list[dict], event_bus: EventBus=None) -> dict:
        # if isinstance(messages[0], Message):
        #     messages = self.messages_to_dict(messages)
        rsp = await self._achat_completion_stream(messages, event_bus=event_bus)
        return rsp
    
    async def acompletion_function(self, messages: list[dict], functions: List[Dict[str, Any]]=None, function_call: Union[str, Dict[str, str]]=None, event_bus: EventBus=None, raise_for_string_output=False) -> dict:
        # if isinstance(messages[0], Message):
        #     messages = self.messages_to_dict(messages)
        rsp = await self._achat_completion_stream(messages, functions=functions, function_call=function_call, event_bus=event_bus)
        return rsp

    async def acompletion_text(self, messages: list[dict], stream=False, event_bus: EventBus=None) -> str:
        """when streaming, print each token in place."""
        if stream:
            return await self._achat_completion_stream(messages, event_bus=event_bus)
        rsp = await self._achat_completion(messages, event_bus=event_bus)
        return self.get_choice_text(rsp)
