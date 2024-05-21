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
from functools import wraps
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
from contextvars import copy_context
from schema_agents.provider.openai_api import RateLimiter, CostManager
import ollama


class OllamaAPI(BaseGPTAPI, RateLimiter):
    """
    Ollama API client.
    """
    def __init__(self, model=None, seed=None, temperature=None, timeout=None):
        self.__init_ollama(CONFIG)
        self.model = model or CONFIG.openai_api_model
        self.temperature = float(temperature or CONFIG.openai_temperature)
        self.timeout = timeout or CONFIG.openai_timeout
        self.seed = int(seed or CONFIG.openai_seed)
        self.auto_max_tokens = False
        self._cost_manager = CostManager()
        RateLimiter.__init__(self, rpm=self.rpm)
        logger.info(f"OpenAI API model: {self.model}")
    
    def __init_ollama(self, config):
        self.client = ollama.AsyncClient(host="http://localhost:11434")
        self.rpm = int(config.get("RPM", 10))

    def _achat_completion_stream(self, messages: list[dict], functions: List[Dict[str, Any]]=None, function_call: Union[str, Dict[str, str]]=None, event_bus: EventBus=None):
        """Stream the completion of messages."""
        pass

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
