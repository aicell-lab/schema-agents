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
from schema_agents.utils.common import EventBus, UnexpectedStringOutputError, current_session
from schema_agents.schema import StreamEvent
from contextvars import copy_context

def retry(max_retries):
    def decorator(f):
        @wraps(f)
        async def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return await f(*args, **kwargs)
                except Exception:
                    if i == max_retries - 1:
                        raise
                    await asyncio.sleep(2 ** i)
        return wrapper
    return decorator


class RateLimiter:
    """Rate control class, each call goes through wait_if_needed, sleep if rate control is needed"""
    def __init__(self, rpm):
        self.last_call_time = 0
        self.interval = 1.1 * 60 / rpm  # Here 1.1 is used because even if the calls are made strictly according to time, they will still be QOS'd; consider switching to simple error retry later
        self.rpm = rpm

    def split_batches(self, batch):
        return [batch[i:i + self.rpm] for i in range(0, len(batch), self.rpm)]

    async def wait_if_needed(self, num_requests):
        current_time = time.time()
        elapsed_time = current_time - self.last_call_time

        if elapsed_time < self.interval * num_requests:
            remaining_time = self.interval * num_requests - elapsed_time
            logger.info(f"sleep {remaining_time}")
            await asyncio.sleep(remaining_time)

        self.last_call_time = time.time()


class Costs(NamedTuple):
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost: float
    total_budget: float


class CostManager(metaclass=Singleton):
    """计算使用接口的开销"""
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0

    def update_cost(self, prompt_tokens, completion_tokens, model):
        """
        Update the total cost, prompt tokens, and completion tokens.

        Args:
        prompt_tokens (int): The number of tokens used in the prompt.
        completion_tokens (int): The number of tokens used in the completion.
        model (str): The model used for the API call.
        """
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        cost = (
            prompt_tokens * TOKEN_COSTS[model]["prompt"]
            + completion_tokens * TOKEN_COSTS[model]["completion"]
        ) / 1000
        self.total_cost += cost
        logger.info(f"Total running cost: ${self.total_cost:.3f} | Max budget: ${CONFIG.max_budget:.3f} | "
                    f"Current cost: ${cost:.3f}, {prompt_tokens=}, {completion_tokens=}")
        CONFIG.total_cost = self.total_cost

    def get_total_prompt_tokens(self):
        """
        Get the total number of prompt tokens.

        Returns:
        int: The total number of prompt tokens.
        """
        return self.total_prompt_tokens

    def get_total_completion_tokens(self):
        """
        Get the total number of completion tokens.

        Returns:
        int: The total number of completion tokens.
        """
        return self.total_completion_tokens

    def get_total_cost(self):
        """
        Get the total cost of API calls.

        Returns:
        float: The total cost of API calls.
        """
        return self.total_cost

    def get_costs(self) -> Costs:
        """获得所有开销"""
        return Costs(self.total_prompt_tokens, self.total_completion_tokens, self.total_cost, self.total_budget)


class OpenAIGPTAPI(BaseGPTAPI, RateLimiter):
    """
    Check https://platform.openai.com/examples for examples
    """
    def __init__(self, model=None, seed=None, temperature=None, stop=None, logprobs=None, top_logprobs=None, timeout=None):
        self.__init_openai(CONFIG)
        self.model = model or CONFIG.openai_api_model
        self.temperature = float(temperature or CONFIG.openai_temperature)
        self.stop = stop or CONFIG.openai_stop
        self.logprobs = logprobs or CONFIG.openai_logprobs
        self.top_logprobs = top_logprobs or CONFIG.openai_top_logprobs
        self.timeout = timeout or CONFIG.openai_timeout
        self.seed = int(seed or CONFIG.openai_seed)
        self.auto_max_tokens = False
        self._cost_manager = CostManager()
        RateLimiter.__init__(self, rpm=self.rpm)
        logger.info(f"OpenAI API model: {self.model}")

    def __init_openai(self, config):
        if config.openai_proxy:
            openai.http_client = httpx.Client(
                proxies=config.openai_proxy,
            )
        if config.openai_api_type:
            openai.api_type = config.openai_api_type
            openai.api_version = config.openai_api_version
        self.aclient = AsyncOpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_api_base,
        )
        self.client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_api_base,
        )
        self.rpm = int(config.get("RPM", 10))

    async def _achat_completion_stream(self, messages: list[dict], event_bus: EventBus=None, raise_for_string_output=False, **kwargs) -> str:
        response = await self.aclient.chat.completions.create(
            **self._cons_kwargs(messages, functions=kwargs.get("functions")),
            stream=True,
            **kwargs
        )

        # create variables to collect the stream of chunks
        collected_chunks = []
        collected_messages = []
        collected_logprobs = []
        function_call_detected = False
        tool_call_detected = False
        func_call = {}
        tool_calls = {}
        query_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        session = current_session.get() if current_session in copy_context() else None
        acc_message = ""
        try:
            # iterate through the stream of events
            async for raw_chunk in response:
                collected_chunks.append(raw_chunk)  # save the event response
                choice0 = raw_chunk.choices[0]
                chunk_message = choice0.delta.dict()  # extract the message
                collected_messages.append(chunk_message)  # save the message
                if kwargs.get("logprobs") and choice0.logprobs and choice0.logprobs.content:
                    collected_logprobs.extend(choice0.logprobs.content)
                if choice0.finish_reason == "length":
                    raise RuntimeError("Incomplete model output due to max_tokens parameter or token limit")
                elif choice0.finish_reason == "content_filter":
                    raise RuntimeError("Omitted content due to a flag from our content filters")

                if "tool_calls" in chunk_message and chunk_message["tool_calls"]:
                    _tool_calls = chunk_message["tool_calls"]
                    for _tool_call in _tool_calls:
                        if _tool_call.get("type") == "function":
                            func_call = _tool_call.get("function")
                            if event_bus:
                                event_bus.emit("stream", StreamEvent(type="function_call", query_id=query_id, session=session, name=func_call["name"], arguments=func_call.get("arguments", ""), status="start"))
                        if _tool_call['index'] not in tool_calls:
                            tool_calls[_tool_call['index']] = _tool_call
                        else:
                            func_call = tool_calls[_tool_call['index']]['function']
                            func_call['arguments'] += _tool_call['function']['arguments']
                            if event_bus:
                                event_bus.emit("stream", StreamEvent(type="function_call", query_id=query_id, session=session, name=func_call["name"], arguments=_tool_call['function']['arguments'], status="in_progress"))
                    tool_call_detected = True
                    
                elif "function_call" in chunk_message and chunk_message["function_call"]:
                    if "name" in chunk_message["function_call"] and chunk_message["function_call"]["name"]:
                        func_call["name"] = chunk_message["function_call"]["name"]
                        if event_bus:
                            event_bus.emit("stream", StreamEvent(type="function_call", query_id=query_id, session=session, name=func_call["name"], arguments=func_call.get("arguments", ""), status="start"))
                    if "arguments" in chunk_message["function_call"]:
                        if "arguments" not in func_call:
                            func_call["arguments"] = ""
                        func_call["arguments"] += chunk_message["function_call"]["arguments"]
                        if event_bus:
                            event_bus.emit("stream", StreamEvent(type="function_call", query_id=query_id, session=session, name=func_call["name"], arguments=chunk_message["function_call"]["arguments"], status="in_progress"))
                    function_call_detected = True
                elif "content" in chunk_message and chunk_message["content"]:
                    if raise_for_string_output:
                        raise UnexpectedStringOutputError(f"Received a unexpected string output: {chunk_message['content']}")
                    if event_bus:
                        acc_message += chunk_message["content"]
                        if acc_message == chunk_message["content"]:
                            event_bus.emit("stream", StreamEvent(type="text", query_id=query_id, session=session, content=chunk_message["content"], status="start"))
                        else:
                            event_bus.emit("stream", StreamEvent(type="text", query_id=query_id, session=session, content=chunk_message["content"], status="in_progress"))

                if event_bus:
                    if function_call_detected:
                        if choice0.finish_reason in ["function_call", "stop"]:
                            event_bus.emit("function_call", func_call)
                            event_bus.emit("stream", StreamEvent(type="function_call", query_id=query_id, session=session, name=func_call["name"], arguments=func_call["arguments"], status="finished"))
                    elif tool_call_detected:
                        if choice0.finish_reason in ["tool_calls", "stop"]:
                            for _tool_call in tool_calls.values():
                                if _tool_call.get("type") == "function":
                                    func_call = _tool_call.get("function")
                                    event_bus.emit("function_call", func_call)
                                    event_bus.emit("stream", StreamEvent(type="function_call", query_id=query_id, session=session, name=func_call["name"], arguments=func_call.get("arguments", ""), status="finished"))
        except Exception:
            raise
        finally:
            await response.close()
            
        if function_call_detected:
            full_reply_content = full_reply_content = {"type": "function_call", "function_call": func_call}
            if raw_chunk.system_fingerprint:
                full_reply_content['system_fingerprint'] = raw_chunk.system_fingerprint
            usage = self._calc_usage(messages, f"{func_call['name']}({func_call['arguments']})", functions=kwargs.get("functions", None))
        elif tool_call_detected:
            full_reply_content = {"type": "tool_calls", "tool_calls": [tool_calls[k] for k in sorted(list(tool_calls.keys()))]}
            if raw_chunk.system_fingerprint:
                full_reply_content['system_fingerprint'] = raw_chunk.system_fingerprint
            _msg = ""
            for _tool_call in tool_calls.values():
                del _tool_call['index']
                func_call = _tool_call['function']
                _msg += f"{func_call['name']}({func_call['arguments']})\n"
            tools = kwargs["tools"]
            functions = [tool['function'] for tool in tools]
            usage = self._calc_usage(messages, _msg, functions=functions)
        else:
            full_reply_content = {"type": "text", "content": ''.join([m.get('content', '') for m in collected_messages if m.get('content')])}
            if raw_chunk.system_fingerprint:
                full_reply_content['system_fingerprint'] = raw_chunk.system_fingerprint
            usage = self._calc_usage(messages, full_reply_content["content"])
            if event_bus:
                event_bus.emit("stream", StreamEvent(type="text", query_id=query_id, session=session, content=full_reply_content["content"], status="finished"))
        self._update_costs(usage)
        if event_bus:
            event_bus.emit("completion", full_reply_content)
        if collected_logprobs:
            return full_reply_content, collected_logprobs
        return full_reply_content


    def _cons_kwargs(self, messages: list[dict], functions: list[dict]=None) -> dict:
        kwargs = {
            "messages": messages,
            "max_tokens": self.get_max_tokens(messages, functions=functions),
            "n": 1,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "seed": self.seed,
        }
        if self.stop:
            kwargs["stop"] = self.stop
        if self.logprobs:
            kwargs["logprobs"] = self.logprobs
        if self.top_logprobs:
            kwargs["top_logprobs"] = self.top_logprobs

        if CONFIG.openai_api_type == "azure":
            if CONFIG.deployment_name and CONFIG.deployment_id:
                raise ValueError("You can only use one of the `deployment_id` or `deployment_name` model")
            elif not CONFIG.deployment_name and not CONFIG.deployment_id:
                raise ValueError("You must specify `DEPLOYMENT_NAME` or `DEPLOYMENT_ID` parameter")
            kwargs_mode = (
                {"engine": CONFIG.deployment_name}
                if CONFIG.deployment_name
                else {"deployment_id": CONFIG.deployment_id}
            )
        else:
            kwargs_mode = {"model": self.model}
        kwargs.update(kwargs_mode)
        return kwargs

    async def _achat_completion(self, messages: list[dict], event_bus: EventBus=None, **kwargs) -> dict:
        kwargs.update(self._cons_kwargs(messages, functions=kwargs.get("functions")))
        rsp = await self.aclient.chat.completions.create(**kwargs)
        self._update_costs(rsp.usage.dict())
        if event_bus:
            event_bus.emit("completion", rsp)
        return rsp

    def _chat_completion(self, messages: list[dict], event_bus: EventBus=None) -> dict:
        rsp = self.client.chat.completions.create(**self._cons_kwargs(messages))
        self._update_costs(rsp)
        if event_bus:
            event_bus.emit("completion", rsp)
        return rsp

    def completion(self, messages: list[dict], event_bus: EventBus=None) -> dict:
        rsp = self._chat_completion(messages, event_bus=event_bus)
        return rsp

    async def acompletion(self, messages: list[dict], event_bus: EventBus=None, **kwargs) -> dict:
        rsp = await self._achat_completion_stream(messages, event_bus=event_bus, **kwargs)
        return rsp
    
    async def acompletion_function(self, messages: list[dict], functions: List[Dict[str, Any]]=None, function_call: Union[str, Dict[str, str]]=None, event_bus: EventBus=None, raise_for_string_output=False, **kwargs) -> dict:
        rsp = await self._achat_completion_stream(messages, functions=functions, function_call=function_call, event_bus=event_bus, raise_for_string_output=raise_for_string_output, **kwargs)
        return rsp

    async def acompletion_tool(self, messages: list[dict], tools: List[Dict[str, Any]]=None, tool_choice: Union[str, Dict[str, str]]=None, event_bus: EventBus=None, raise_for_string_output=False,**kwargs) -> dict:
        rsp = await self._achat_completion_stream(messages, tools=tools, tool_choice=tool_choice, event_bus=event_bus, raise_for_string_output=raise_for_string_output, **kwargs)
        return rsp

    async def acompletion_text(self, messages: list[dict], stream=False, event_bus: EventBus=None, **kwargs) -> str:
        """when streaming, print each token in place."""    
        if stream:
            return await self._achat_completion_stream(messages, event_bus=event_bus, **kwargs)
        rsp = await self._achat_completion(messages, event_bus=event_bus, **kwargs)
        if "logprobs" in kwargs:
            return self.get_choice_text(rsp), self.get_choice_logprobs(rsp)

        return self.get_choice_text(rsp)

    def _calc_usage(self, messages: list[dict], rsp: str, functions: List[dict]=None) -> dict:
        usage = {}
        prompt_tokens = count_message_tokens(messages, self.model, functions=functions)
        completion_tokens = count_string_tokens(rsp, self.model)
        usage['prompt_tokens'] = prompt_tokens
        usage['completion_tokens'] = completion_tokens
        return usage

    async def acompletion_batch(self, batch: list[list[dict]], event_bus: EventBus=None) -> list[dict]:
        split_batches = self.split_batches(batch)
        all_results = []

        for small_batch in split_batches:
            logger.info(small_batch)
            await self.wait_if_needed(len(small_batch))

            future = [self.acompletion(prompt, event_bus=event_bus) for prompt in small_batch]
            results = await asyncio.gather(*future)
            logger.info(results)
            all_results.extend(results)

        return all_results

    async def acompletion_batch_text(self, batch: list[list[dict]], event_bus: EventBus=None) -> list[str]:
        raw_results = await self.acompletion_batch(batch, event_bus=event_bus)
        results = []
        for idx, raw_result in enumerate(raw_results, start=1):
            result = self.get_choice_text(raw_result)
            results.append(result)
            logger.info(f"Result of task {idx}: {result}")
        return results

    def _update_costs(self, usage: dict):
        prompt_tokens = int(usage['prompt_tokens'])
        completion_tokens = int(usage['completion_tokens'])
        self._cost_manager.update_cost(prompt_tokens, completion_tokens, self.model)

    def get_costs(self) -> Costs:
        return self._cost_manager.get_costs()

    def get_max_tokens(self, messages: list[dict], functions: List[dict]=None):
        if not self.auto_max_tokens:
            return CONFIG.max_tokens_rsp
        return get_max_completion_tokens(messages, functions, self.model, CONFIG.max_tokens_rsp)

    def moderation(self, content: Union[str, list[str]]):
        try:
            if not content:
                logger.error("content cannot be empty!")
            else:
                rsp = self._moderation(content=content)
                return rsp
        except Exception as e:
            logger.error(f"moderating failed:{e}")

    def _moderation(self, content: Union[str, list[str]]):
        rsp = self.client.chat.moderations.create(input=content)
        return rsp

    async def amoderation(self, content: Union[str, list[str]]):
        try:
            if not content:
                logger.error("content cannot be empty!")
            else:
                rsp = await self._amoderation(content=content)
                return rsp
        except Exception as e:
            logger.error(f"moderating failed:{e}")

    async def _amoderation(self, content: Union[str, list[str]]):
        rsp = await self.aclient.chat.moderations.create(input=content)
        return rsp
