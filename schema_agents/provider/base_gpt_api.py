#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/5 23:04
@Author  : alexanderwu
@File    : base_gpt_api.py
"""
from abc import abstractmethod
from typing import Optional, Union, Dict, List, Any

from schema_agents.provider.base_chatbot import BaseChatbot
from schema_agents.utils.common import EventBus


class BaseGPTAPI(BaseChatbot):
    """GPT API abstract class, requiring all inheritors to provide a series of standard capabilities"""
    system_prompt = 'You are a helpful assistant.'

    def _user_msg(self, msg: str) -> dict[str, str]:
        if isinstance(msg, str):
            return {"role": "user", "content": msg}
        else:
            raise TypeError(f"msg must be str or dict, but got {type(msg)}")

    def _assistant_msg(self, msg: str) -> dict[str, str]:
        return {"role": "assistant", "content": msg}

    def _system_msg(self, msg: str) -> dict[str, str]:
        return {"role": "system", "content": msg}

    def _system_msgs(self, msgs: list[str]) -> list[dict[str, str]]:
        return [self._system_msg(msg) for msg in msgs]

    def _default_system_msg(self):
        return self._system_msg(self.system_prompt)

    def ask(self, msg: Union[str, Dict[str, str]], functions: List[Dict[str, Any]]=None, function_call: Union[str, Dict[str, str]]=None, event_bus:EventBus=None) -> str:
        message = [self._default_system_msg(), self._user_msg(msg)]
        if function_call is not None:
            assert isinstance(function_call, dict) or function_call in ['none', 'auto', 'required'], f"function_call must be dict or 'none', 'auto', 'required', but got {function_call}"
        rsp = self.completion(message, functions=functions, function_call=function_call, event_bus=event_bus)
        fc = rsp.get("choices")[0]["message"].get("function_call")
        if fc is not None:
            return rsp.get("choices")[0]["message"].get("function_call")
        return self.get_choice_text(rsp)

    @abstractmethod
    async def aask(self, msg: Union[str, Dict[str, str], List[Dict[str, str]]], system_msgs: Optional[list[str]] = None, functions: List[Dict[str, Any]]=None, function_call: Union[str, Dict[str, str]]=None, event_bus: EventBus=None, use_tool_calls=True) -> str:
        """Asynchronous ask function with optional tools. Returns tool call
        schemas and prompt completion from LLM"""

    def _extract_assistant_rsp(self, context):
        return "\n".join([i["content"] for i in context if i["role"] == "assistant"])

    def ask_batch(self, msgs: list, event_bus:EventBus=None) -> str:
        context = []
        for msg in msgs:
            umsg = self._user_msg(msg)
            context.append(umsg)
            rsp = self.completion(context)
            rsp_text = self.get_choice_text(rsp)
            context.append(self._assistant_msg(rsp_text))
        return self._extract_assistant_rsp(context)

    async def aask_batch(self, msgs: list, event_bus:EventBus=None) -> str:
        """Sequential questioning"""
        context = []
        for msg in msgs:
            umsg = self._user_msg(msg)
            context.append(umsg)
            rsp_text = await self.acompletion_text(context, event_bus=event_bus)
            context.append(self._assistant_msg(rsp_text))
        return self._extract_assistant_rsp(context)

    def ask_code(self, msgs: list[str], event_bus:EventBus=None) -> str:
        """FIXME: No code segment filtering has been done here, and all results are actually displayed"""
        rsp_text = self.ask_batch(msgs, event_bus=event_bus)
        return rsp_text

    async def aask_code(self, msgs: list[str], event_bus:EventBus=None) -> str:
        """FIXME: No code segment filtering has been done here, and all results are actually displayed"""
        rsp_text = await self.aask_batch(msgs, event_bus=event_bus)
        return rsp_text

    @abstractmethod
    def completion(self, messages: list[dict], event_bus:EventBus=None,
                   functions=None, function_call=None):
        """All GPTAPIs are required to provide the standard OpenAI completion interface
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello, show me python hello world code"},
            # {"role": "assistant", "content": ...}, # If there is an answer in the history, also include it
        ]
        """

    @abstractmethod
    async def acompletion(self, messages: list[dict], event_bus:EventBus=None):
        """Asynchronous version of completion
        All GPTAPIs are required to provide the standard OpenAI completion interface
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello, show me python hello world code"},
            # {"role": "assistant", "content": ...}, # If there is an answer in the history, also include it
        ]
        """

    @abstractmethod
    async def acompletion_text(self, messages: list[dict], stream=False, event_bus:EventBus=None) -> str:
        """Asynchronous version of completion. Return str. Support stream-print"""

    def get_choice_text(self, rsp: dict) -> str:
        """Required to provide the first text of choice"""
        return rsp.choices[0].dict()["message"]["content"]
    
    def get_choice_logprobs(self, rsp: dict) -> str:
        """Required to provide the first logprobs of choice"""
        return rsp.choices[0].dict()["logprobs"]["content"]

    def messages_to_prompt(self, messages: list[dict]):
        """[{"role": "user", "content": msg}] to user: <msg> etc."""
        return '\n'.join([f"{i['role']}: {i['content']}" for i in messages])

    def messages_to_dict(self, messages):
        """objects to [{"role": "user", "content": msg}] etc."""
        return [i.to_dict() for i in messages]
