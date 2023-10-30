#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import inspect
import json
import uuid
import traceback
import types
import typing
from functools import partial
from inspect import signature
from typing import Iterable, Optional, Union, Callable
from pydantic import BaseModel

from schema_agents.logs import logger
from schema_agents.utils import (parse_special_json,
                              schema_to_function)
from schema_agents.llm import LLM
from schema_agents.schema import Message
from schema_agents.memory.long_term_memory import LongTermMemory
from schema_agents.utils.common import EventBus
from pydantic import BaseModel

PREFIX_TEMPLATE = """You are a {profile}, named {name}, your goal is {goal}, and the constraint is {constraints}. """

class RoleSetting(BaseModel):
    """Role setting"""
    name: str
    profile: str
    goal: str
    constraints: Optional[str]
    desc: str

    def __str__(self):
        return f"{self.name}({self.profile})"

    def __repr__(self):
        return self.__str__()


class Role:
    """Role is a person or group who has a specific job or purpose within an organization."""
    def __init__(self, name="", profile="", goal="", constraints=None, desc="", long_term_memory: Optional[LongTermMemory]=None, event_bus:EventBus =None, actions: list[Callable] = None):
        self._llm = LLM()
        self._setting = RoleSetting(name=name, profile=profile, goal=goal, constraints=constraints, desc=desc)
        self._states = []
        self._actions = actions or []
        self._role_id = str(self._setting)
        self._input_schemas = []
        self._output_schemas = []
        self._action_index = {}
        self._user_support_actions = []
        self._watch_actions = set()
        self.long_term_memory = long_term_memory
        if event_bus:
            self.set_event_bus(event_bus)
        else:
            self.set_event_bus(EventBus(f"{self._setting.profile} - {self._setting.name}"))
        self._init_actions(self._actions)

    def _reset(self):
        self._states = []
        self._actions = []


    def _watch(self, actions: Iterable[Union[str, BaseModel]]):
        """Watch actions."""
        self._watch_actions.update(actions)
    
    def set_event_bus(self, event_bus: EventBus):
        """Set event bus."""
        self._event_bus = event_bus

        async def handle_message(msg):
            if msg.data and type(msg.data) in self._watch_actions:
                await self.handle(msg)
            elif msg.data is None and str in self._watch_actions:
                await self.handle(msg)

        self._event_bus.on("message", handle_message)
        logger.info(f"Mounting {self._setting} to event bus: {self._event_bus.name}.")
    
    def get_event_bus(self):
        """Get event bus."""
        return self._event_bus

    @property
    def profile(self):
        """Get profile."""
        return self._setting.profile

    def _get_prefix(self):
        """Get prefix."""
        if self._setting.desc:
            return self._setting.desc
        return PREFIX_TEMPLATE.format(**self._setting.dict())
    
    @property
    def user_support_actions(self):
        return self._user_support_actions
    
    @property
    def prefix(self):
        return self._get_prefix()

    def _init_actions(self, actions):
        self._output_schemas = []
        self._input_schemas = []
        for action in actions:
            if isinstance(action, partial):
                action.__doc__ = action.func.__doc__
                action.__name__ = action.func.__name__
            assert action.__doc__, "Action must have docstring"
            assert isinstance(action, (partial, types.FunctionType, types.MethodType)), f"Action must be function, but got {action}"
            sig = signature(action)
            positional_annotation = [p.annotation for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD][0]
            assert positional_annotation == str or isinstance(positional_annotation, typing._UnionGenericAlias) or issubclass(positional_annotation, BaseModel), f"Action only support pydantic BaseModel, typing.Union or str, but got {positional_annotation}"
            output_schemas = [sig.return_annotation] if not isinstance(sig.return_annotation, typing._UnionGenericAlias) else list(sig.return_annotation.__args__)
            input_schemas = [positional_annotation] if not isinstance(positional_annotation, typing._UnionGenericAlias) else list(positional_annotation.__args__)
            self._output_schemas += output_schemas
            self._input_schemas += input_schemas
            for schema in input_schemas:
                if schema not in self._action_index:
                    self._action_index[schema] = [action]
                else:
                    self._action_index[schema].append(action)
            # mark as user support action if the input schema is str
            if str in input_schemas:
                self._user_support_actions.append(action)
        self._output_schemas = list(set(self._output_schemas))
        self._input_schemas = list(set(self._input_schemas))
        self._watch(self._input_schemas)
        
        self._reset()
        for idx, action in enumerate(actions):
            self._actions.append(action)
            self._states.append(f"{idx}. {action}")
    
    async def _run_action(self, action, msg):
        sig = signature(action)
        keys = [p.name for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD and p.annotation == Role]
        kwargs = {k: self for k in keys}
        pos = [p for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD and p.annotation != Role]
        for p in pos:
            if not msg.data and isinstance(msg.content, str):
                kwargs[p.name] = msg.content
                msg.processed_by.add(self)
                break
            elif msg.data and isinstance(msg.data, p.annotation.__args__ if isinstance(p.annotation, typing._UnionGenericAlias) else p.annotation):
                kwargs[p.name] = msg.data
                msg.processed_by.add(self)
                break
            if p.name not in kwargs:
                kwargs[p.name] = None
        
        if inspect.iscoroutinefunction(action):
            return await action(**kwargs)
        else:
            return action(**kwargs)

    def can_handle(self, message: Message) -> bool:
        """Check if the role can handle the message."""
        context_class = message.data.__class__ if message.data else type(message.content)
        if context_class in self._input_schemas:
            return True
        return False
    
    # @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
    async def handle(self, msg: Union[str, Message]) -> list[Message]:
        """Handle message"""
        if isinstance(msg, str):
            msg = Message(role="User", content=msg)
        if not self.can_handle(msg):
            raise ValueError(f"Invalid message, the role {self._setting} cannot handle the message: {msg}")
        session_id = str(uuid.uuid4())
        msg.session_ids.append(session_id)
        messages = []
        def on_message(new_msg):
            if session_id in new_msg.session_ids:
                messages.append(new_msg)

        self._event_bus.on("message", on_message)
        try:
            context_class = msg.data.__class__ if msg.data else type(msg.content)
            responses = []
            if context_class in self._input_schemas:
                actions = self._action_index[context_class]
                for action in actions:
                    responses.append(self._run_action(action, msg))
            responses = await asyncio.gather(*responses)
            outputs = []  
            for response in responses:
                if not response:
                    continue
                # logger.info(response)
                if isinstance(response, str):
                    output = Message(content=response, role=self.profile, cause_by=action, session_ids=msg.session_ids.copy())
                else:
                    assert isinstance(response, BaseModel), f"Action must return pydantic BaseModel, but got {response}"
                    output = Message(content=response.json(), data=response, session_ids=msg.session_ids.copy(),
                                role=self.profile, cause_by=action)
                # self._rc.memory.add(output)
                # logger.debug(f"{response}")
                outputs.append(output)
                await self._event_bus.aemit("message", output)
        finally:
            self._event_bus.off("message", on_message)
        return messages

    # @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
    async def aask(self, req, output_schema=None, prompt=None):
        output_schema = output_schema or str
        if isinstance(req, str):
            messages = [{"role": "user", "content": req}]
            input_schema = None
        elif isinstance(req, dict):
            messages = [req]
            input_schema = None
        elif isinstance(req, BaseModel):
            input_schema = req.__class__
            messages = [{"role": "function", "name": input_schema.__name__, "content": req.json()}]
        else:
            assert isinstance(req, list)
            messages = []
            for r in req:
                if isinstance(r, str):
                    messages.append({"role": "user", "content": r})
                    input_schema = None
                elif isinstance(r, dict):
                    messages.append(r)
                    input_schema = None
                elif isinstance(r, BaseModel):
                    input_schema = r.__class__
                    messages.append({"role": "function", "name": input_schema.__name__, "content": r.json()})
                else:
                    raise ValueError(f"Invalid request {r}")
        
        assert output_schema is str or isinstance(output_schema, typing._UnionGenericAlias) or issubclass(output_schema, BaseModel)
        
        if input_schema:
            prefix = f"Please generate a response based on the result of `{input_schema.__name__}`. "
        else:
            prefix = ""
        if output_schema is str:
            output_types = []
            prompt = prompt or f"{prefix}"
            messages.append({"role": "user", "content": f"{prompt}"})
        elif isinstance(output_schema, typing._UnionGenericAlias):
            output_types = list(output_schema.__args__)
            schema_names = ",".join([f"`{s.__name__}`" for s in output_types])
            prompt = prompt or f"{prefix}You MUST call one of the following functions: {schema_names}. DO NOT return text directly."
            messages.append({"role": "user", "content": f"{prompt}"})
        else:
            output_types = [output_schema]
            prompt = prompt or f"{prefix}You MUST call the `{output_schema.__name__}` function."
            messages.append({"role": "user", "content": f"{prompt}"})
        system_msgs = [self._get_prefix()]
        
        if output_schema is str:
            function_call = "none"
            return await self._llm.aask(messages, system_msgs, functions=[input_schema] if input_schema else [], function_call=function_call, event_bus=self._event_bus)

        functions = [schema_to_function(s) for s in set(output_types + ([input_schema] if input_schema else []))]
        if len(output_types) == 1:
            function_call = {"name": output_types[0].__name__}
        else:
            function_call = "auto"
        response = await self._llm.aask(messages, system_msgs, functions=functions, function_call=function_call, event_bus=self._event_bus)
        try:
            schema_names = ",".join([f"`{s.__name__}`" for s in output_types])
            assert not isinstance(response, str), f"Invalid response, you MUST call one of the following functions: {schema_names}. DO NOT return text directly."
            assert response["name"] in [s.__name__ for s in output_types], f"Invalid function name: {response['name']}"
            idx = [s.__name__ for s in output_types].index(response["name"])
            arguments = parse_special_json(response["arguments"])
            return output_types[idx].parse_obj(arguments)
        except Exception:
            logger.error(f"Failed to parse the response, error:\n{traceback.format_exc()}\nPlease regenerate to fix the error.")
            messages.append({"role": "assistant", "content": str(response)})
            messages.append({"role": "user", "content": f"Failed to parse the response, error:\n{traceback.format_exc()}\nPlease regenerate to fix the error."})
            response = await self._llm.aask(messages, system_msgs, functions=functions, function_call=function_call, event_bus=self._event_bus)
            assert response["name"] in [s.__name__ for s in output_types], f"Invalid function name: {response['name']}"
            idx = [s.__name__ for s in output_types].index(response["name"])
            arguments = json.loads(response["arguments"])
            return output_types[idx].parse_obj(arguments)
            
        