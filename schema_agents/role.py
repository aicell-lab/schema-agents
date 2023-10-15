#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import inspect
import json
import re
import traceback
import types
import typing
from functools import partial
from inspect import signature
from typing import Dict, Iterable, List, Optional, Type, Union

from schema_agents.action import (Action, ActionOutput, parse_special_json,
                              schema_to_function)
# from schema_agents.environment import Environment
from schema_agents.memory import Memory
from schema_agents.llm import LLM
from schema_agents.logs import logger
from schema_agents.schema import Message
from schema_agents.memory.long_term_memory import LongTermMemory
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed

PREFIX_TEMPLATE = """You are a {profile}, named {name}, your goal is {goal}, and the constraint is {constraints}. """



class RoleSetting(BaseModel):
    """角色设定"""
    name: str
    profile: str
    goal: str
    constraints: Optional[str]
    desc: str

    def __str__(self):
        return f"{self.name}({self.profile})"

    def __repr__(self):
        return self.__str__()


class RoleContext(BaseModel):
    """角色运行时上下文"""
    env: 'Environment' = Field(default=None)
    memory: Memory = Field(default_factory=Memory)
    state: int = Field(default=0)
    todo: Action = Field(default=None)
    watch: set[Type[Action]] = Field(default_factory=set)
    args: BaseModel = Field(default_factory=BaseModel)

    class Config:
        arbitrary_types_allowed = True

    def check(self, role_id: str):
        pass

    @property
    def important_memory(self) -> list[Message]:
        """获得关注动作对应的信息"""
        actions = [f for f in self.watch if isinstance(f, Action)]
        schemas = [f for f in self.watch if f is str or inspect.isclass(f) and issubclass(f, BaseModel)]
        return self.memory.get_by_actions(actions) + self.memory.get_by_schemas(schemas)

    @property
    def history(self) -> list[Message]:
        return self.memory.get()


class Role:
    """角色/代理"""
    def __init__(self, name="", profile="", goal="", constraints=None, desc="", long_term_memory: Optional[LongTermMemory]=None):
        self._llm = LLM()
        self._setting = RoleSetting(name=name, profile=profile, goal=goal, constraints=constraints, desc=desc)
        self._states = []
        self._actions = []
        self._role_id = str(self._setting)
        self._rc = RoleContext(role=self)
        self._input_schemas = []
        self._output_schemas = []
        self._action_index = {}
        self._user_support_actions = []
        self.long_term_memory = long_term_memory
        
    def _reset(self):
        self._states = []
        self._actions = []


    def _watch(self, actions: Iterable[Type[Action]]):
        """监听对应的行为"""
        self._rc.watch.update(actions)
        # check RoleContext after adding watch actions
        self._rc.check(self._role_id)

    def _set_state(self, state):
        """Update the current state."""
        self._rc.state = state
        logger.debug(self._actions)
        self._rc.todo = self._actions[self._rc.state]

    def set_env(self, env: 'Environment'):
        """设置角色工作所处的环境，角色可以向环境说话，也可以通过观察接受环境消息"""
        self._rc.env = env

    @property
    def profile(self):
        """获取角色描述（职位）"""
        return self._setting.profile

    def _get_prefix(self):
        """获取角色前缀"""
        if self._setting.desc:
            return self._setting.desc
        return PREFIX_TEMPLATE.format(**self._setting.dict())


    async def _observe(self) -> int:
        """从环境中观察，获得重要信息，并加入记忆"""
        if not self._rc.env:
            return 0
        env_msgs = self._rc.env.memory.get()
        
        actions = [f for f in self._rc.watch if isinstance(f, Action)]
        schemas = [f for f in self._rc.watch if f is str or inspect.isclass(f) and issubclass(f, BaseModel)]
        observed = self._rc.env.memory.get_by_actions(actions) + self._rc.env.memory.get_by_schemas(schemas)

        news = self._rc.memory.remember(observed)  # remember recent exact or similar memories

        for i in env_msgs:
            self.recv(i)

        news_text = [f"{i.role}: {i.content[:20]}..." for i in news]
        if news_text:
            logger.debug(f'{self._setting} observed: {news_text}')
        return len(news)

    def _publish_message(self, msg):
        """如果role归属于env，那么role的消息会向env广播"""
        if not self._rc.env:
            # 如果env不存在，不发布消息
            return
        self._rc.env.publish_message(msg)


    def recv(self, message: Message) -> None:
        """add message to history."""
        # self._history += f"\n{message}"
        # self._context = self._history
        if message in self._rc.memory.get():
            return
        self._rc.memory.add(message)

    async def handle(self, message: Message) -> list[Message]:
        """接收信息，并用行动回复"""
        # logger.debug(f"{self.name=}, {self.profile=}, {message.role=}")
        self.recv(message)

        return await self._react()

    async def run(self, message=None):
        """观察，并基于观察的结果思考、行动"""
        if message:
            if isinstance(message, str):
                message = Message(message)
            if isinstance(message, Message):
                self.recv(message)
            if isinstance(message, list):
                self.recv(Message("\n".join(message)))
        elif not await self._observe():
            # 如果没有任何新信息，挂起等待
            logger.debug(f"{self._setting}: no news. waiting.")
            return

        rsp = await self._react()
        # 将回复发布到环境，等待下一个订阅者处理
        if isinstance(rsp, list):
            for msg in rsp:
                self._publish_message(msg)
        else:
            self._publish_message(rsp)
        return rsp

    @staticmethod
    def create(name, profile, goal, constraints=None, actions=None, desc='', long_term_memory: Optional[LongTermMemory]=None):
        # Convert the profile into a valid class name
        class_name = re.sub(r'\W+', '', profile.replace(' ', ''))

        # Define the __init__ method for the new class
        def __init__(self, name=name, profile=profile, goal=goal, constraints=constraints, desc=desc):
            super(self.__class__, self).__init__(name=name, profile=profile, goal=goal, constraints=constraints, desc=desc, long_term_memory=long_term_memory)
            self._init_actions(actions or [])

        # Create the new class with 'type'
        return type(class_name, (Role,), {'__init__': __init__})


    
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
            if inspect.isclass(action) and issubclass(action, Action):
                i = action("")
                i.set_prefix(self._get_prefix(), self.profile)
            elif isinstance(action, Action):
                i = action
                i.set_prefix(self._get_prefix(), self.profile)
            else:
                i = action
            self._actions.append(i)
            self._states.append(f"{idx}. {action}")
    
    async def _run_action(self, action, context):
        sig = signature(action)
        keys = [p.name for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD and p.annotation == Role]
        kwargs = {k: self for k in keys}
        pos = [p for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD and p.annotation != Role]
        for p in pos:
            for c in context:
                if not c.instruct_content and isinstance(c.content, str):
                    kwargs[p.name] = c.content
                    c.processed_by.add(self)
                    break
                elif c.instruct_content and isinstance(c.instruct_content, p.annotation.__args__ if isinstance(p.annotation, typing._UnionGenericAlias) else p.annotation):
                    kwargs[p.name] = c.instruct_content
                    c.processed_by.add(self)
                    break
            if p.name not in kwargs:
                kwargs[p.name] = None
        
        if inspect.iscoroutinefunction(action):
            return await action(**kwargs)
        else:
            return action(**kwargs)
        

    # @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
    async def _react(self) -> list[Message]:
        context = self._rc.important_memory
        # Only process messages that are not processed by this role
        context = [msg for msg in context if self not in msg.processed_by]
        assert context and isinstance(context, list)
        responses = []
        for msg in context:
            context_class = msg.instruct_content.__class__ if msg.instruct_content else type(msg.content)
            if context_class in self._input_schemas:
                actions = self._action_index[context_class]
                for action in actions:
                    responses.append(self._run_action(action, context))
        responses = await asyncio.gather(*responses)
        outputs = []  
        for response in responses:
            if not response:
                continue
            # logger.info(response)
            if isinstance(response, str):
                output = Message(content=response, role=self.profile, cause_by=action)
            else:
                assert isinstance(response, BaseModel), f"Action must return pydantic BaseModel, but got {response}"
                output = Message(content=response.json(), instruct_content=response,
                            role=self.profile, cause_by=action)
            # self._rc.memory.add(output)
            # logger.debug(f"{response}")
            outputs.append(output)
                  
        return outputs

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
    async def _aask_v2(self, prompt: Union[str, Dict[str, str]],
                       output_schema: Optional[BaseModel] = None,
                       input_schema: Optional[BaseModel] = None,
                       system_msgs: Optional[list[str]] = None,
                       schemas: List[BaseModel]=None,
                       function_call: Union[str, Dict[str, str]]=None) -> ActionOutput:
        """Append default prefix, support pydantic schema"""
        if not system_msgs:
            system_msgs = []
        
        functions = []
        schema_dict = {}
        schemas = schemas or []

        if input_schema and input_schema not in schemas:
            schemas.append(input_schema)
        if output_schema and output_schema not in schemas:
            schemas.append(output_schema)

        for schema in schemas:
            functions.append(schema_to_function(schema))
            schema_dict[schema.__name__] = schema

        if output_schema:
            function_call = function_call or {"name": output_schema.__name__}
        if isinstance(prompt, dict):
            assert input_schema is not None, f"If prompt is dict, input_schema must be provided, but got {input_schema}"
        if isinstance(prompt, dict):
            prompt = [prompt]
        if input_schema is not None:
            assert isinstance(prompt, list), f"If input_schema is provided, prompt must be dict or list, but got {type(prompt)}"
        for p in prompt:
            if p["role"] == "function":
                assert set(p.keys()) == {"name", "content", "role"}, f"If input_schema is provided, prompt must have keys 'name', 'content', 'role', but got {prompt.keys()}"
                assert json.loads(p["content"]), "prompt['content'] must be a valid json string"
        content = await self._llm.aask(prompt, system_msgs, functions=functions, function_call=function_call)
        logger.debug(content)
        if isinstance(content, str):
            return content
        assert content['name'] in schema_dict.keys(), f"Function name {content['name']} is not in schema_dict {schema_dict.keys()}"
        try:
            args = parse_special_json(content['arguments'])
        except json.JSONDecodeError:
            logger.error(f"Failed to parse arguments: {content['arguments']}")
            raise
        arguments = schema_dict[content['name']].parse_obj(args)
        return arguments

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
            return await self._llm.aask(messages, system_msgs, functions=[input_schema] if input_schema else [], function_call=function_call)

        functions = [schema_to_function(s) for s in set(output_types + ([input_schema] if input_schema else []))]
        if len(output_types) == 1:
            function_call = {"name": output_types[0].__name__}
        else:
            function_call = "auto"
        response = await self._llm.aask(messages, system_msgs, functions=functions, function_call=function_call)
        try:
            schema_names = ",".join([f"`{s.__name__}`" for s in output_types])
            assert not isinstance(response, str), f"Invalid response, you MUST call one of the following functions: {schema_names}. DO NOT return text directly."
            assert response["name"] in [s.__name__ for s in output_types], f"Invalid function name: {response['name']}"
            idx = [s.__name__ for s in output_types].index(response["name"])
            arguments = parse_special_json(response["arguments"])
            return output_types[idx].parse_obj(arguments)
        except Exception:
            messages.append({"role": "assistant", "content": str(response)})
            messages.append({"role": "user", "content": f"Failed to parse the response, error:\n{traceback.format_exc()}\nPlease regenerate to fix the error."})
            response = await self._llm.aask(messages, system_msgs, functions=functions, function_call=function_call)
            assert response["name"] in [s.__name__ for s in output_types], f"Invalid function name: {response['name']}"
            idx = [s.__name__ for s in output_types].index(response["name"])
            arguments = json.loads(response["arguments"])
            return output_types[idx].parse_obj(arguments)
            
        