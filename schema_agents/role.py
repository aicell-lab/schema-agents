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
from pydantic import BaseModel, Field, ValidationError
from pydantic.fields import FieldInfo

from schema_agents.logs import logger
from schema_agents.utils import parse_special_json, schema_to_function
from schema_agents.llm import LLM
from schema_agents.schema import Message, RoleSetting, Session
from schema_agents.memory.long_term_memory import LongTermMemory
from schema_agents.utils import dict_to_pydantic_model
from schema_agents.utils.common import EventBus
from schema_agents.utils.common import current_session
from contextlib import asynccontextmanager
from contextvars import copy_context


@asynccontextmanager
async def create_session_context(id=None, role_setting=None):
    pre_session = current_session.get()
    if pre_session:
        id = id or pre_session.id
        role_setting = role_setting or pre_session.role_setting
    current_session.set(Session(id=id, role_setting=role_setting))
    yield copy_context()
    current_session.set(pre_session)


class ToolExecutionError(BaseModel):
    error: str
    traceback: str


class Role:
    """Role is a person or group who has a specific job or purpose within an organization."""

    def __init__(
        self,
        name="",
        profile="",
        goal="",
        constraints=None,
        instructions="",
        icon="🤖",
        long_term_memory: Optional[LongTermMemory] = None,
        event_bus: EventBus = None,
        actions: list[Callable] = None,
        **kwargs,
    ):
        self._llm = LLM(**kwargs)
        self._setting = RoleSetting(
            name=name,
            profile=profile,
            goal=goal,
            constraints=constraints,
            instructions=instructions,
            icon=icon,
        )
        self._states = []
        self._actions = actions or []
        self._role_id = str(self._setting)
        self._input_schemas = []
        self._output_schemas = []
        self._action_index = {}
        self._user_support_actions = []
        self._watch_schemas = set()
        self.long_term_memory = long_term_memory
        if event_bus:
            self.set_event_bus(event_bus)
        else:
            self.set_event_bus(
                EventBus(f"{self._setting.profile} - {self._setting.name}")
            )
        self._init_actions(self._actions)

    @property
    def llm(self):
        return self._llm

    def _reset(self):
        self._states = []
        self._actions = []

    def _watch(self, schemas: Iterable[Union[str, BaseModel]]):
        """Watch actions."""
        self._watch_schemas.update(schemas)

    def set_event_bus(self, event_bus: EventBus):
        """Set event bus."""
        self._event_bus = event_bus

        async def handle_message(msg):
            if msg.data and type(msg.data) in self._watch_schemas:
                if msg.cause_by not in self._action_index[type(msg.data)]:
                    await self.handle(msg)
            elif msg.data is None and str in self._watch_schemas:
                if msg.cause_by not in self._action_index[str]:
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
        if self._setting.instructions:
            return self._setting.instructions
        prompt = f"""You are a {self._setting.profile}, named {self._setting.name}, your goal is {self._setting.goal}"""
        if self._setting.constraints:
            prompt += f", and the constraints are: {self._setting.constraints}"
        return prompt

    @property
    def user_support_actions(self):
        return self._user_support_actions

    @property
    def prefix(self):
        return self._get_prefix()

    def _extract_schemas(self, function):
        sig = signature(function)
        positional_annotation = [
            p.annotation
            for p in sig.parameters.values()
            if p.kind == p.POSITIONAL_OR_KEYWORD
        ][0]
        assert (
            positional_annotation == str
            or isinstance(positional_annotation, typing._UnionGenericAlias)
            or issubclass(positional_annotation, BaseModel)
        ), f"Action only support pydantic BaseModel, typing.Union or str, but got {positional_annotation}"
        output_schemas = (
            [sig.return_annotation]
            if not isinstance(sig.return_annotation, typing._UnionGenericAlias)
            else list(sig.return_annotation.__args__)
        )
        input_schemas = (
            [positional_annotation]
            if not isinstance(positional_annotation, typing._UnionGenericAlias)
            else list(positional_annotation.__args__)
        )
        return input_schemas, output_schemas

    def _init_actions(self, actions):
        self._output_schemas = []
        self._input_schemas = []
        for action in actions:
            if isinstance(action, partial):
                action.__doc__ = action.func.__doc__
                action.__name__ = action.func.__name__
            assert action.__doc__, "Action must have docstring"
            assert isinstance(
                action, (partial, types.FunctionType, types.MethodType)
            ), f"Action must be function, but got {action}"
            input_schemas, output_schemas = self._extract_schemas(action)
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
        keys = [
            p.name
            for p in sig.parameters.values()
            if p.kind == p.POSITIONAL_OR_KEYWORD and p.annotation == Role
        ]
        kwargs = {k: self for k in keys}
        pos = [
            p
            for p in sig.parameters.values()
            if p.kind == p.POSITIONAL_OR_KEYWORD and p.annotation != Role
        ]
        for p in pos:
            if not msg.data and isinstance(msg.content, str):
                kwargs[p.name] = msg.content
                msg.processed_by.add(self)
                break
            elif msg.data and isinstance(
                msg.data,
                (
                    p.annotation.__args__
                    if isinstance(p.annotation, typing._UnionGenericAlias)
                    else p.annotation
                ),
            ):
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
        context_class = (
            message.data.__class__ if message.data else type(message.content)
        )
        if context_class in self._input_schemas:
            return True
        return False

    # @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
    async def handle(self, msg: Union[str, Message]) -> list[Message]:
        """Handle message"""
        if isinstance(msg, str):
            msg = Message(role="User", content=msg)
        if not self.can_handle(msg):
            raise ValueError(
                f"Invalid message, the role {self._setting} cannot handle the message: {msg}"
            )
        _session_id = msg.session_id or str(uuid.uuid4())
        msg.session_history.append(_session_id)
        messages = []

        def on_message(new_msg):
            if _session_id in new_msg.session_history:
                messages.append(new_msg)

        self._event_bus.on("message", on_message)
        try:
            context_class = msg.data.__class__ if msg.data else type(msg.content)
            responses = []
            if context_class in self._input_schemas:
                actions = self._action_index[context_class]
                for action in actions:
                    responses.append(self._run_action(action, msg))
            async with create_session_context(
                id=msg.session_id, role_setting=self._setting
            ):
                responses = await asyncio.gather(*responses)

            outputs = []
            for response in responses:
                if not response:
                    continue
                # logger.info(response)
                if isinstance(response, str):
                    output = Message(
                        content=response,
                        role=self.profile,
                        cause_by=action,
                        session_history=msg.session_history.copy(),
                    )
                else:
                    assert isinstance(
                        response, BaseModel
                    ), f"Action must return str or pydantic BaseModel, but got {response}"
                    output = Message(
                        content=response.json(),
                        data=response,
                        session_history=msg.session_history.copy(),
                        role=self.profile,
                        cause_by=action,
                    )
                # self._rc.memory.add(output)
                # logger.debug(f"{response}")
                outputs.append(output)
                async with create_session_context(
                    id=msg.session_id, role_setting=self._setting
                ):
                    await self._event_bus.aemit("message", output)
        finally:
            self._event_bus.off("message", on_message)
        return messages

    def _normalize_messages(self, req):
        input_schema = []
        if isinstance(req, str):
            messages = [{"role": "user", "content": req}]
        elif isinstance(req, dict):
            messages = [req]
        elif isinstance(req, BaseModel):
            input_schema.append(req.__class__)
            messages = [
                {
                    "role": "function",
                    "name": req.__class__.__name__,
                    "content": req.json(),
                }
            ]
        else:
            assert isinstance(req, list)
            messages = []
            for r in req:
                if isinstance(r, str):
                    messages.append({"role": "user", "content": r})
                elif isinstance(r, dict):
                    messages.append(r)
                elif isinstance(r, BaseModel):
                    input_schema.append(r.__class__)
                    messages.append(
                        {
                            "role": "function",
                            "name": r.__class__.__name__,
                            "content": r.json(),
                        }
                    )
                else:
                    raise ValueError(f"Invalid request {r}")
        return messages, input_schema

    def _parse_outputs(self, response, output_types=None, parallel_call=None):
        if response["type"] == "text":
            return response["content"], {
                "system_fingerprint": response["system_fingerprint"]
            }
        elif response["type"] == "function_call":
            func_call = response["function_call"]
            assert func_call["name"] in [
                s.__name__ for s in output_types
            ], f"Invalid function name: {func_call['name']}"
            idx = [s.__name__ for s in output_types].index(func_call["name"])
            arguments = parse_special_json(func_call["arguments"])
            function_args = output_types[idx].parse_obj(arguments)
            return function_args, {
                "system_fingerprint": response["system_fingerprint"],
                "function_call": func_call,
            }
        elif response["type"] == "tool_calls":
            tool_calls = response["tool_calls"]
            functions = []
            ids = []
            for tool_call in tool_calls:
                assert tool_call["type"] == "function"
                func_call = tool_call["function"]
                assert func_call["name"] in [
                    s.__name__ for s in output_types
                ], f"Invalid function name: {func_call['name']}"
                idx = [s.__name__ for s in output_types].index(func_call["name"])
                arguments = parse_special_json(func_call["arguments"])
                if len(arguments.keys()) == 1 and "_" in arguments.keys():
                    arguments = arguments["_"]
                model = output_types[idx]
                try:
                    fargs = model.parse_obj(arguments)
                except ValidationError:
                    model_schema = model.schema()
                    keys = model_schema["properties"].keys()
                    # only one argument
                    if len(keys) == 1:
                        fargs = model.parse_obj({list(keys)[0]: arguments})
                    else:
                        raise
                functions.append(fargs)
                ids.append(tool_call["id"])
            if not parallel_call:
                functions = functions[0]
            return functions, {
                "tool_ids": ids,
                "system_fingerprint": response["system_fingerprint"],
                "tool_calls": tool_calls,
            }

    def _parse_tools(self, tools, thoughts_schema=None):
        tool_inputs_models = []
        arg_names = []

        tool_output_models = []
        for tool in tools:
            assert callable(tool), "Tools must be callable functions"
            sig = signature(tool)
            var_positional = [
                p.name for p in sig.parameters.values() if p.kind == p.VAR_POSITIONAL
            ]
            kwargs_args = [
                p.name for p in sig.parameters.values() if p.kind != p.VAR_POSITIONAL
            ]
            arg_names.append((var_positional, kwargs_args))
            names = [p.name for p in sig.parameters.values()]
            types = [sig.parameters[name].annotation for name in names]
            defaults = []
            for i, name in enumerate(names):
                if sig.parameters[name].default == inspect._empty:
                    defaults.append(Field(..., description=types[i].__doc__))
                elif isinstance(sig.parameters[name].default, FieldInfo):
                    defaults.append(sig.parameters[name].default)
                else:
                    defaults.append(
                        Field(
                            sig.parameters[name].default, description=types[i].__doc__
                        )
                    )
            tool_output_models.append(sig.return_annotation)
            tool_args = {names[i]: (types[i], defaults[i]) for i in range(len(names))}
            if thoughts_schema:
                tool_args["thoughts"] = (
                    thoughts_schema,
                    Field(..., description="Thoughts about this tool call."),
                )
            tool_inputs_models.append(
                dict_to_pydantic_model(
                    tool.__name__,
                    tool_args,
                    tool.__doc__,
                )
            )

        return arg_names, tool_inputs_models

    async def acall(
        self,
        req,
        tools,
        output_schema=None,
        thoughts_schema=None,
        max_loop_count=10,
        return_metadata=False,
        prompt=None,
    ):
        output_schema = output_schema or str
        messages, _ = self._normalize_messages(req)
        arg_names, tool_inputs_models = self._parse_tools(
            tools, thoughts_schema=thoughts_schema
        )

        class FinalRespondToUser(BaseModel):
            """Call this tool in the end to respond to the user."""

            response: output_schema = Field(
                ..., description="Final response based on all the previous tool calls."
            )

        FinalRespondToUser.update_forward_refs(output_schema=output_schema)

        out_schemas = tool_inputs_models + [FinalRespondToUser]

        async def call_tools(tool_calls, tool_ids):
            promises = []
            for fargs in tool_calls:
                if FinalRespondToUser == fargs.__class__:
                    result_dict[FinalRespondToUser] = fargs
                    continue
                idx = tool_inputs_models.index(fargs.__class__)
                args_ns, kwargs_ns = arg_names[idx]
                args = [getattr(fargs, name) for name in args_ns]
                kwargs = {name: getattr(fargs, name) for name in kwargs_ns}
                if thoughts_schema and hasattr(fargs, "thoughts"):
                    result_dict[thoughts_schema] = fargs.thoughts

                tool = tools[idx]

                async def wrap_func():
                    try:
                        if inspect.iscoroutinefunction(tool):
                            return await tool(*args, **kwargs)
                        else:
                            loop = asyncio.get_running_loop()

                            def sync_func():
                                return tool(*args, **kwargs)

                            return await loop.run_in_executor(None, sync_func)
                    except Exception as exp:
                        return ToolExecutionError(
                            error=str(exp), traceback=traceback.format_exc()
                        )

                promises.append(wrap_func())

            results = await asyncio.gather(*promises)

            tool_call_reports = []
            for call_id, result in enumerate(results):
                fargs = tool_calls[call_id]
                idx = tool_inputs_models.index(fargs.__class__)
                result_dict[tools[idx]] = result
                tool_call_reports.append(
                    {
                        "tool_call_id": tool_ids[call_id],
                        "role": "tool",
                        "name": tools[idx].__name__,
                        "content": (
                            result.json()
                            if isinstance(result, BaseModel)
                            else str(result)
                        ),
                    }
                )  # extend conversation with function response
            return tool_call_reports

        result_steps = []
        # Extract class names in out_schemas
        fix_doc = lambda doc: doc.replace("\n", ";")[:100]
        tool_entry = lambda s: f" - {s.__name__}: {fix_doc(s.__doc__)}"
        tool_schema_names = "\n".join([tool_entry(s) for s in tool_inputs_models])
        tool_prompt = f"To respond to the question, you will be placed inside a loop where you can decide whether you want to call `FinalRespondToUser` or the following tools:\n{tool_schema_names}\nFor every step in the loop, you can call one or more tools (or the same tool with different arguments). After execution, tool call results or error will be submitted. If you want to end the loop either because you already know the anwser or too many failed tries, call `FinalRespondToUser` to end the loop. In that case, FinalRespondToUser should be only tool you call. Try to make the best use of the tools to get the answer. Always call a tool, text response is not allowed."
        loop_count = 0
        while True:
            result_dict = {}
            result_steps.append(result_dict)
            loop_count += 1
            tool_calls, metadata = await self.aask(
                messages,
                out_schemas,
                use_tool_calls=True,
                return_metadata=True,
                prompt=prompt
                or (
                    tool_prompt
                    + f"\nCurrent loop count: {loop_count}; Try to use fewer loops if possible, and NEVER exceed the maximum loop count: {max_loop_count}."
                ),
            )
            if isinstance(tool_calls, str):
                messages.append(
                    {
                        "role": "user",
                        "content": f"DO NOT respond with text directly. You MUST call `FinalRespondToUser` or the following tools:\n{tool_schema_names}",
                    }
                )
                continue

            tool_ids = metadata["tool_ids"]
            messages.append({"role": "assistant", "tool_calls": metadata["tool_calls"]})

            tool_call_reports = await call_tools(tool_calls, tool_ids)
            messages.extend(tool_call_reports)

            if FinalRespondToUser in result_dict:
                break

            if loop_count >= max_loop_count + 1:
                raise RuntimeError(
                    f"Exceeded the maximum loop count: {max_loop_count}."
                )

        response = result_dict[FinalRespondToUser]
        if return_metadata:
            return response.response, {"steps": result_steps}
        return response.response

    def _format_tool_prompt(
        self, prefix, input_schema, output_schema, parallel_call=True
    ):
        schema_names = ",".join(
            [f"`{s.__name__}`" for s in output_schema if s is not str]
        )
        avoid_schema_names = ",".join([f"`{s.__name__}`" for s in input_schema])
        allow_text = str in output_schema
        if allow_text:
            text_prompt = "respond with text or "
        else:
            text_prompt = ""
        if parallel_call:
            parallel_prompt = "one or more of these functions in parallel: "
        else:
            parallel_prompt = "one of these functions: "
        if avoid_schema_names:
            prompt = f"{prefix}you MUST {text_prompt}call {parallel_prompt}{schema_names}. DO NOT call any of the following functions: {avoid_schema_names}."
        else:
            prompt = (
                f"{prefix}you MUST {text_prompt}call {parallel_prompt}{schema_names}."
            )
        if not allow_text:
            prompt += " DO NOT respond with text directly."
        return prompt

    # @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
    async def aask(
        self,
        req,
        output_schema=None,
        prompt=None,
        use_tool_calls=False,
        return_metadata=False,
        extra_schemas=None,
    ):

        assert extra_schemas is None or isinstance(extra_schemas, list)
        output_schema = output_schema or str
        messages, input_schema = self._normalize_messages(req)
        if use_tool_calls:
            assert (
                output_schema is str
                or isinstance(output_schema, (tuple, list))
                or isinstance(output_schema, typing._UnionGenericAlias)
                or issubclass(output_schema, BaseModel)
            )
        else:
            assert (
                output_schema is str
                or isinstance(output_schema, typing._UnionGenericAlias)
                or issubclass(output_schema, BaseModel)
            )

        if extra_schemas:
            input_schema += extra_schemas

        if input_schema:
            sch = ",".join([f"`{i.__name__}`" for i in input_schema])
            prefix = (
                f"Please generate a response based on results from: {sch}, meanwhile, "
            )
        else:
            prefix = "To respond to the user, "

        parallel_call = False
        allow_str_output = False
        if output_schema is str:
            output_types = []
            prompt = prompt or f"{prefix}"
        elif isinstance(output_schema, (tuple, list)):
            # A list means can call multiple functions in parallel
            assert (
                use_tool_calls
            ), "Please set `use_tool_calls` to True when passing a list of output schemas."
            allow_str_output = str in output_schema
            output_types = [sch for sch in output_schema if sch is not str]
            parallel_call = True
            prompt = prompt or self._format_tool_prompt(
                prefix, input_schema, output_schema, parallel_call=True
            )
        elif isinstance(output_schema, typing._UnionGenericAlias):
            # A union type means can only call one function at a time
            allow_str_output = str in output_schema.__args__
            output_types = [sch for sch in output_schema.__args__ if sch is not str]
            prompt = prompt or self._format_tool_prompt(
                prefix, input_schema, output_schema.__args__, parallel_call=False
            )
        else:
            output_types = [output_schema]
            prompt = (
                prompt
                or f"{prefix}you MUST call the `{output_schema.__name__}` function."
            )

        system_prompt = self._get_prefix()
        if prompt:
            system_prompt += f"\n{prompt}"
            # messages.append({"role": "user", "content": f"{prompt}"})
        system_msgs = [system_prompt]

        if output_schema is str:
            function_call = "none"
            response = await self._llm.aask(
                messages,
                system_msgs,
                functions=[schema_to_function(s) for s in input_schema],
                function_call=function_call,
                event_bus=self._event_bus,
            )
            assert (
                response["type"] == "text"
            ), f"Invalid response type, it must be a text"
            content, metadata = self._parse_outputs(response)
            if return_metadata:
                return content, metadata
            return content

        functions = [schema_to_function(s) for s in set(output_types + input_schema)]
        if len(output_types) == 1 and not allow_str_output:
            function_call = {"name": output_types[0].__name__}
        else:
            function_call = "auto"
        response = await self._llm.aask(
            messages,
            system_msgs,
            functions=functions,
            function_call=function_call,
            event_bus=self._event_bus,
            use_tool_calls=use_tool_calls,
        )

        try:
            assert not isinstance(response, str), f"Invalid response. {prompt}"
            function_args, metadata = self._parse_outputs(
                response, output_types, parallel_call
            )
            if return_metadata:
                return function_args, metadata
            return function_args
        except Exception:
            logger.error(
                f"Failed to parse the response, error:\n{traceback.format_exc()}\nPlease regenerate to fix the error."
            )
            messages.append({"role": "assistant", "content": str(response)})
            messages.append(
                {
                    "role": "user",
                    "content": f"Failed to parse the response, error:\n{traceback.format_exc()}\nPlease regenerate to fix the error.",
                }
            )
            response = await self._llm.aask(
                messages,
                system_msgs,
                functions=functions,
                function_call=function_call,
                event_bus=self._event_bus,
                use_tool_calls=use_tool_calls,
            )
            function_args, metadata = self._parse_outputs(
                response, output_types, parallel_call
            )
            if return_metadata:
                return function_args, metadata
            return function_args
