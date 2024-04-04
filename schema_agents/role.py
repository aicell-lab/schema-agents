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
from typing import Iterable, Optional, Union, Callable, List
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


class PlanStep(BaseModel):
    """A step in a plan."""

    goals: str = Field(
        ..., description="A concise description of the goals of this step."
    )
    tools: Optional[List[str]] = Field(
        None,
        description="If any, a list tool names to be called. Multiple tools can be called in parallel.",
    )
    condition: Optional[str] = Field(
        None,
        description="A condition under which this step should be executed, if any.",
    )
    max_loops: Optional[int] = Field(
        None, description="The maximum number of loops or attempts for this step."
    )


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
        register_default_events: bool = False,
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
        if register_default_events:
            event_bus = self.get_event_bus()
            event_bus.register_default_events()
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
                        content=response.model_dump_json(),
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
                    "content": req.model_dump_json(),
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
                            "content": r.model_dump_json(),
                        }
                    )
                else:
                    raise ValueError(f"Invalid request {r}")
        return messages, input_schema

    def _parse_outputs(self, response, output_types=None, parallel_call=None):
        if response["type"] == "text":
            return response["content"], {
                "system_fingerprint": response.get("system_fingerprint")
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
                "system_fingerprint": response.get("system_fingerprint"),
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
                "system_fingerprint": response.get("system_fingerprint"),
                "tool_calls": tool_calls,
            }

    def _parse_tools(self, tools, internal_tools, thoughts_schema=None, localns=None):
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
            # types = [sig.parameters[name].annotation for name in names]
            type_hints = typing.get_type_hints(tool, localns=localns)
            types = [type_hints[name] for name in names]
            for t in types:
                if inspect.isclass(t) and issubclass(t, BaseModel):
                    t.__name__ = t.__name__ + "_IN"  # avoid name conflict
                    t.model_rebuild()

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
            # skip adding thoughts to internal tools
            if thoughts_schema and tool not in internal_tools:
                tool_args["thoughts"] = (
                    Optional[thoughts_schema],
                    Field(None, description="Thoughts for the tool call."),
                )
            for t in types:
                if inspect.isclass(t) and issubclass(t, BaseModel):
                    assert (
                        tool.__name__ != t.__name__
                    ), f"Tool name cannot be the same as the input type name. {tool.__name__} == {t.__name__}"
            model = dict_to_pydantic_model(
                tool.__name__,
                tool_args,
                tool.__doc__,
            )
            model.model_rebuild()
            tool_inputs_models.append(model)

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
        tool_usage_prompt=None,
    ):
        for tool in tools:
            assert hasattr(tool, '__is_tool__') and hasattr(tool, 'input_model'), f"Tool function `{tool.__name__}` must be decorated with `@tool`"
        output_schema = output_schema or str
        messages, _ = self._normalize_messages(req)
        _max_loop = 5

        async def CompleteUserQuery(
            response: output_schema = Field(
                ..., description="Final response based on all the previous tool calls."
            ),
            knowledge_update: Optional[str] = Field(
                None,
                title="Knowledge Update",
                description="Optional internal notes on insights or observations made during task execution, intended for storing in the agent's memory to inform and improve future interactions.",
            ),
        ):
            """Call this tool in the end to respond to the user."""
            return "Done"

        async def StartNewPlan(
            steps: List[PlanStep] = Field(
                ...,
                title="Plan Steps",
                description="A list of steps that make up the plan.",
            ),
            rationale: Optional[str] = Field(
                None,
                title="Rationale for Plan",
                description="An optional field for providing a rationale or explanation for the plan.",
            ),
        ):
            """
            This tool the agent to sketch a structured plan with ordered steps, each with actions and optional condition for execution.
            """
            nonlocal _max_loop, current_out_schemas
            for step in steps:
                step.max_loops = step.max_loops or 1
                _max_loop += step.max_loops
            _max_loop += 1  # to account for the final response
            # reset the current_out_schemas to the tool output schemas
            current_out_schemas = all_out_schemas
            return "Done"

        internal_tools = [CompleteUserQuery, StartNewPlan]
        tools = internal_tools + tools

        arg_names, tool_inputs_models = self._parse_tools(
            tools,
            internal_tools,
            thoughts_schema=thoughts_schema,
            localns=locals(),  # to get FinalResponse
        )

        complete_user_query_model = tool_inputs_models[tools.index(CompleteUserQuery)]
        all_out_schemas = tool_inputs_models

        async def call_tool(tool, args, kwargs):
            try:
                if inspect.iscoroutinefunction(tool):
                    return await tool(*args, **kwargs)
                else:
                    loop = asyncio.get_running_loop()

                    def sync_func():
                        result = tool(*args, **kwargs)
                        assert not inspect.isawaitable(result), f"Sync function {tool.__name__} returned an awaitable object. Use async function instead."
                        return result

                    return await loop.run_in_executor(None, sync_func)
            except Exception as exp:
                return ToolExecutionError(
                    error=str(exp), traceback=traceback.format_exc()
                )

        async def call_tools(tool_calls, tool_ids, result_list):
            promises = []
            for fargs in tool_calls:
                idx = tool_inputs_models.index(fargs.__class__)
                args_ns, kwargs_ns = arg_names[idx]
                args = [getattr(fargs, name) for name in args_ns]
                kwargs = {name: getattr(fargs, name) for name in kwargs_ns}
                func_info = {
                    "name": tools[idx].__name__,
                    "args": args,
                    "kwargs": kwargs,
                }
                result_list.append(func_info)
                if thoughts_schema and hasattr(fargs, "thoughts"):
                    func_info["thoughts"] = fargs.thoughts

                tool = tools[idx]
                promises.append(call_tool(tool, args, kwargs))

            results = await asyncio.gather(*promises)

            tool_call_reports = []
            for call_id, result in enumerate(results):
                fargs = tool_calls[call_id]
                idx = tool_inputs_models.index(fargs.__class__)
                result_list[call_id]["result"] = result
                tool_call_reports.append(
                    {
                        "tool_call_id": tool_ids[call_id],
                        "role": "tool",
                        "name": tools[idx].__name__,
                        "content": (
                            result.model_dump_json()
                            if isinstance(result, BaseModel)
                            else str(result)
                        )
                        or "None",
                    }
                )  # extend conversation with function response
            return tool_call_reports

        result_steps = []
        # Extract class names in current_out_schemas
        # fix_doc = lambda doc: doc.replace("\n", ";")[:100]
        # tool_entry = lambda s: f" - {s.__name__}: {fix_doc(s.__doc__)}"
        internal_tool_names = [s.__name__ for s in internal_tools]
        get_doc = lambda s: s.__doc__.replace('\n', ';')[:100]
        tool_schema_names = "\n".join(
            [
                f" - {s.__name__}: {get_doc(s)}"
                for s in tool_inputs_models
                if s not in internal_tool_names
            ]
        )
        tool_usage_prompt = tool_usage_prompt or f"Utilize listed tools for specific tasks: {tool_schema_names}\n"
        prompt = prompt or (
            "Address the user's query exclusively through the `CompleteUserQuery` tool for final responses.\n"
            f"{tool_usage_prompt}\n"
            "Directly use `CompleteUserQuery` for straightforward queries. For more complex inquiries, create a strategic plan with `StartNewPlan` and then execute the plan, adhering to this approach unless significant updates are necessary. Compile all insights into one final summary via `CompleteUserQuery`. Text responses generated during the process will serve as comments in the loop history and will not be shown to users. "
            "Ensure loops are concise and within limits, keeping each action relevant to the original query. If the query is unresolved, outline your approach for internal review and, if required, seek further clarification. "
            "IMPORTANT: The only way to communicate a response to the user is by calling `CompleteUserQuery`. All text responses will be prefixed with `[Internal Comment]:`, contributing to the loop history but not visible to users."
        )

        messages.append(
            {"role": "system", "content": prompt}  # + loop_count_prompt,
        )

        loop_count = 0
        final_response = None
        current_out_schemas = all_out_schemas
        while True:
            loop_count += 1
            loop_count_prompt = (
                f"\nLoop count: {loop_count}/{_max_loop}. Exceeding the max loop count will terminate the process. "
                "If needed, call `StartNewPlan` to update the maximum loop count as you near the maximum."
            )
            tool_calls, metadata = await self.aask(
                messages,
                current_out_schemas,
                use_tool_calls=True,
                return_metadata=True,
                prompt=loop_count_prompt,
                raise_for_string_output=False,
            )
            if isinstance(tool_calls, str):
                messages.append(
                    {
                        "role": "assistant",
                        "content": tool_calls if tool_calls.startswith("Internal Comment:") else f"[Internal Comment]: {tool_calls}",
                    }
                )
                continue

            tool_ids = metadata["tool_ids"]
            messages.append({"role": "assistant", "tool_calls": metadata["tool_calls"]})

            result_list = []

            for fargs in tool_calls:
                idx = tool_inputs_models.index(fargs.__class__)
                if tools[idx] == CompleteUserQuery:
                    args_ns, kwargs_ns = arg_names[idx]
                    args = [getattr(fargs, name) for name in args_ns]
                    kwargs = {name: getattr(fargs, name) for name in kwargs_ns}
                    func_info = {
                        "name": tools[idx].__name__,
                        "args": args,
                        "kwargs": kwargs,
                    }
                    if len(result_steps) > 1:
                        result_list.append(func_info)
                    final_response = fargs.response
                    break

            if final_response:
                if result_list:
                    result_steps.append(result_list)
                break

            tool_call_reports = await call_tools(tool_calls, tool_ids, result_list)
            if tool_call_reports:
                messages.extend(tool_call_reports)

            _max = min(_max_loop + 1, max_loop_count)
            if loop_count >= _max_loop:
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"Force terminating the process due to exceeding the maximum loop count: {_max_loop}.",
                    }
                )
                # Force quitting by setting the only tool to CompleteUserQuery
                current_out_schemas = [complete_user_query_model]
            elif loop_count >= _max:
                raise RuntimeError(f"Exceeded the maximum loop count: {_max}.")

            if result_list:
                result_steps.append(result_list)

        if return_metadata:
            return final_response, {"steps": result_steps}
        return final_response

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

    async def aask(
        self,
        req,
        output_schema=None,
        prompt=None,
        use_tool_calls=False,
        return_metadata=False,
        extra_schemas=None,
        raise_for_string_output=None,
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
            raise_for_string_output=raise_for_string_output if raise_for_string_output is not None else (not allow_str_output),
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
                    "role": "system",
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
