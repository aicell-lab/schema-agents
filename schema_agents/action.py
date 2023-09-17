#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:43
@Author  : alexanderwu
@File    : action.py
"""

import inspect
import json
import re
import typing
from abc import ABC
from functools import partial
from inspect import signature
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

from schema_agents.llm import LLM
from schema_agents.logs import logger
from schema_agents.utils.common import OutputParser
from pydantic import BaseModel, create_model, root_validator, validator
from tenacity import retry, stop_after_attempt, wait_fixed


class ActionOutput:
    content: str
    instruct_content: BaseModel

    def __init__(self, content: str, instruct_content: BaseModel):
        self.content = content
        self.instruct_content = instruct_content

    @classmethod
    def create_model_class(cls, class_name: str, mapping: Dict[str, Type]):
        new_class = create_model(class_name, **mapping)

        @validator('*', allow_reuse=True)
        def check_name(v, field):
            if field.name not in mapping.keys():
                raise ValueError(f'Unrecognized block: {field.name}')
            return v

        @root_validator(pre=True, allow_reuse=True)
        def check_missing_fields(values):
            required_fields = set(mapping.keys())
            missing_fields = required_fields - set(values.keys())
            if missing_fields:
                raise ValueError(f'Missing fields: {missing_fields}')
            return values

        new_class.__validator_check_name = classmethod(check_name)
        new_class.__root_validator_check_missing_fields = classmethod(check_missing_fields)
        return new_class


_converted_actions = {}

def _to_title_case(snake_case_name: str) -> str:
    if "_" not in snake_case_name:
        # capitalize first letter, keep other letters
        return snake_case_name[0].upper() + snake_case_name[1:]
    words = snake_case_name.split('_')
    title_case_name = ''.join(word.capitalize() for word in words)
    return title_case_name


def _func_to_action(func):
    sig = signature(func)
    positional_annotation = [p.annotation for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD][0]
    assert positional_annotation == str or issubclass(positional_annotation, BaseModel), f"Action only support pydantic BaseModel or str, but got {positional_annotation}"
    output_schemas = [sig.return_annotation] if not isinstance(sig.return_annotation, typing._UnionGenericAlias) else list(sig.return_annotation.__args__)
    input_schemas = [positional_annotation] if not isinstance(positional_annotation, typing._UnionGenericAlias) else list(positional_annotation.__args__)

    def constructor(self, name="", desc="", system_prompt=None, context=None, llm=None):
        Action.__init__(self, name=name, desc=desc, input_schemas=input_schemas, output_schemas=output_schemas, system_prompt=system_prompt, context=context, llm=llm)  
    
    _actions = [p.name for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD and p.annotation == Action]

    async def _run(self, *args, **kwargs):
        for k in _actions:
            kwargs[k] = self
        if inspect.iscoroutinefunction(func):
            ret = await func(*args, **kwargs)
        else:
            ret = func(*args, **kwargs)
        assert isinstance(ret, sig.return_annotation), f"Function {func} must return {sig.return_annotation}, but got {ret}"
        if sig.return_annotation == str:
            return ActionOutput(content=ret, instruct_content=None)

        return ActionOutput(content=ret.json(),
                            instruct_content=ret)

    if isinstance(func, partial):
        name = func.func.__name__
    else:
        name = func.__name__
    # creating class dynamically
    return type(_to_title_case(name), (Action, ), {
        "__init__": constructor,
        "run": _run,
        "__doc__": func.__doc__
    })

def parse_special_json(json_string):
    # Regex pattern to find string values enclosed in double quotes or backticks, considering escaped quotes
    pattern = r'"(?:[^"\\]|\\.)*"|`[^`]*`'
    # Extract all matches and store them in a list
    code_blocks = re.findall(pattern, json_string)

    mapping = {}
    # Replace each match in the JSON string with a special placeholder
    for i, block in enumerate(code_blocks):
        json_string = json_string.replace(f'{block}', f'"###CODE-BLOCK-PLACEHOLDER-{i}###"')
        mapping[f'###CODE-BLOCK-PLACEHOLDER-{i}###'] = block[1:-1].encode('utf-8').decode('unicode_escape')

    # Parse the JSON string into a Python dictionary
    data = json.loads(json_string)
    
    def restore_codeblock(data):
        if isinstance(data, str):
            if re.match(r'###CODE-BLOCK-PLACEHOLDER-\d+###', data):
                return mapping[data]
            else:
                return data
        if isinstance(data, (int, float, bool)) or data is None:
            return data
        # Replace each placeholder with the corresponding code block
        if isinstance(data, list):
            cdata = []
            for d in data:
                cdata.append(restore_codeblock(d))
            return cdata

        assert isinstance(data, dict)
        cdata = {}
        for key in list(data.keys()):
            value = data[key]
            value = restore_codeblock(value)
            key = restore_codeblock(key)
            cdata[key] = value
        return cdata
    
    return restore_codeblock(data)


def output_schema_to_instruct_content(arguments: BaseModel, output_schema, output_class_name):
    data = arguments.dict()
    output_data_mapping, name_title_mapping = schema_to_output_mapping(output_schema)
    parsed_data = {}
    for key, value in data.items():
        parsed_data[name_title_mapping[key]] = value
    output_class = ActionOutput.create_model_class(output_class_name, output_data_mapping)
    instruct_content = output_class(**parsed_data)
    return instruct_content

def schema_to_output_mapping(schema):
    assert issubclass(schema, BaseModel)
    output_mapping = {}
    field_name_to_title = {}
    for name, field in schema.__annotations__.items():
        # extract field type and subtype (if any)
        if get_origin(field):
            field_type = get_origin(field)
            field_subtype = get_args(field)
        else:
            field_type = field
            field_subtype = None

        # retrieve field title
        field_title = schema.__fields__[name].field_info.title

        # add to the dictionaries
        output_mapping[field_title] = (field_type, field_subtype)
        field_name_to_title[name] = field_title
    
    return output_mapping, field_name_to_title

# https://stackoverflow.com/a/58938747
def remove_a_key(d, remove_key):
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == remove_key:
                del d[key]
            else:
                remove_a_key(d[key], remove_key)

def schema_to_function(schema: Any):
    assert schema.__doc__, f"{schema.__name__} is missing a docstring."
    assert (
        "title" not in schema.__fields__.keys()
    ), "`title` is a reserved keyword and cannot be used as a field name."
    schema_dict = schema.schema()
    remove_a_key(schema_dict, "title")

    return {
        "name": schema.__name__,
        "description": schema.__doc__,
        "parameters": schema_dict,
    }



class Action(ABC):
    def __init__(self, name: str, desc: str="", input_schemas: List[BaseModel]=None, output_schemas: List[BaseModel]=None, system_prompt: Optional[str]=None, context=None, llm=None):
        self.name: str = name
        if llm is None:
            llm = LLM()
        self.llm = llm
        self.context = context
        self.prefix = ""
        self.profile = ""
        self.desc = ""
        self.content = ""
        self.instruct_content = None
        self.desc = desc
        self._system_prompt = system_prompt
        self._input_schemas = input_schemas or []
        self._output_schemas = output_schemas or []
        assert len(self._input_schemas) >= 0
        assert len(self._output_schemas) >= 0

    def set_prefix(self, prefix, profile):
        """Set prefix for later usage"""
        self.prefix = prefix
        self.profile = profile

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()

    async def _aask(self, prompt: str, system_msgs: Optional[list[str]] = None) -> str:
        """Append default prefix"""
        if not system_msgs:
            system_msgs = []
        if self.prefix:
            system_msgs.append(self.prefix)
        return await self.llm.aask(prompt, system_msgs)

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
    async def _aask_v1(self, prompt: str, output_class_name: str,
                       output_data_mapping: dict,
                       system_msgs: Optional[list[str]] = None) -> ActionOutput:
        """Append default prefix"""
        if not system_msgs:
            system_msgs = []
        if self.prefix:
            system_msgs.append(self.prefix)
        content = await self.llm.aask(prompt, system_msgs)
        logger.debug(content)
        output_class = ActionOutput.create_model_class(output_class_name, output_data_mapping)
        parsed_data = OutputParser.parse_data_with_mapping(content, output_data_mapping)
        logger.debug(parsed_data)
        instruct_content = output_class(**parsed_data)
        return ActionOutput(content, instruct_content)


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
        if self.prefix:
            system_msgs.insert(0, self.prefix)
        
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
        content = await self.llm.aask(prompt, system_msgs, functions=functions, function_call=function_call)
        logger.debug(content)
        if isinstance(content, str):
            return ActionOutput(content, None)
        assert content['name'] in schema_dict.keys(), f"Function name {content['name']} is not in schema_dict {schema_dict.keys()}"
        try:
            args = parse_special_json(content['arguments'])
        except json.JSONDecodeError:
            logger.error(f"Failed to parse arguments: {content['arguments']}")
            raise
        arguments = schema_dict[content['name']].parse_obj(args)
        return ActionOutput(json.dumps(args), arguments)


    @staticmethod
    def get(func: Union[callable, BaseModel]):
        if func in _converted_actions:
            return _converted_actions[func]
        return Action.create(func)

    @staticmethod
    def create(func: Union[callable, BaseModel]):
        if func in _converted_actions:
            return _converted_actions[func]
        if inspect.isclass(func) and issubclass(func, BaseModel):
            Schema = func

            def _func(arg: Schema)->Schema:
                return arg
            # TODO: This means the docstring will be appeared both in the user context and the functions
            _func.__doc__ = Schema.__doc__
            _func.__name__ = Schema.__name__

            ret = _func_to_action(_func)
        else:
            ret = _func_to_action(func)
            
        _converted_actions[func] = ret
        return ret



    async def run(self, context: Union[str, List[ActionOutput]]):
        if isinstance(context, str):
            context = [ActionOutput(content=context, instruct_content=None)]
        assert context and isinstance(context, list)
        if len(context) == 1 and context[0].instruct_content:
            if context[0].instruct_content.__class__ in self._output_schemas:
                return context[0]

        messages = []
        schemas = []
        if len(self._input_schemas) == 0:
            for msg in context:
                if not msg.instruct_content:
                    messages.append({"role": "user", "content": msg.content})
                    break
        else:
            assert len(self._input_schemas) == len(context)
            for input_schema in self._input_schemas:
                for msg in context:
                    if isinstance(msg.instruct_content, input_schema):
                        messages.append({
                            "role": "function",
                            "name": msg.instruct_content.__class__.__name__,
                            "content": msg.instruct_content.json()
                        })
                        break
                schemas.append(input_schema)
        for output_schema in self._output_schemas:
            schemas.append(output_schema)
        s_names = [s.__name__ for s in schemas]
        assert len(schemas) == len(set(s_names)), f"Schema name must be unique, but got {s_names}"
        if len(self._output_schemas) == 0:
            suffix = self.desc
            function_call = "none"
        elif len(self._output_schemas) == 1:
            suffix = self.desc + f"\nIMPORTANT: You must use the `{self._output_schemas[0].__name__}` function to respond."
            function_call = {"name": self._output_schemas[0].__name__}
        else:
            suffix = self.desc + f"\nIMPORTANT: You must use one of the following functions to respond: {', '.join([s.__name__ for s in self._output_schemas])}"
            function_call = "auto"
        self._system_prompt = (self._system_prompt + '\n' + suffix) if self._system_prompt else suffix
        system_msgs = self._system_prompt and [self._system_prompt]
        logger.debug(self._system_prompt)
        response = await self._aask_v2(messages, system_msgs=system_msgs, schemas=schemas, function_call=function_call)
        logger.debug(response)
        return response
    
    
