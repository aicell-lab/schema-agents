#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : the implement of serialization and deserialization

import copy
from typing import Tuple, Dict, List, Type
from pydantic import BaseModel, create_model, root_validator, validator
import pickle

from schema_agents.schema import Message, MemoryChunk


def actionoutout_schema_to_mapping(schema: Dict) -> Dict:
    """
    directly traverse the `properties` in the first level.
    schema structure likes
    ```
    {
        "title":"prd",
        "type":"object",
        "properties":{
            "Original Requirements":{
                "title":"Original Requirements",
                "type":"string"
            },
        },
        "required":[
            "Original Requirements",
        ]
    }
    ```
    """
    mapping = dict()
    for field, property in schema['properties'].items():
        if property['type'] == 'string':
            mapping[field] = (str, ...)
        elif property['type'] == 'array' and property['items']['type'] == 'string':
            mapping[field] = (List[str], ...)
        elif property['type'] == 'array' and property['items']['type'] == 'array':
            # here only consider the `Tuple[str, str]` situation
            mapping[field] = (List[Tuple[str, str]], ...)
    return mapping

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

def serialize_message(message: Message):
    message_cp = copy.deepcopy(message)  # avoid `instruct_content` value update by reference
    ic = message_cp.instruct_content
    if ic:
        # model create by pydantic create_model like `pydantic.main.prd`, can't pickle.dump directly
        schema = ic.schema()
        mapping = actionoutout_schema_to_mapping(schema)

        message_cp.instruct_content = {
            'class': schema['title'],
            'mapping': mapping,
            'value': ic.dict()
        }
    msg_ser = pickle.dumps(message_cp)

    return msg_ser


def deserialize_message(message_ser: str) -> Message:
    message = pickle.loads(message_ser)
    if message.instruct_content:
        ic = message.instruct_content
        ic_obj = ActionOutput.create_model_class(class_name=ic['class'],
                                                 mapping=ic['mapping'])
        ic_new = ic_obj(**ic['value'])
        message.instruct_content = ic_new

    return message


def serialize_memory(memory: MemoryChunk):
    memory_cp = copy.deepcopy(memory)  # avoid `instruct_content` value update by reference
    ic = memory_cp.content
    if ic:
        # model create by pydantic create_model like `pydantic.main.prd`, can't pickle.dump directly
        schema = ic.schema()
        mapping = actionoutout_schema_to_mapping(schema)

        memory_cp.content = {
            'class': schema['title'],
            'mapping': mapping,
            'value': ic.dict()
        }
    mmr_ser = pickle.dumps(memory_cp)

    return mmr_ser


def deserialize_memory(mmr_ser: str) -> MemoryChunk:
    memory = pickle.loads(mmr_ser)
    if memory.content:
        ic = memory.content
        ic_obj = ActionOutput.create_model_class(class_name=ic['class'],
                                                 mapping=ic['mapping'])
        ic_new = ic_obj(**ic['value'])
        memory.content = ic_new

    return memory