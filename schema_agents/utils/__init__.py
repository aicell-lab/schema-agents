#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/29 15:50
@Author  : alexanderwu
@File    : __init__.py
"""

import os
import subprocess
import tempfile
import re
import json

from pydantic import BaseModel, create_model



def convert_key_name(key_name):
    words = key_name.split('_')
    capitalized_words = [word.capitalize() for word in words]
    return ' '.join(capitalized_words)

def dict_to_md(dict_obj):
    md_string = ""
    for key, value in dict_obj.items():
        md_string += f"\n## {convert_key_name(key)}\n\n"
        if isinstance(value, list):
            for item in value:
                if isinstance(item, tuple):
                    item = ', '.join(item)
                md_string += f"- {item}\n"
        else:
            md_string += f"{value}\n"
    return md_string

def apply_patch(original_text, patch_text):
    # Create a temporary file to hold the original code
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as original_file:
        original_file.write(original_text)
        original_path = original_file.name

    # Create a temporary file to hold the patch
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as patch_file:
        patch_file.write(patch_text)
        patch_path = patch_file.name

    # Use the patch command to apply the patch
    result = subprocess.run(['patch', original_path, patch_path], capture_output=True)

    # Read the patched content from the original file
    with open(original_path, 'r', encoding="utf-8") as file:
        patched_text = file.read()

    # Clean up the temporary files
    os.unlink(original_path)
    os.unlink(patch_path)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to apply patch: {result.stdout and result.stdout.decode()}\n{result.stderr and result.stderr.decode()}")
    else:
        return patched_text


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

# https://stackoverflow.com/a/58938747
def remove_a_key(d, remove_key):
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == remove_key:
                del d[key]
            else:
                remove_a_key(d[key], remove_key)

def schema_to_function(schema: BaseModel):
    assert schema.__doc__, f"{schema.__name__} is missing a docstring."
    assert (
        "title" not in schema.model_fields.keys()
    ), "`title` is a reserved keyword and cannot be used as a field name."
    schema_dict = schema.model_json_schema()
    remove_a_key(schema_dict, "title")

    return {
        "name": schema.__name__,
        "description": schema.__doc__,
        "parameters": schema_dict,
    }


def dict_to_pydantic_model(name: str, dict_def: dict, doc: str = None):
    fields = {}
    for field_name, value in dict_def.items():
        if isinstance(value, tuple):
            fields[field_name] = value
        elif isinstance(value, dict):
            fields[field_name] = (dict_to_pydantic_model(f"{name}_{field_name}", value), ...)
        else:
            raise ValueError(f"Field {field_name}:{value} has invalid syntax")
    model = create_model(name, **fields)
    model.__doc__ = doc
    return model
