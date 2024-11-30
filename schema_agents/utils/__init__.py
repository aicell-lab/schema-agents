#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import subprocess
import tempfile
import re
import json
from typing import List, Dict, Any, Callable, Union

from pydantic import BaseModel, create_model
from collections import defaultdict
import inspect
import pydantic


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



def organize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Organizes and merges consecutive messages of the same type and ensures the system message is first.

    Args:
        messages (List[Dict[str, Any]]): A list of message dictionaries.

    Returns:
        List[Dict[str, Any]]: The organized and merged messages.
    """
    if not messages:
        return []

    # Separate the system message and others
    system_message = None
    non_system_messages = []

    for message in messages:
        if message["role"] == "system" and system_message is None:
            system_message = message
        else:
            non_system_messages.append(message)

    # Combine the system message with the rest
    organized_messages = [system_message] if system_message else []
    organized_messages += merge_consecutive_messages(non_system_messages)

    return organized_messages


def merge_consecutive_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merges consecutive messages of the same type.

    Args:
        messages (List[Dict[str, Any]]): A list of message dictionaries.

    Returns:
        List[Dict[str, Any]]: The merged messages.
    """
    if not messages:
        return []

    merged_messages = []
    current_message = None

    for message in messages:
        if not current_message:
            current_message = message
            continue

        # Merge consecutive messages based on role and content
        if (
            current_message["role"] == message["role"]
            and current_message["role"] != "tool"  # Don't merge tool role
            and ("tool_calls" in current_message) == ("tool_calls" in message)  # Match presence of tool_calls
        ):
            if "content" in current_message and "content" in message:
                current_message["content"] += message["content"]

            if "tool_calls" in current_message and "tool_calls" in message:
                current_message["tool_calls"] += message["tool_calls"]
        else:
            merged_messages.append(current_message)
            current_message = message

    if current_message:
        merged_messages.append(current_message)

    return merged_messages


def _parse_docstring(doc_string: Union[str, None]) -> dict[str, str]:
  """
  Parse the docstring of a function into a dictionary.
  This function was taken from ollama made by @Ollama team released under the MIT license.
  Source: https://github.com/ollama/ollama-python/blob/main/ollama/_utils.py
  """
  parsed_docstring = defaultdict(str)
  if not doc_string:
    return parsed_docstring

  key = hash(doc_string)
  for line in doc_string.splitlines():
    lowered_line = line.lower().strip()
    if lowered_line.startswith('args:'):
      key = 'args'
    elif lowered_line.startswith('returns:') or lowered_line.startswith('yields:') or lowered_line.startswith('raises:'):
      key = '_'

    else:
      # maybe change to a list and join later
      parsed_docstring[key] += f'{line.strip()}\n'

  last_key = None
  for line in parsed_docstring['args'].splitlines():
    line = line.strip()
    if ':' in line:
      # Split the line on either:
      # 1. A parenthetical expression like (integer) - captured in group 1
      # 2. A colon :
      # Followed by optional whitespace. Only split on first occurrence.
      parts = re.split(r'(?:\(([^)]*)\)|:)\s*', line, maxsplit=1)

      arg_name = parts[0].strip()
      last_key = arg_name

      # Get the description - will be in parts[1] if parenthetical or parts[-1] if after colon
      arg_description = parts[-1].strip()
      if len(parts) > 2 and parts[1]:  # Has parenthetical content
        arg_description = parts[-1].split(':', 1)[-1].strip()

      parsed_docstring[last_key] = arg_description

    elif last_key and line:
      parsed_docstring[last_key] += ' ' + line

  return parsed_docstring


def convert_function_to_tool(func: Callable) -> dict:
  """
  Convert a function to a tool schema.
  This function was taken from ollama made by @Ollama team released under the MIT license.
  Source: https://github.com/ollama/ollama-python/blob/main/ollama/_utils.py
  """
  doc_string_hash = hash(inspect.getdoc(func))
  parsed_docstring = _parse_docstring(inspect.getdoc(func))
  schema = type(
    func.__name__,
    (pydantic.BaseModel,),
    {
      '__annotations__': {k: v.annotation if v.annotation != inspect._empty else str for k, v in inspect.signature(func).parameters.items()},
      '__signature__': inspect.signature(func),
      '__doc__': parsed_docstring[doc_string_hash],
    },
  ).model_json_schema()

  for k, v in schema.get('properties', {}).items():
    # If type is missing, the default is string
    types = {t.get('type', 'string') for t in v.get('anyOf')} if 'anyOf' in v else {v.get('type', 'string')}
    if 'null' in types:
      schema['required'].remove(k)
      types.discard('null')

    schema['properties'][k] = {
      'description': parsed_docstring[k],
      'type': ', '.join(types),
    }

  return {
    "name": func.__name__,
    "description": schema.get('description', ''),
    "parameters": schema,
  }