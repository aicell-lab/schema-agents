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

from schema_agents.utils.singleton import Singleton
from schema_agents.utils.token_counter import (
    TOKEN_COSTS,
    count_message_tokens,
    count_string_tokens,
)



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
    with open(original_path, 'r') as file:
        patched_text = file.read()

    # Clean up the temporary files
    os.unlink(original_path)
    os.unlink(patch_path)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to apply patch: {result.stdout and result.stdout.decode()}\n{result.stderr and result.stderr.decode()}")
    else:
        return patched_text