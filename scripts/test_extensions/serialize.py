import json
from typing import Any
import pickle as pkl
import sys
import ast

def get_docstring_from_async_function(async_function_node):
    """
    Extracts the docstring from an ast.AsyncFunctionDef node if it exists.
    
    :param async_function_node: An instance of ast.AsyncFunctionDef
    :return: The docstring as a string if found, otherwise None
    """
    # Check if the async function has a body and the first element of the body is an expression
    if async_function_node.body and isinstance(async_function_node.body[0], ast.Expr):
        # The first element of the body could be a docstring (as an expression)
        first_element = async_function_node.body[0]
        # Depending on the Python version, the docstring can be in ast.Str (Python <3.8) or ast.Constant (Python >=3.8)
        if hasattr(first_element, 'value') and isinstance(first_element.value, (ast.Str, ast.Constant)):
            return first_element.value.s  # Extract the string value
    return "No docstring provided"  # No docstring found

def ast_to_dict(node):
    """Convert an AST node to a dictionary."""
    if isinstance(node, ast.AsyncFunctionDef):
        return {
            'type': 'AsyncFunctionDef',
            'name': node.name,
            'args': [arg.arg for arg in node.args.args],  # Simplified; doesn't handle defaults or varargs
            'docstring': get_docstring_from_async_function(node),
            # 'body': [ast_to_dict(child) for child in node.body],  # Recursive call
        }
    elif isinstance(node, ast.Expr):
        if isinstance(node.value, ast.Call):
            return {
                'type': 'FunctionCall',
                'func': getattr(node.value.func, 'id', repr(node.value.func)),  # Function name or repr
                'args': [ast_to_dict(arg) for arg in node.value.args],  # Recursive call
            }
    elif isinstance(node, ast.Await):
        return {
            'type': 'Await',
            'value': ast_to_dict(node.value)  # Recursive call
        }
    # Add more elif blocks here to handle other node types as needed
    else:
        return repr(node)  # Fallback representation

def convert_to_serializable(data):
    if isinstance(data, bytes):
        return data.decode('utf-8')
    elif isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_to_serializable(item) for item in data)
    elif hasattr(data, 'dict'):  # Check if it's a Pydantic model instance
        return convert_to_serializable(data.dict())
    elif isinstance(data, (ast.AsyncFunctionDef, ast.Expr, ast.Await)):
        return ast_to_dict(data)
    return data

def dump_metadata_json(metadata, file_path: str):
    serialized_metadata = convert_to_serializable(metadata)
    with open(file_path, 'w') as f:
        json.dump(serialized_metadata, f)

# if __name__ == "__main__":
#     metadata = pkl.load(open("/Users/gkreder/schema-agents/test.pkl", "rb"))
#     # Assuming `steps` is your complex nested list that you've mentioned
#     steps_serializable = convert_to_serializable(metadata['steps'])
#     # Now you can safely dump this to a JSON file
#     with open('steps.json', 'w') as f:
#         json.dump(steps_serializable, f)
#     print(metadata)
