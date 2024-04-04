"""Provide conversion functions for OpenAPI, JSON, and function schemas."""
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from inspect import signature
import inspect

from typing import get_origin, get_args, Union, Any


from openapi_pydantic import OpenAPI
from openapi_pydantic.util import PydanticSchema, construct_open_api_with_schema_class
from pydantic import BaseModel, Field
from schema_agents.utils import schema_to_function, dict_to_pydantic_model

def extract_tool_schemas(func, func_name=None):
    assert callable(func), "Tool function must be callable functions"
    sig = signature(func)
    # handle partial functions
    func_name = func.__name__ if hasattr(func, "__name__") else func.func.__name__ + "(partial)"
    names = [p.name for p in sig.parameters.values()]
    for name in names:
        assert sig.parameters[name].annotation != inspect._empty, f"Argument `{name}` for `{func_name}` must have type annotation"
    types = [sig.parameters[name].annotation for name in names]
    defaults = []
    for name in names:
        if sig.parameters[name].default == inspect._empty:
            defaults.append(Field(..., description=name))
        else:
            assert isinstance(sig.parameters[name].default, FieldInfo), "Argument default must be a FieldInfo object with description"
            assert sig.parameters[name].default.description is not None, f"Argument `{name}` for `{func_name}` must have a description"
            defaults.append(sig.parameters[name].default)
    
    func_name = func_name or func.__name__
    return (
        dict_to_pydantic_model(
            func_name,
            {names[i]: (types[i], defaults[i]) for i in range(len(names))},
            func.__doc__,
        ),
        sig.return_annotation,
    )


def get_primitive_schema(type_, is_json_schema=False):
    """Maps Python types to JSON schema and OpenAPI schema types."""
    if type_ is str:
        return {"type": "string"}
    elif type_ is int:
        return {"type": "integer"}
    elif type_ is float:
        return {"type": "number"}
    elif type_ is bool:
        return {"type": "boolean"}
    elif type_ is Any or type_ == inspect._empty:
        return {}
    elif is_json_schema:
        # For converting to json schema
        if type_ is None:
            return {"type": "null"}
        elif inspect.isclass(type_) and issubclass(type_, BaseModel):
            return type_.model_json_schema()
        else:
            origin = get_origin(type_)
            if origin is list:
                return {"type": "array", "items": get_primitive_schema(get_args(type_)[0], is_json_schema)}
            elif origin is dict:
                return {"type": "object", "additionalProperties": get_primitive_schema(get_args(type_)[1], is_json_schema)}
            elif origin is Union:
                types = get_args(type_)
                if len(types) == 2 and types[1] is type(None):
                    return {"type": get_primitive_schema(types[0], is_json_schema), "null": True}
                else:
                    return {"anyOf": [get_primitive_schema(t, is_json_schema) for t in types]}
            else:
                raise ValueError(f"Unsupported type: {type_}")
    else:
        # For converting to openapi schema
        if type_ is None:
            return {"nullable": True}
        elif inspect.isclass(type_) and issubclass(type_, BaseModel):
            return PydanticSchema(schema_class=type_)
        else:
            origin = get_origin(type_)
            if origin is list:
                return {"type": "array", "items": get_primitive_schema(get_args(type_)[0], is_json_schema)}
            elif origin is dict:
                return {"type": "object", "additionalProperties": get_primitive_schema(get_args(type_)[1], is_json_schema)}
            elif origin is Union:
                types = get_args(type_)
                if len(types) == 2 and types[1] is type(None):
                    return {"type": get_primitive_schema(types[0], is_json_schema), "nullable": True}
                else:
                    return {"anyOf": [get_primitive_schema(t, is_json_schema) for t in types]}
            else:
                raise ValueError(f"Unsupported type: {type_}")

def create_function_openapi_schema(func, func_name=None, method="post"):
    func_name = func_name or func.__name__
    input_schema, output_schema = extract_tool_schemas(func, func_name=func_name)
    output_schema_type = get_primitive_schema(output_schema, is_json_schema=False)
    return {
        method: {
            "description": func.__doc__,
            "operationId": func_name,
            "requestBody": {
                "content": {"application/json": {
                    "schema": PydanticSchema(schema_class=input_schema)
                }},
            },
            "responses": {"200": {
                "description": "Successful response",
                "content": {"application/json": {
                    "schema": output_schema_type if output_schema_type else None
                }},
            }},
        }
    }

def create_function_json_schema(func, func_name=None):
    func_name = func_name or func.__name__
    input_schema, output_schema = extract_tool_schemas(func, func_name=func_name)
    output_schema_type = get_primitive_schema(output_schema, is_json_schema=True)
    return input_schema.schema(), output_schema_type

def get_service_functions(service_config):
    functions = {}
    
    def extract_functions(config, path=""):
        for key, value in config.items():
            if isinstance(value, dict):
                extract_functions(value, path + key + ".")
            elif callable(value):
                functions[path + key] = value
    
    extract_functions(service_config)
    return functions

def get_service_openapi_schema(service_config, service_url="/"):
    functions = get_service_functions(service_config)
    paths = {}
    for path, func in functions.items():
        paths[f"/{path}"] = create_function_openapi_schema(func, func_name=path.replace(".", "_"))
    
    open_api = OpenAPI.model_validate({
        "info": {"title": service_config['name'], "version": "v0.1.0"},
        "servers": [{"url": service_url or "/"}],
        "paths": paths,
    })
    open_api = construct_open_api_with_schema_class(open_api)

    # Return the generated OpenAPI schema in JSON format
    return open_api.model_dump(by_alias=True, exclude_none=True)


def get_service_json_schema(service_config):
    functions = get_service_functions(service_config)
    schemas = {}

    for path, func in functions.items():
        input_schema, output_schema = create_function_json_schema(func, func_name=path.replace(".", "_"))
        schemas[path] = {
            "input_schema": input_schema,
            "output_schema": output_schema
        }
    return schemas


def get_service_function_schema(service_config):
    functions = get_service_functions(service_config)
    function_schemas = []

    for path, func in functions.items():
        input_schema, _ = extract_tool_schemas(func, func_name=path)
        function_schemas.append({"type": "function", "function": schema_to_function(input_schema)})
    return function_schemas
