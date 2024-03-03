from schema_agents.utils.schema_conversion import extract_tool_schemas
from functools import wraps
from typing import get_args, get_origin, Union
try:
    from pydantic_core import PydanticUndefined
except ImportError:
    from pydantic.fields import Undefined as PydanticUndefined

def tool(tool_func):
    """Decorator for tool functions."""
    assert callable(tool_func)
    assert tool_func.__doc__ is not None
    input_schema, output_schema = extract_tool_schemas(tool_func)
    assert input_schema is not None
    assert output_schema is not None
    defaults = {}
    default_factories = {}
    required = []
    for name, field in input_schema.model_fields.items():
        # check if field has no default value, and no default_factory
        if field.default != PydanticUndefined or field.default_factory is not None:
            if field.default_factory is not None:
                default_factories[name] = field.default_factory
            else:
                defaults[name] = field.default
        else:
            # check field.annotation is typing.Optional
            if get_origin(field.annotation) is Union and type(None) in get_args(field.annotation):
                defaults[name] = None
            else:
                required.append(name)

    @wraps(tool_func)
    def wrapper(*args, **kwargs):
        for req in required:
            assert req in kwargs, f"Tool function `{tool_func.__name__}` missing required argument `{req}`"
        for k in default_factories:
            if k not in kwargs:
                kwargs[k] = default_factories[k]()
        for k in defaults:
            if k not in kwargs:
                kwargs[k] = defaults[k]
        return tool_func(*args, **kwargs)

    wrapper.input_model = input_schema
    wrapper.output_model= output_schema
    wrapper.original_function = tool_func
    wrapper.__is_tool__ = True
    return wrapper
