from schema_agents.utils.schema_conversion import extract_tool_schemas
import inspect
from typing import get_args, get_origin, Union
from makefun import with_signature
from inspect import Signature, Parameter
from pydantic_core import PydanticUndefined

def schema_tool(tool_func, input_model=None):
    """Decorator for tool functions."""
    assert callable(tool_func)
    assert tool_func.__doc__ is not None
    if input_model:
        input_schema, output_schema = input_model, None
        parameters = []
        for name, field in input_model.model_fields.items():
            parameters.append(Parameter(name, kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=field.annotation, default=field))
        func_sig = Signature(parameters)
    else:
        input_schema, output_schema = extract_tool_schemas(tool_func)
        func_sig = inspect.signature(tool_func)
    assert input_schema is not None
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


    @with_signature(func_sig, func_name=tool_func.__name__, doc=tool_func.__doc__)
    async def wrapper(*args, **kwargs):
        for req in required:
            assert req in kwargs, f"Tool function `{tool_func.__name__}` missing required argument `{req}`"
        for k in default_factories:
            if k not in kwargs:
                kwargs[k] = default_factories[k]()
        for k in defaults:
            if k not in kwargs:
                kwargs[k] = defaults[k]
        ret = tool_func(*args, **kwargs)
        if inspect.isawaitable(ret):
            return await ret
        return ret

    wrapper.input_model = input_schema
    wrapper.output_model= output_schema
    wrapper.original_function = tool_func
    wrapper.__is_tool__ = True
    return wrapper
