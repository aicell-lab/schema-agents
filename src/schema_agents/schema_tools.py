from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, get_type_hints
import inspect

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from schema_agents.tools import Tool

T = TypeVar('T')


def schema_tool(
    func=None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
):
    """Decorator for schema tools.
    
    This decorator allows using Pydantic Field for parameter descriptions and default values.
    The Field metadata is automatically extracted to create a properly documented tool.
    
    Args:
        func: The function to decorate
        name: Optional custom name for the tool (defaults to function name)
        description: Optional custom description (defaults to function docstring)
    
    Example:
        @schema_tool
        async def say_hello(
            name: str = Field(default="alice", description="Name of the person to greet")
        ) -> str:
            '''Say hello to someone'''
            return f"Hello {name}!"
            
        @schema_tool(name="calculator")
        async def calculate(
            x: int = Field(..., description="First number"),
            y: int = Field(default=10, description="Second number"),
            operation: str = Field(default="add", description="Operation type")
        ) -> str:
            '''Perform a calculation'''
            return f"{x} {operation} {y}"
    """
    def decorator(func):
        # Extract function signature and docstring
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""
        
        # Determine if the function is async
        is_async = inspect.iscoroutinefunction(func)
        
        # Determine tool name
        tool_name = name or func.__name__
        
        # Determine tool description
        tool_description = description or doc
        
        # Create inputs dictionary for the tool
        tool_inputs = {}
        for param_name, param in sig.parameters.items():
            # Skip self parameter if it exists
            if param_name == 'self':
                continue
                
            # Get parameter type
            param_type = param.annotation
            if param_type is inspect.Parameter.empty:
                param_type = Any
                
            # Determine type string for the tool input
            type_str = _get_type_string(param_type)
            
            # Extract Field metadata if present
            param_description = f"Parameter: {param_name}"
            is_required = True
            default_value = None
            
            if param.default is not inspect.Parameter.empty:
                if isinstance(param.default, FieldInfo):
                    field_info = param.default
                    if field_info.description:
                        param_description = field_info.description
                    # Check if the field is required
                    # For Pydantic v2, we need to check if the default is a special value
                    # that indicates a required field
                    is_required = str(field_info.default) == "PydanticUndefined" or field_info.default is ...
                    if not is_required and field_info.default is not None:
                        default_value = field_info.default
                else:
                    is_required = False
                    default_value = param.default
            
            # Create the input definition
            tool_inputs[param_name] = {
                "type": type_str,
                "description": param_description,
            }
            
            # Add nullable only if parameter is optional
            if not is_required:
                tool_inputs[param_name]["nullable"] = True
        
        # Create the tool class
        class SchemaTool(Tool):
            name = tool_name
            description = tool_description
            inputs = tool_inputs
            output_type = _get_type_string(sig.return_annotation)
            # Skip forward signature validation since we're using a generic forward method
            skip_forward_signature_validation = True
            # Always mark schema tools as async for consistent usage with await
            is_async = True
            
            def __init__(self):
                super().__init__()
                self.is_initialized = True
                self.original_is_async = is_async
            
            async def __call__(self, *args, **kwargs):
                """Override the __call__ method to ensure the forward method is properly awaited."""
                return await self.forward(*args, **kwargs)
                
            async def forward(self, *args, **kwargs):
                # Clean up kwargs to handle Field instances
                clean_kwargs = {}
                
                # Process positional arguments
                if args:
                    param_names = [p for p in sig.parameters.keys() if p != 'self']
                    for i, arg in enumerate(args):
                        if i < len(param_names):
                            clean_kwargs[param_names[i]] = arg
                
                # Process keyword arguments
                for name, param in sig.parameters.items():
                    if name == 'self':
                        continue
                        
                    if name in kwargs:
                        clean_kwargs[name] = kwargs[name]
                    elif name not in clean_kwargs and param.default is not inspect.Parameter.empty:
                        if isinstance(param.default, FieldInfo):
                            if param.default.default is not ...:
                                clean_kwargs[name] = param.default.default
                        else:
                            clean_kwargs[name] = param.default
                
                # Call the original function
                result = func(**clean_kwargs)
                if inspect.isawaitable(result):
                    result = await result
                return result
        
        # Create an instance of the tool
        tool_instance = SchemaTool()
        
        # Add the tool to the function as an attribute
        func.__tool__ = tool_instance
        
        # Return the original function
        return func
    
    if func is None:
        return decorator
    return decorator(func)


def create_tool_from_schema_function(func: Callable) -> Tool:
    """Create a Tool instance from a function decorated with @schema_tool.
    
    Args:
        func: The function decorated with @schema_tool
        
    Returns:
        A Tool instance
        
    Raises:
        ValueError: If the function is not decorated with @schema_tool
    """
    if hasattr(func, "__tool__"):
        return func.__tool__
    else:
        raise ValueError(f"Function {func.__name__} is not decorated with @schema_tool")


def _get_type_string(type_hint) -> str:
    """Convert Python type hints to tool type strings."""
    if type_hint is inspect.Parameter.empty:
        return "any"
    
    # Handle basic types
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        None: "null",
        type(None): "null",
    }
    
    if type_hint in type_mapping:
        return type_mapping[type_hint]
    
    # Handle return annotation
    if type_hint is inspect._empty:
        return "any"
    
    # Try to get the name of the type
    try:
        type_name = type_hint.__name__.lower()
        if type_name in ["str", "int", "float", "bool"]:
            return type_mapping[eval(type_name)]
        return "object"
    except (AttributeError, NameError):
        return "object"
