from __future__ import annotations

import asyncio
import copy
import dataclasses
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast, Callable, AsyncIterator

from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import RunContext, models, result, exceptions
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.tools import AgentDepsT, Tool, ToolFuncEither, ToolDefinition
from pydantic.fields import PydanticUndefined

ResultDataT = TypeVar('ResultDataT')
RunResultDataT = TypeVar('RunResultDataT')

def _build_system_prompt(role: str | None, goal: str | None, backstory: str | None) -> str:
    """Build a system prompt from role, goal and backstory."""
    parts = []
    if role:
        parts.append(f"Role: {role}")
    if goal:
        parts.append(f"Goal: {goal}")
    if backstory:
        parts.append(f"Backstory: {backstory}")
    return "\n".join(parts)

@dataclasses.dataclass
class AgentConfig:
    """Configuration for Schema Agent."""
    role: str | None = None
    goal: str | None = None
    backstory: str | None = None
    reasoning_config: BaseModel | None = None

class Agent(PydanticAgent[AgentDepsT, ResultDataT]):
    """Schema Agents Platform Agent implementation.
    
    This extends the Pydantic AI Agent with additional capabilities:
    - Runtime tool attachment
    - Dynamic reasoning strategies
    - Improved concurrency safety
    - Vector memory integration
    """
    def __init__(
        self,
        model: models.Model | models.KnownModelName | None = None,
        *,
        name: str | None = None,
        result_type: Type[ResultDataT] = str,
        deps_type: Type[AgentDepsT] = type(None),
        config: AgentConfig | None = None,
        **kwargs
    ):
        """Initialize a Schema Agent.
        
        Args:
            model: The LLM model to use
            name: Agent name
            result_type: Expected result type
            deps_type: Dependencies type
            config: Optional agent configuration
            **kwargs: Additional arguments passed to PydanticAgent
        """
        # Create config from kwargs if not provided
        if config is None:
            config = AgentConfig(
                role=kwargs.pop('role', None),
                goal=kwargs.pop('goal', None),
                backstory=kwargs.pop('backstory', None),
                reasoning_config=kwargs.pop('reasoning_config', None)
            )

        # Build system prompt from config
        system_prompt = _build_system_prompt(config.role, config.goal, config.backstory)
        if system_prompt:
            kwargs['system_prompt'] = system_prompt

        # Initialize parent class
        super().__init__(
            model=model,
            name=name,
            result_type=result_type,
            deps_type=deps_type,
            **kwargs
        )
        
        # Store config
        self.config = config
        
        # Initialize message history
        self._message_history: List[Dict[str, Any]] = []

    async def run(
        self,
        user_prompt: str,
        *,
        result_type: Type[RunResultDataT] | None = None,
        message_history: List[Dict[str, Any]] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDepsT = None,
        tools: List[Union[Tool[AgentDepsT], ToolFuncEither[AgentDepsT, Any]]] | None = None,
        **kwargs
    ) -> result.RunResult[Any]:
        """Run the agent with optional runtime tools.
        
        This overrides the base run() to support:
        1. Runtime tool attachment
        2. Safe concurrent execution
        3. Reasoning strategy integration
        """
        # Create a copy of the agent to avoid concurrent modification
        agent_copy = self._fork_agent()
        
        # Use internal message history if none provided
        if message_history is None:
            message_history = self._message_history
        
        # Attach runtime tools if provided
        if tools:
            for tool in tools:
                if hasattr(tool, '__tool__'):
                    agent_copy._register_tool(tool.__tool__)
                elif isinstance(tool, Tool):
                    agent_copy._register_tool(tool)
                else:
                    agent_copy._register_tool(Tool(tool))
        
        try:
            # Run with the copied agent
            result = await super(Agent, agent_copy).run(
                user_prompt,
                result_type=result_type,
                message_history=message_history,
                model=model,
                deps=deps,
                **kwargs
            )

            # Validate result type
            if result_type:
                try:
                    if not isinstance(result.data, result_type):
                        # Try to parse the result as the expected type
                        if issubclass(result_type, BaseModel):
                            result.data = result_type.model_validate(result.data)
                        else:
                            raise exceptions.UserError(f"Invalid response type: expected {result_type.__name__}, got {type(result.data).__name__}")
                except ValidationError as e:
                    raise exceptions.UserError(f"Invalid response format: {str(e)}")

            # Update history if deps has history attribute
            if deps and hasattr(deps, 'history'):
                await deps.add_to_history(user_prompt)
                if isinstance(result.data, str):
                    await deps.add_to_history(result.data)
                else:
                    await deps.add_to_history(str(result.data))

            # Update internal message history
            self._message_history.extend(result._all_messages)

            return result
        except Exception as e:
            # Wrap exceptions to ensure consistent error handling
            if isinstance(e, exceptions.ModelRetry):
                raise e
            if isinstance(e, (ValueError, TypeError, ValidationError)):
                # Convert validation errors to UserError
                raise exceptions.UserError(str(e)) from e
            raise e

    def run_sync(
        self,
        user_prompt: str,
        *,
        result_type: Type[RunResultDataT] | None = None,
        message_history: List[Dict[str, Any]] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDepsT = None,
        tools: List[Union[Tool[AgentDepsT], ToolFuncEither[AgentDepsT, Any]]] | None = None,
        **kwargs
    ) -> result.RunResult[Any]:
        """Synchronous version of run()"""
        return asyncio.get_event_loop().run_until_complete(
            self.run(
                user_prompt,
                result_type=result_type,
                message_history=message_history,
                model=model,
                deps=deps,
                tools=tools,
                **kwargs
            )
        )

    def _fork_agent(self) -> Agent[AgentDepsT, ResultDataT]:
        """Create a copy of the agent with independent tool state.
        
        This ensures concurrent runs don't interfere with each other.
        """
        # Create a shallow copy
        agent_copy = copy.copy(self)
        
        # Deep copy mutable state
        agent_copy._function_tools = copy.deepcopy(self._function_tools)
        agent_copy._message_history = copy.deepcopy(self._message_history)
        
        return agent_copy

    def schema_tool(
        self,
        func=None,
        *,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        takes_ctx: bool | None = None,
    ):
        """Decorator for schema tools.
        
        This decorator allows using Pydantic Field for parameter descriptions and default values.
        The Field metadata is automatically extracted to create a properly documented tool.
        
        Args:
            func: The function to decorate
            retries: The number of retries to allow for this tool, defaults to the agent's default retries
            prepare: Custom method to prepare the tool definition for each step, return None to omit this
                tool from a given step. See ToolPrepareFunc for more details.
            takes_ctx: Whether the function takes RunContext as its first argument. If None, will be inferred.
        
        Example:
            @agent.schema_tool
            async def say_hello(
                name: Optional[str] = Field(default="alice", description="Name of the person to greet")
            ) -> str:
                '''Say hello to someone'''
                return f"Hello {name}!"
                
            @agent.schema_tool(retries=3)
            async def calculate(
                x: int = Field(..., description="First number"),
                y: int = Field(default=10, description="Second number"),
                operation: str = Field(default="add", description="Operation type")
            ) -> str:
                '''Perform a calculation'''
                return f"{x} {operation} {y}"
        """
        from inspect import Parameter, signature
        from pydantic.fields import FieldInfo
        from typing import get_args, get_origin, Union
        import inspect
        import functools
        from pydantic_ai.tools import ToolDefinition

        async def prepare_tool(ctx: RunContext[AgentDepsT], tool_name: str) -> Dict[str, Any]:
            """Extract Field metadata to create tool schema"""
            sig = signature(func)
            tool_schema = {
                "name": tool_name,
                "description": func.__doc__ or "",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            
            for name, param in sig.parameters.items():
                # Skip ctx parameter if takes_ctx is True
                if name == 'ctx' and takes_ctx:
                    continue
                    
                # Get parameter type
                param_type = param.annotation
                if get_origin(param_type) is Union:
                    # Handle Optional types
                    args = get_args(param_type)
                    if type(None) in args:
                        param_type = next(arg for arg in args if arg is not type(None))

                # Add parameter to schema
                param_schema = {
                    "type": param_type.__name__.lower(),
                }

                # Extract Field metadata if present
                if param.default is not Parameter.empty:
                    if isinstance(param.default, FieldInfo):
                        field_info = param.default
                        if field_info.description:
                            param_schema["description"] = field_info.description
                        if field_info.default is not ... and field_info.default is not PydanticUndefined:
                            param_schema["default"] = field_info.default
                        if field_info.default is ... or field_info.default is PydanticUndefined:
                            tool_schema["parameters"]["required"].append(name)
                    else:
                        param_schema["default"] = param.default
                else:
                    tool_schema["parameters"]["required"].append(name)

                tool_schema["parameters"]["properties"][name] = param_schema

            return tool_schema

        async def wrapper(*args, **kwargs):
            """Remove Field instances from kwargs and apply defaults"""
            sig = signature(func)
            clean_kwargs = {}
            
            # Get default values from Field instances in signature
            for name, param in sig.parameters.items():
                if name in kwargs:
                    clean_kwargs[name] = kwargs[name]
                elif param.default is not Parameter.empty:
                    if isinstance(param.default, FieldInfo):
                        if param.default.default is not ... and param.default.default is not PydanticUndefined:
                            clean_kwargs[name] = param.default.default
                    else:
                        clean_kwargs[name] = param.default
            
            # Call original function
            result = func(*args, **clean_kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        if func is None:
            return lambda f: self.schema_tool(f, retries=retries, prepare=prepare, takes_ctx=takes_ctx)

        # Copy function metadata
        functools.update_wrapper(wrapper, func)

        # Create a prepare function that combines our schema preparation with any custom prepare function
        async def combined_prepare(ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> ToolDefinition | None:
            # First get our schema
            schema = await prepare_tool(ctx, tool_def.name)
            
            # Create a new tool definition with our schema
            new_tool_def = ToolDefinition(
                name=schema["name"],
                description=schema["description"],
                parameters_json_schema=schema["parameters"]
            )
            
            # If there's a custom prepare function, call it with our tool definition
            if prepare is not None:
                return await prepare(ctx, new_tool_def)
            return new_tool_def

        # Create tool with prepare function to handle Field metadata
        tool = Tool(
            wrapper,
            prepare=combined_prepare,
            takes_ctx=takes_ctx if takes_ctx is not None else False,
            max_retries=retries,
        )
        
        # Register the tool
        self._register_tool(tool)
        wrapper.__tool__ = tool
        
        return wrapper

    @asynccontextmanager
    async def run_stream(
        self,
        user_prompt: str,
        *,
        result_type: Type[RunResultDataT] | None = None,
        message_history: List[Dict[str, Any]] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDepsT = None,
        tools: List[Union[Tool[AgentDepsT], ToolFuncEither[AgentDepsT, Any]]] | None = None,
        **kwargs
    ) -> AsyncIterator[result.StreamedRunResult[AgentDepsT, Any]]:
        """Run the agent with streaming output and optional runtime tools."""
        # Create a copy of the agent to avoid concurrent modification
        agent_copy = self._fork_agent()
        
        # Use internal message history if none provided
        if message_history is None:
            message_history = self._message_history
        
        # Attach runtime tools if provided
        if tools:
            for tool in tools:
                if hasattr(tool, '__tool__'):
                    agent_copy._register_tool(tool.__tool__)
                elif isinstance(tool, Tool):
                    agent_copy._register_tool(tool)
                else:
                    # Convert function to Tool
                    agent_copy._register_function(tool, False, None, None, 'auto', False)
        
        try:
            # Run with the copied agent
            async with super(Agent, agent_copy).run_stream(
                user_prompt,
                result_type=result_type,
                message_history=message_history,
                model=model,
                deps=deps,
                **kwargs
            ) as response:
                # Update history if deps has history attribute
                if deps and hasattr(deps, 'history'):
                    await deps.add_to_history(user_prompt)
                yield response

                # Validate result type
                if result_type:
                    try:
                        if not isinstance(response.data, result_type):
                            # Try to parse the result as the expected type
                            if issubclass(result_type, BaseModel):
                                response.data = result_type.model_validate(response.data)
                            else:
                                raise exceptions.UserError(f"Invalid response type: expected {result_type.__name__}, got {type(response.data).__name__}")
                    except ValidationError as e:
                        raise exceptions.UserError(f"Invalid response format: {str(e)}")

                # Update internal message history after streaming completes
                self._message_history.extend(response._all_messages)
        except Exception as e:
            # Wrap exceptions to ensure consistent error handling
            if isinstance(e, exceptions.ModelRetry):
                raise e
            if isinstance(e, (ValueError, TypeError, ValidationError)):
                # Convert validation errors to UserError
                raise exceptions.UserError(str(e)) from e
            raise e
