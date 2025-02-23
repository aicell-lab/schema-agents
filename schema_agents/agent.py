from __future__ import annotations

import asyncio
import copy
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Type, TypeVar, Union, AsyncIterator
from inspect import Parameter, signature
from pydantic.fields import FieldInfo
from typing import get_args, get_origin, Union
import inspect
import functools

from pydantic_ai.tools import ToolDefinition
from pydantic import BaseModel, ValidationError, Field
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import RunContext, models, result, exceptions
from pydantic_ai.tools import ToolPrepareFunc, AgentDepsT, Tool, ToolFuncEither, ToolDefinition
from pydantic.fields import PydanticUndefined
from pydantic_ai.messages import ModelResponse, TextPart
from schema_agents.reasoning import ReasoningStrategy

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
        role: str | None = None,
        goal: str | None = None,
        backstory: str | None = None,
        reasoning_strategy: ReasoningStrategy | None = None,
        **kwargs
    ):
        """Initialize a Schema Agent.
        
        Args:
            model: The LLM model to use
            name: Agent name
            result_type: Expected result type
            deps_type: Dependencies type
            role: Agent role description
            goal: Agent goal description
            backstory: Agent backstory
            reasoning_strategy: Optional reasoning strategy configuration
            **kwargs: Additional arguments passed to PydanticAgent
        """
        # Build system prompt from config
        system_prompt = _build_system_prompt(role, goal, backstory)
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
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.reasoning_strategy = reasoning_strategy
        self._model = model  # Store the model instance
        self._deps_type = deps_type  # Store deps_type
        
        # Initialize message history
        self._message_history: List[Dict[str, Any]] = []

    @property
    def deps_type(self) -> Type[AgentDepsT]:
        """Get the dependencies type."""
        return self._deps_type

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
            # Check if we should use reasoning strategy
            if self.reasoning_strategy:
                strategy = self.reasoning_strategy
                model_used = self._get_model(model)
                usage_obj = result.Usage()
                run_context = RunContext(
                    deps=deps,
                    model=model_used,
                    usage=usage_obj,
                    prompt=user_prompt,
                    messages=message_history or [],
                    run_step=0
                )

                # Execute reasoning strategy
                if isinstance(strategy.type, str):
                    strategy_types = [strategy.type]
                else:
                    strategy_types = strategy.type

                final_answer = None
                for strategy_type in strategy_types:
                    if strategy_type == "react":
                        if not strategy.react_config:
                            raise ValueError("ReAct strategy selected but no react_config provided")
                        from schema_agents.reasoning import execute_react_reasoning
                        final_answer = await execute_react_reasoning(
                            user_prompt,
                            model_used,
                            list(agent_copy._function_tools.values()),
                            strategy,
                            run_context
                        )
                    # Add other strategy types here
                    else:
                        raise ValueError(f"Unknown strategy type: {strategy_type}")

                # Create a RunResult with the final answer
                run_result = result.RunResult(
                    message_history or [],
                    len(message_history) if message_history else 0,
                    final_answer,
                    None,  # No tool name for reasoning results
                    run_context.usage
                )
            else:
                # Run with the copied agent without reasoning strategy
                run_result = await super(Agent, agent_copy).run(
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
                    if not isinstance(run_result.data, result_type):
                        # Try to parse the result as the expected type
                        if issubclass(result_type, BaseModel):
                            run_result.data = result_type.model_validate(run_result.data)
                        else:
                            raise exceptions.UserError(f"Invalid response type: expected {result_type.__name__}, got {type(run_result.data).__name__}")
                except ValidationError as e:
                    raise exceptions.UserError(f"Invalid response format: {str(e)}")

            # Update history if deps has history attribute
            if deps and hasattr(deps, 'history'):
                await deps.add_to_history(user_prompt)
                if isinstance(run_result.data, str):
                    await deps.add_to_history(run_result.data)
                else:
                    await deps.add_to_history(str(run_result.data))

            # Update internal message history
            self._message_history.extend(run_result._all_messages)

            return run_result
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

    async def register(self, server, service_id: str | None = None):
        """Register the agent as a Hypha service.
        
        Args:
            server: The Hypha server instance to register with
            service_id: Optional service ID, defaults to agent name if not provided
        
        Returns:
            The registered service object
        """
        # Create service ID from agent name if not provided
        if not service_id:
            service_id = self.name.lower().replace(" ", "-")

        # Create a wrapper for the run method that handles both streaming and non-streaming
        deps_type = self.deps_type  # Store deps_type for closure
        async def run_wrapper(user_prompt: str, callback=None, *, deps=None, context=None, **kwargs):
            """Wrapper for run that handles dependencies and optional streaming"""
            # Convert dict to deps instance if needed
            if deps is not None:
                if isinstance(deps, dict):
                    deps = deps_type.model_validate(deps)
                elif hasattr(deps, 'model_dump'):
                    # If it's already a pydantic model, convert to dict and back to ensure proper validation
                    deps = deps_type.model_validate(deps.model_dump())
                elif hasattr(deps, '__dict__'):
                    # If it's a dataclass, convert to dict and back
                    deps = deps_type(**deps.__dict__)

            if callback:
                # Streaming mode
                async with self.run_stream(user_prompt, deps=deps, **kwargs) as response:
                    async for chunk in response._stream_response:
                        if isinstance(chunk, ModelResponse):
                            for part in chunk.parts:
                                if isinstance(part, TextPart):
                                    await callback({
                                        "type": "text",
                                        "content": part.content,
                                        "model": chunk.model_name
                                    })
                        else:
                            # For non-ModelResponse chunks (e.g. from reasoning strategies)
                            await callback({
                                "type": "text",
                                "content": str(chunk),
                                "model": self._model.model_name if self._model else None
                            })
                    result = await response.get_data()
                    if hasattr(result, 'model_dump'):
                        return result.model_dump()
                    return result
            else:
                # Non-streaming mode
                result = await self.run(user_prompt, deps=deps, **kwargs)
                if hasattr(result.data, 'model_dump'):
                    return result.data.model_dump()
                return result.data

        # Prepare tool metadata
        tools_meta = []
        if self._function_tools:
            for tool in self._function_tools.values():
                # Get tool definition
                tool_def = await tool.prepare_tool_def(None)
                if tool_def:
                    tools_meta.append({
                        "name": tool_def.name,
                        "description": tool_def.description,
                        "parameters": tool_def.parameters_json_schema
                    })

        # Register the service
        service_info = await server.register_service({
            "id": service_id,
            "name": self.name,
            "description": f"Agent service for {self.name} ({self.role})",
            "docs": f"Name: {self.name}\nRole: {self.role}\nGoal: {self.goal}\nBackstory: {self.backstory}",
            "config": {
                "visibility": "public",
                "require_context": True,
                "run_in_executor": True  # Add this to ensure proper execution
            },
            "role": self.role,
            "goal": self.goal,
            "backstory": self.backstory,
            "reasoning_strategy": self.reasoning_strategy.model_dump(mode="json") if self.reasoning_strategy else None,
            "result_type": self.result_type.__name__ if self.result_type else None,
            "tools": tools_meta,
            "model": str(self._model) if self._model else None,
            "run": run_wrapper,
        })

        return service_info

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
            # Check if we should use reasoning strategy
            if self.reasoning_strategy:
                strategy = self.reasoning_strategy
                model_used = self._get_model(model)
                usage_obj = result.Usage()
                run_context = RunContext(
                    deps=deps,
                    model=model_used,
                    usage=usage_obj,
                    prompt=user_prompt,
                    messages=message_history or [],
                    run_step=0
                )

                # Execute reasoning strategy with streaming
                if isinstance(strategy.type, str):
                    strategy_types = [strategy.type]
                else:
                    strategy_types = strategy.type

                for strategy_type in strategy_types:
                    if strategy_type == "react":
                        if not strategy.react_config:
                            raise ValueError("ReAct strategy selected but no react_config provided")
                        from schema_agents.reasoning import execute_react_reasoning
                        
                        # Create a StreamedRunResult for the ReAct reasoning
                        streamed_result = result.StreamedRunResult(
                            message_history or [],
                            len(message_history) if message_history else 0,
                            None,  # No usage limits for now
                            None,  # No result stream yet
                            None,  # No result schema
                            run_context,
                            [],  # No result validators
                            None,  # No tool name
                            None,  # No on_complete callback
                        )
                        
                        # Start the ReAct reasoning with streaming
                        stream = await execute_react_reasoning(
                            user_prompt,
                            model_used,
                            list(agent_copy._function_tools.values()),
                            strategy,
                            run_context,
                            stream=True
                        )
                        
                        # Create a ModelResponse for streaming
                        async def stream_generator():
                            async for chunk in stream:
                                # Update the streamed result with the new chunk
                                streamed_result.data = chunk
                                if isinstance(chunk, ModelResponse):
                                    yield chunk
                                else:
                                    yield ModelResponse(parts=[TextPart(content=str(chunk))], model_name=model_used.model_name)
                        
                        # Create a StreamedResponse
                        streamed_result._stream_response = stream_generator()
                        yield streamed_result
                        
                        # Consume the stream to ensure it completes
                        async for _ in streamed_result._stream_response:
                            pass
                    else:
                        raise ValueError(f"Unknown strategy type: {strategy_type}")
            else:
                # Run with the copied agent without reasoning strategy
                async with super(Agent, agent_copy).run_stream(
                    user_prompt,
                    result_type=result_type,
                    message_history=message_history,
                    model=model,
                    deps=deps,
                    **kwargs
                ) as response:
                    yield response

        except Exception as e:
            # Wrap exceptions to ensure consistent error handling
            if isinstance(e, exceptions.ModelRetry):
                raise e
            if isinstance(e, (ValueError, TypeError, ValidationError)):
                # Convert validation errors to UserError
                raise exceptions.UserError(str(e)) from e
            raise e
