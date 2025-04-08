from typing import Type, List, Dict, Any, Optional, Callable, Union, TypeVar

from pydantic import BaseModel

from schema_agents.agents import CodeAgent
from schema_agents.tools import Tool
from schema_agents.models import ChatMessage
from schema_agents.pydantic_tools import PydanticFinalAnswerTool
from schema_agents.schema_tools import schema_tool as schema_tool_decorator


ResultDataT = TypeVar('ResultDataT')


class Agent(CodeAgent):
    """Schema Agents Platform Agent implementation.
    
    This extends the Pydantic AI Agent with additional capabilities:
    - Runtime tool attachment
    - Dynamic reasoning strategies
    - Improved concurrency safety
    - Vector memory integration
    """
    def __init__(
        self,
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        *,
        name: str | None = None,
        result_type: Type[ResultDataT] = str,
        role: str | None = None,
        goal: str | None = None,
        backstory: str | None = None,
        tools: List[Tool] = None,
        additional_authorized_imports: Optional[List[str]] = None,
        planning_interval: Optional[int] = None,
        executor_type: str = "local",
        executor_kwargs: Optional[Dict[str, Any]] = None,
        max_print_outputs_length: Optional[int] = None,
        **kwargs
    ):
        """Initialize the Schema Agent.
        
        Args:
            model: Model that will generate the agent's actions
            name: Name of the agent
            result_type: Type of the result data
            role: Role of the agent
            goal: Goal of the agent
            backstory: Backstory of the agent
            tools: Tools that the agent can use
            additional_authorized_imports: Additional authorized imports for the agent
            planning_interval: Interval at which the agent will run a planning step
            executor_type: Which executor type to use between "local", "e2b", or "docker"
            executor_kwargs: Additional arguments to pass to initialize the executor
            max_print_outputs_length: Maximum length of the print outputs
            **kwargs: Additional keyword arguments
        """
        # Set agent-specific attributes
        self.result_type = result_type
        self.role = role
        self.goal = goal
        self.backstory = backstory
        
        # Initialize tools if not provided
        if tools is None:
            tools = []
        
        # Set description based on role and goal if not in kwargs
        if 'description' not in kwargs and (role or goal):
            description = ""
            if role:
                description += f"Role: {role}\n"
            if goal:
                description += f"Goal: {goal}\n"
            if backstory:
                description += f"Backstory: {backstory}"
            kwargs['description'] = description.strip()
        
        # Initialize the parent CodeAgent
        super().__init__(
            tools=tools,
            model=model,
            name=name,
            additional_authorized_imports=additional_authorized_imports,
            planning_interval=planning_interval,
            executor_type=executor_type,
            executor_kwargs=executor_kwargs,
            max_print_outputs_length=max_print_outputs_length,
            **kwargs
        )
        
        # Replace the final_answer tool with a PydanticFinalAnswerTool if result_type is a Pydantic model
        if issubclass(self.result_type, BaseModel):
            self.tools["final_answer"] = PydanticFinalAnswerTool(self.result_type)
    
    def schema_tool(self, func=None, *, name: Optional[str] = None, description: Optional[str] = None):
        """Decorator for schema tools.
        
        This decorator allows using Pydantic Field for parameter descriptions and default values.
        The Field metadata is automatically extracted to create a properly documented tool.
        
        Args:
            func: The function to decorate
            name: Optional custom name for the tool (defaults to function name)
            description: Optional custom description (defaults to function docstring)
        
        Example:
            @agent.schema_tool
            async def say_hello(
                name: str = Field(default="alice", description="Name of the person to greet")
            ) -> str:
                '''Say hello to someone'''
                return f"Hello {name}!"
                
            @agent.schema_tool(name="calculator")
            async def calculate(
                x: int = Field(..., description="First number"),
                y: int = Field(default=10, description="Second number"),
                operation: str = Field(default="add", description="Operation type")
            ) -> str:
                '''Perform a calculation'''
                return f"{x} {operation} {y}"
        """
        def decorator(func):
            # Use the schema_tool_decorator to create the tool
            decorated_func = schema_tool_decorator(func, name=name, description=description)
            
            # Add the tool to this agent
            self.tools[decorated_func.__tool__.name] = decorated_func.__tool__
            # Return the decorated function
            return decorated_func
        
        if func is None:
            return decorator
        return decorator(func)
    
    async def run(
        self, 
        query: str, 
        *, 
        tools: List[Tool] = None, 
        result_type: Type = None,
        managed_agents: List = None,
        state: Dict[str, Any] = None,
        reset_memory: bool = None,
        **kwargs
    ) -> ResultDataT:
        """Run the agent with the given query.
        
        Args:
            query: Query to run the agent with
            tools: Optional list of tools to use for this run, overriding the agent's default tools
            result_type: Optional result type to use for this run, overriding the agent's default result_type
            managed_agents: Optional list of managed agents to use for this run
            state: Optional state dictionary to use for this run
            reset_memory: Optional flag to control whether to reset the memory for this run
                          (if not provided, uses the reset parameter passed to parent run method)
            **kwargs: Additional arguments to pass to the parent run method (stream, reset, images, additional_args, max_steps)
            
        Returns:
            The result of the agent's execution, cast to the result_type
        """
        # If any overrides are provided, create a forked instance to avoid concurrency issues
        if tools is not None or result_type is not None or managed_agents is not None or state is not None or reset_memory is not None:
            # Create a shallow copy of the agent
            import copy
            forked_agent = copy.copy(self)
            
            # Set up tools for the forked agent if provided
            if tools is not None:
                # Convert list to dict format expected by the agent
                forked_agent.tools = {tool.name: tool for tool in tools}
                # Ensure final_answer tool is present
                if "final_answer" not in forked_agent.tools:
                    # Use appropriate final answer tool based on result_type (or self.result_type if not specified)
                    rt = result_type or self.result_type
                    if issubclass(rt, BaseModel):
                        from schema_agents.pydantic_tools import PydanticFinalAnswerTool
                        forked_agent.tools["final_answer"] = PydanticFinalAnswerTool(rt)
                    else:
                        from schema_agents.default_tools import FinalAnswerTool
                        forked_agent.tools["final_answer"] = FinalAnswerTool()
            
            # Set up result_type for the forked agent if provided
            if result_type is not None:
                forked_agent.result_type = result_type
                
                # Update final_answer tool if needed for the new result_type
                if issubclass(result_type, BaseModel) and (tools is None or "final_answer" not in forked_agent.tools):
                    from schema_agents.pydantic_tools import PydanticFinalAnswerTool
                    forked_agent.tools["final_answer"] = PydanticFinalAnswerTool(result_type)
            
            # Set up managed_agents for the forked agent if provided
            if managed_agents is not None:
                # Convert list to dict format expected by the agent
                forked_agent.managed_agents = {agent.name: agent for agent in managed_agents}
            
            # Set up state for the forked agent if provided
            if state is not None:
                forked_agent.state = state.copy()
            
            # Handle reset_memory parameter
            if reset_memory is not None:
                # Override the reset parameter in kwargs
                kwargs['reset'] = reset_memory
            
            # Create a new python executor for the forked agent
            if hasattr(self, "python_executor"):
                # Create a new executor of the same type
                forked_agent.python_executor = self.create_python_executor(
                    self.executor_type, 
                    self.executor_kwargs
                )
                # Initialize the executor with the agent's state and tools
                await forked_agent.python_executor.send_variables(variables=forked_agent.state)
                await forked_agent.python_executor.send_tools({
                    **forked_agent.tools, 
                    **forked_agent.managed_agents
                })
            
            # Run the forked agent
            return await forked_agent.run(query, **kwargs)
        
        # If no overrides, run normally
        result = await super().run(query, **kwargs)
        
        # If we're using a PydanticFinalAnswerTool, the result should already be converted
        if isinstance(result, self.result_type):
            return result
        
        # Convert the result to the specified result_type if needed
        if result is not None and self.result_type != str:
            try:
                return self.result_type(result)
            except (ValueError, TypeError):
                return result
        
        return result