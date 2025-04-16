from typing import Any, Dict, Type, TypeVar, Union, get_type_hints
import json
import inspect

from pydantic import BaseModel, ValidationError

from schema_agents.default_tools import FinalAnswerTool
from schema_agents.tools import Tool
from schema_agents.utils import FINAL_ANSWER_MARKER

T = TypeVar('T', bound=BaseModel)

class FinalAnswerException(Exception):
    """Exception raised when the final_answer function is called."""
    def __init__(self, answer: Any):
        self.answer = answer
        super().__init__(f"{FINAL_ANSWER_MARKER}: {answer.model_dump_json()}")

class PydanticFinalAnswerTool(FinalAnswerTool):
    """A final answer tool that converts the answer to a Pydantic model.
    
    This tool extends the standard FinalAnswerTool to support Pydantic models
    as result types. It attempts to parse the answer as JSON and validate it
    against the provided Pydantic model.
    """
    
    def __init__(self, model_type: Type[T]):
        """Initialize the PydanticFinalAnswerTool.
        
        Args:
            model_type: The Pydantic model type to convert the answer to
        """
        super().__init__()
        self.model_type = model_type
        self.is_async = True  # Mark this tool as async
        
        # Update the description to include information about the expected model
        model_fields = self.model_type.__annotations__ if hasattr(self.model_type, '__annotations__') else {}
        field_descriptions = []
        
        for field_name, field_type in model_fields.items():
            field_info = getattr(self.model_type, field_name, None)
            description = getattr(field_info, 'description', f"Type: {field_type}")
            field_descriptions.append(f"- {field_name}: {description}")
        
        model_description = "\n".join(field_descriptions)
        self.description = f"""Provides a final answer to the given problem as a structured {self.model_type.__name__} object.
The answer should be a valid JSON object with the following fields:
{model_description}
"""
    
    def forward(self, answer: Any) -> T:
        """Convert the answer to a Pydantic model.
        
        Args:
            answer: The answer to convert, can be a string (JSON) or a dict
            
        Returns:
            The answer converted to a Pydantic model
            
        Raises:
            ValidationError: If the answer cannot be converted to the model
        """
        if isinstance(answer, self.model_type):
            return answer
        
        if isinstance(answer, str):
            try:
                # Try to parse the answer as JSON
                data = json.loads(answer)
            except json.JSONDecodeError:
                # If the answer is not valid JSON, try to parse it as a string representation of a dict
                try:
                    # This is a common format when LLMs output Python dicts as strings
                    data = eval(answer, {"__builtins__": {}}, {})
                    if not isinstance(data, dict):
                        raise ValueError("Evaluated result is not a dictionary")
                except Exception as e:
                    raise ValidationError(
                        f"Could not parse answer as JSON or dict: {e}",
                        model=self.model_type
                    )
        else:
            # Assume the answer is already a dict or similar structure
            data = answer
        
        # Convert the data to the Pydantic model
        return self.model_type.model_validate(data)
        
    async def async_forward(self, answer: Any) -> T:
        """Async version of forward that properly raises FinalAnswerException.
        
        Args:
            answer: The answer to convert, can be a string (JSON) or a dict
            
        Returns:
            The answer converted to a Pydantic model
            
        Raises:
            FinalAnswerException: With the converted Pydantic model as the value
        """
        # Convert the answer to a Pydantic model
        converted_answer = self.forward(answer)
        
        # Raise FinalAnswerException with the converted answer
        raise FinalAnswerException(converted_answer)


def create_pydantic_agent(agent_class, model_type: Type[BaseModel], *args, **kwargs):
    """Create an agent with a PydanticFinalAnswerTool.
    
    This function creates an agent with a PydanticFinalAnswerTool that converts
    the final answer to the specified Pydantic model type.
    
    Args:
        agent_class: The agent class to create
        model_type: The Pydantic model type to convert the answer to
        *args: Additional positional arguments to pass to the agent constructor
        **kwargs: Additional keyword arguments to pass to the agent constructor
        
    Returns:
        An agent with a PydanticFinalAnswerTool
    """
    # Create the agent
    agent = agent_class(*args, **kwargs)
    
    # Replace the final_answer tool with a PydanticFinalAnswerTool
    agent.tools["final_answer"] = PydanticFinalAnswerTool(model_type)
    
    return agent 