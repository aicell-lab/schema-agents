import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Optional

from pydantic import BaseModel, Field

from schema_agents.models import ChatMessage
from schema_agents.pydantic_tools import PydanticFinalAnswerTool, create_pydantic_agent
from schema_agents import Agent


class Person(BaseModel):
    """A person model for testing."""
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    hobbies: List[str] = Field(default_factory=list, description="The person's hobbies")
    address: Optional[str] = Field(default=None, description="The person's address")


class TestPydanticFinalAnswerTool:
    """Test suite for the PydanticFinalAnswerTool implementation."""

    def test_initialization(self):
        """Test that the tool initializes correctly."""
        tool = PydanticFinalAnswerTool(Person)
        
        # Check that the tool has the expected attributes
        assert tool.name == "final_answer"
        assert tool.model_type == Person
        assert "Person" in tool.description
        assert "name" in tool.description
        assert "age" in tool.description
        assert "hobbies" in tool.description
        assert "address" in tool.description

    @pytest.mark.asyncio
    async def test_forward_with_dict(self):
        """Test that the tool correctly converts a dict to a Pydantic model."""
        tool = PydanticFinalAnswerTool(Person)
        
        # Test with a dict
        result = await tool({"name": "John", "age": 30, "hobbies": ["reading", "coding"]})
        
        # Check that the result is a Person
        assert isinstance(result, Person)
        assert result.name == "John"
        assert result.age == 30
        assert result.hobbies == ["reading", "coding"]
        assert result.address is None

    @pytest.mark.asyncio
    async def test_forward_with_json_string(self):
        """Test that the tool correctly converts a JSON string to a Pydantic model."""
        tool = PydanticFinalAnswerTool(Person)
        
        # Test with a JSON string
        result = await tool('{"name": "Jane", "age": 25, "hobbies": ["swimming", "hiking"], "address": "123 Main St"}')
        
        # Check that the result is a Person
        assert isinstance(result, Person)
        assert result.name == "Jane"
        assert result.age == 25
        assert result.hobbies == ["swimming", "hiking"]
        assert result.address == "123 Main St"

    @pytest.mark.asyncio
    async def test_forward_with_python_dict_string(self):
        """Test that the tool correctly converts a Python dict string to a Pydantic model."""
        tool = PydanticFinalAnswerTool(Person)
        
        # Test with a Python dict string
        result = await tool("{'name': 'Bob', 'age': 40, 'hobbies': ['gardening', 'cooking']}")
        
        # Check that the result is a Person
        assert isinstance(result, Person)
        assert result.name == "Bob"
        assert result.age == 40
        assert result.hobbies == ["gardening", "cooking"]
        assert result.address is None

    @pytest.mark.asyncio
    async def test_forward_with_model_instance(self):
        """Test that the tool correctly handles a Pydantic model instance."""
        tool = PydanticFinalAnswerTool(Person)
        
        # Test with a Person instance
        person = Person(name="Alice", age=35, hobbies=["painting", "music"])
        result = await tool(person)
        
        # Check that the result is the same Person
        assert result is person
        assert result.name == "Alice"
        assert result.age == 35
        assert result.hobbies == ["painting", "music"]
        assert result.address is None

    @pytest.mark.asyncio
    async def test_forward_with_invalid_input(self):
        """Test that the tool raises a ValidationError for invalid input."""
        tool = PydanticFinalAnswerTool(Person)
        
        # Test with invalid input
        with pytest.raises(Exception):
            await tool("not a valid input")

    @pytest.mark.asyncio
    async def test_forward_with_missing_required_field(self):
        """Test that the tool raises a ValidationError for missing required fields."""
        tool = PydanticFinalAnswerTool(Person)
        
        # Test with missing required field
        with pytest.raises(Exception):
            await tool({"name": "John"})  # Missing age


class TestCreatePydanticAgent:
    """Test suite for the create_pydantic_agent function."""

    def test_create_pydantic_agent(self):
        """Test that the function correctly creates an agent with a PydanticFinalAnswerTool."""
        # Create a mock model
        mock_model = AsyncMock()
        mock_model.return_value = ChatMessage(role="assistant", content="Final Answer: Test result")
        mock_model.last_input_token_count = 10
        mock_model.last_output_token_count = 5
        
        # Create an agent
        agent = create_pydantic_agent(Agent, Person, model=mock_model, name="TestAgent")
        
        # Check that the agent has the expected attributes
        assert agent.name == "TestAgent"
        assert isinstance(agent.tools["final_answer"], PydanticFinalAnswerTool)
        assert agent.tools["final_answer"].model_type == Person

    def test_agent_with_pydantic_model(self):
        """Test that the Agent class correctly initializes with a Pydantic model as result_type."""
        # Create a mock model
        mock_model = AsyncMock()
        mock_model.return_value = ChatMessage(role="assistant", content="Final Answer: Test result")
        mock_model.last_input_token_count = 10
        mock_model.last_output_token_count = 5
        
        # Create an agent with a Pydantic model as result_type
        agent = Agent(
            model=mock_model,
            name="TestAgent",
            result_type=Person,
        )
        
        # Check that the agent has a PydanticFinalAnswerTool
        assert isinstance(agent.tools["final_answer"], PydanticFinalAnswerTool)
        assert agent.tools["final_answer"].model_type == Person
        
        # Check that the agent has the correct result_type
        assert agent.result_type == Person 