import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Optional

from pydantic import BaseModel, Field

from schema_agents.models import ChatMessage
from schema_agents.schema_agent import Agent
from schema_agents.schema_tools import schema_tool
from schema_agents.tools import Tool


class TestSchemaTools:
    """Test suite for the schema_tool functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        mock = AsyncMock()
        mock.return_value = ChatMessage(role="assistant", content="Final Answer: Test result")
        # Add token count attributes to avoid format errors
        mock.last_input_token_count = 10
        mock.last_output_token_count = 5
        return mock

    def test_schema_tool_decorator(self, mock_model):
        """Test that the schema_tool decorator creates a valid tool."""
        @schema_tool
        async def say_hello(
            name: str = Field(default="world", description="Name of the person to greet")
        ) -> str:
            """Say hello to someone"""
            return f"Hello {name}!"

        # Check that the function has a __tool__ attribute
        assert hasattr(say_hello, "__tool__")
        
        # Check that the tool has the expected attributes
        tool = say_hello.__tool__
        assert isinstance(tool, Tool)
        assert tool.name == "say_hello"
        assert tool.description == "Say hello to someone"
        assert "name" in tool.inputs
        assert tool.inputs["name"]["type"] == "string"
        assert tool.inputs["name"]["description"] == "Name of the person to greet"
        assert tool.inputs["name"]["nullable"] is True
        assert tool.output_type == "string"

    def test_schema_tool_with_custom_name_description(self, mock_model):
        """Test that the schema_tool decorator works with custom name and description."""
        @schema_tool(name="greeter", description="A tool that greets people")
        async def say_hello(
            name: str = Field(default="world", description="Name of the person to greet")
        ) -> str:
            """Say hello to someone"""
            return f"Hello {name}!"

        # Check that the tool has the custom name and description
        tool = say_hello.__tool__
        assert tool.name == "greeter"
        assert tool.description == "A tool that greets people"

    def test_schema_tool_with_required_param(self, mock_model):
        """Test that the schema_tool decorator handles required parameters correctly."""
        @schema_tool
        async def calculate(
            x: int = Field(..., description="First number"),
            y: int = Field(default=10, description="Second number")
        ) -> int:
            """Perform a calculation"""
            return x + y

        # Check that the tool has the expected attributes
        tool = calculate.__tool__
        assert "x" in tool.inputs
        assert "y" in tool.inputs
        assert "nullable" not in tool.inputs["x"]  # Required parameter
        assert tool.inputs["y"]["nullable"] is True  # Optional parameter

    @pytest.mark.asyncio
    async def test_schema_tool_execution(self, mock_model):
        """Test that the schema_tool can be executed."""
        @schema_tool
        async def add_numbers(
            x: int = Field(..., description="First number"),
            y: int = Field(default=10, description="Second number")
        ) -> int:
            """Add two numbers"""
            return x + y

        # Execute the tool
        tool = add_numbers.__tool__
        result = await tool(x=5)
        assert result == 15
        
        result = await tool(x=5, y=7)
        assert result == 12

    def test_agent_schema_tool_method(self, mock_model):
        """Test that the agent.schema_tool method works correctly."""
        # Create an agent
        agent = Agent(model=mock_model, name="TestAgent")
        
        # Define a tool using the agent.schema_tool method
        @agent.schema_tool
        async def say_hello(
            name: str = Field(default="world", description="Name of the person to greet")
        ) -> str:
            """Say hello to someone"""
            return f"Hello {name}!"
            
        # Check that the tool was added to the agent
        assert "say_hello" in agent.tools
        assert agent.tools["say_hello"] is say_hello.__tool__
        
        # Define another tool with custom name
        @agent.schema_tool(name="adder", description="Add numbers")
        async def add_numbers(
            x: int = Field(..., description="First number"),
            y: int = Field(default=10, description="Second number")
        ) -> int:
            """Add two numbers"""
            return x + y
            
        # Check that the tool was added with the custom name
        assert "adder" in agent.tools
        assert agent.tools["adder"] is add_numbers.__tool__ 