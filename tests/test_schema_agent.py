import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Optional

from pydantic import BaseModel, Field

from schema_agents.schema_agent import Agent
from schema_agents.models import ChatMessage
from schema_agents.tools import Tool


class TestSchemaAgent:
    """Test suite for the Schema Agent implementation."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        mock = AsyncMock()
        mock.return_value = ChatMessage(role="assistant", content="Final Answer: Test result")
        # Add token count attributes to avoid format errors
        mock.last_input_token_count = 10
        mock.last_output_token_count = 5
        return mock

    @pytest.fixture
    def test_tool(self):
        """Create a test tool for the agent."""
        mock_tool = MagicMock(spec=Tool)
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        # Add required attributes for template rendering
        mock_tool.inputs = {"param": {"type": "string", "description": "A parameter"}}
        mock_tool.output_type = "string"
        return mock_tool

    def test_agent_initialization(self, mock_model, test_tool):
        """Test that the agent initializes correctly."""
        # Create an agent with minimal parameters
        agent = Agent(
            model=mock_model,
            name="TestAgent",
            tools=[test_tool],
        )
        
        # Check that the agent has the expected attributes
        assert agent.name == "TestAgent"
        assert agent.result_type == str
        assert agent.role is None
        assert agent.goal is None
        assert agent.backstory is None
        assert "test_tool" in agent.tools
        assert agent.tools["test_tool"] == test_tool

    def test_agent_initialization_with_role_goal(self, mock_model):
        """Test that the agent initializes correctly with role and goal."""
        # Create an agent with role and goal
        agent = Agent(
            model=mock_model,
            name="TestAgent",
            role="Test Role",
            goal="Test Goal",
            backstory="Test Backstory",
        )
        
        # Check that the agent has the expected attributes
        assert agent.role == "Test Role"
        assert agent.goal == "Test Goal"
        assert agent.backstory == "Test Backstory"
        assert "Role: Test Role" in agent.description
        assert "Goal: Test Goal" in agent.description
        assert "Backstory: Test Backstory" in agent.description

    @pytest.mark.asyncio
    @patch("schema_agents.agents.CodeAgent.run")
    async def test_run_with_string_result(self, mock_run, mock_model):
        """Test running the agent with a string result type."""
        # Set up the mock to return a string
        mock_run.return_value = "Final Answer: Test result"
        
        # Create an agent
        agent = Agent(
            model=mock_model,
            name="TestAgent",
        )
        
        # Run the agent
        result = await agent.run("Test query")
        
        # Check that the result is a string
        assert isinstance(result, str)
        assert result == "Final Answer: Test result"
        
        # Verify that the parent run method was called
        mock_run.assert_called_once_with("Test query")

    @pytest.mark.asyncio
    @patch("schema_agents.agents.CodeAgent.run")
    async def test_run_with_custom_result_type(self, mock_run, mock_model):
        """Test running the agent with a custom result type."""
        # Set up the mock to return a string
        mock_run.return_value = "Final Answer: Test result"
        
        # Create a custom result type
        class CustomResult:
            def __init__(self, value):
                self.value = value
        
        # Create an agent with the custom result type
        agent = Agent(
            model=mock_model,
            name="TestAgent",
            result_type=CustomResult,
        )
        
        # Run the agent
        result = await agent.run("Test query")
        
        # Check that the result is of the custom type
        assert isinstance(result, CustomResult)
        assert result.value == "Final Answer: Test result"
        
        # Verify that the parent run method was called
        mock_run.assert_called_once_with("Test query")

    @pytest.mark.asyncio
    @patch("schema_agents.agents.CodeAgent.run")
    async def test_run_with_invalid_result_conversion(self, mock_run, mock_model):
        """Test running the agent with an invalid result conversion."""
        # Set up the mock to return a string
        mock_run.return_value = "Final Answer: Test result"
        
        # Create a custom result type that will fail conversion
        class CustomResult:
            def __init__(self, value):
                if not isinstance(value, int):
                    raise ValueError("Value must be an integer")
                self.value = value
        
        # Create an agent with the custom result type
        agent = Agent(
            model=mock_model,
            name="TestAgent",
            result_type=CustomResult,
        )
        
        # Run the agent - should return the original string since conversion fails
        result = await agent.run("Test query")
        
        # Check that the result is the original string
        assert isinstance(result, str)
        assert result == "Final Answer: Test result"
        
        # Verify that the parent run method was called
        mock_run.assert_called_once_with("Test query")

    def test_schema_tool_decorator(self, mock_model):
        """Test that the agent.schema_tool decorator creates a valid tool."""
        from pydantic import Field
        
        # Create an agent
        agent = Agent(
            model=mock_model,
            name="TestAgent",
        )
        
        # Define a tool using the agent.schema_tool decorator
        @agent.schema_tool
        async def say_hello(
            name: str = Field(default="world", description="Name of the person to greet")
        ) -> str:
            """Say hello to someone"""
            return f"Hello {name}!"
        
        # Check that the tool was added to the agent
        assert "say_hello" in agent.tools
        
        # Check that the tool has the expected attributes
        tool = agent.tools["say_hello"]
        assert tool.name == "say_hello"
        assert tool.description == "Say hello to someone"
        assert "name" in tool.inputs
        assert tool.inputs["name"]["type"] == "string"
        assert tool.inputs["name"]["description"] == "Name of the person to greet"
        assert tool.inputs["name"]["nullable"] is True
        assert tool.output_type == "string"
        
    def test_schema_tool_with_custom_name_description(self, mock_model):
        """Test that the agent.schema_tool decorator works with custom name and description."""
        from pydantic import Field
        
        # Create an agent
        agent = Agent(
            model=mock_model,
            name="TestAgent",
        )
        
        # Define a tool using the agent.schema_tool decorator with custom name and description
        @agent.schema_tool(name="greeter", description="A tool that greets people")
        async def say_hello(
            name: str = Field(default="world", description="Name of the person to greet")
        ) -> str:
            """Say hello to someone"""
            return f"Hello {name}!"
        
        # Check that the tool was added to the agent with the custom name
        assert "greeter" in agent.tools
        
        # Check that the tool has the expected attributes
        tool = agent.tools["greeter"]
        assert tool.name == "greeter"
        assert tool.description == "A tool that greets people"
        assert "name" in tool.inputs
        assert tool.inputs["name"]["type"] == "string"
        assert tool.inputs["name"]["description"] == "Name of the person to greet"
        
    def test_schema_tool_with_required_param(self, mock_model):
        """Test that the agent.schema_tool decorator handles required parameters correctly."""
        from pydantic import Field
        
        # Create an agent
        agent = Agent(
            model=mock_model,
            name="TestAgent",
        )
        
        # Define a tool using the agent.schema_tool decorator with a required parameter
        @agent.schema_tool
        async def calculate(
            x: int = Field(..., description="First number"),
            y: int = Field(default=10, description="Second number")
        ) -> int:
            """Perform a calculation"""
            return x + y
        
        # Check that the tool was added to the agent
        assert "calculate" in agent.tools
        
        # Check that the tool has the expected attributes
        tool = agent.tools["calculate"]
        assert "x" in tool.inputs
        assert "y" in tool.inputs
        assert "nullable" not in tool.inputs["x"]  # Required parameter
        assert tool.inputs["y"]["nullable"] is True  # Optional parameter
        
    @pytest.mark.asyncio
    async def test_schema_tool_execution(self, mock_model):
        """Test that the schema_tool can be executed through the agent."""
        from pydantic import Field
        
        # Create an agent
        agent = Agent(
            model=mock_model,
            name="TestAgent",
        )
        
        # Define a tool using the agent.schema_tool decorator
        @agent.schema_tool
        async def add_numbers(
            x: int = Field(..., description="First number"),
            y: int = Field(default=10, description="Second number")
        ) -> int:
            """Add two numbers"""
            return x + y
        
        # Execute the tool
        tool = agent.tools["add_numbers"]
        result = await tool(x=5)
        assert result == 15
        
        result = await tool(x=5, y=7)
        assert result == 12
        
    @pytest.mark.asyncio
    @patch("schema_agents.agents.CodeAgent.run")
    async def test_agent_run_with_schema_tools(self, mock_run, mock_model):
        """Test running the agent with schema tools."""
        from pydantic import Field
        
        # Set up the mock to return a string
        mock_run.return_value = "Final Answer: 15"
        
        # Create an agent
        agent = Agent(
            model=mock_model,
            name="TestAgent",
        )
        
        # Define a tool using the agent.schema_tool decorator
        @agent.schema_tool
        async def add_numbers(
            x: int = Field(..., description="First number"),
            y: int = Field(default=10, description="Second number")
        ) -> int:
            """Add two numbers"""
            return x + y
        
        # Run the agent
        result = await agent.run("What is 5 + 10?")
        
        # Check that the result is correct
        assert result == "Final Answer: 15"
        
        # Verify that the parent run method was called
        mock_run.assert_called_once_with("What is 5 + 10?")
        
    @pytest.mark.asyncio
    async def test_schema_tool_with_pydantic_models(self, mock_model):
        """Test that the schema_tool can work with Pydantic models."""
        # Define a Pydantic model for input
        class PersonInput(BaseModel):
            name: str = Field(description="Person's name")
            age: int = Field(description="Person's age")
            hobbies: Optional[List[str]] = Field(default=None, description="Person's hobbies")
            
        # Define a Pydantic model for output
        class PersonOutput(BaseModel):
            greeting: str = Field(description="Greeting message")
            age_next_year: int = Field(description="Age next year")
            hobby_count: int = Field(description="Number of hobbies")
            
        # Create an agent
        agent = Agent(
            model=mock_model,
            name="TestAgent",
        )
        
        # Define a tool using the agent.schema_tool decorator with Pydantic models
        @agent.schema_tool
        async def process_person(
            person: PersonInput = Field(..., description="Person information")
        ) -> PersonOutput:
            """Process person information and return a greeting with age next year."""
            hobby_count = len(person.hobbies) if person.hobbies else 0
            return PersonOutput(
                greeting=f"Hello {person.name}!",
                age_next_year=person.age + 1,
                hobby_count=hobby_count
            )
        
        # Execute the tool with a Pydantic model
        tool = agent.tools["process_person"]
        
        # Create a PersonInput instance
        person_input = PersonInput(
            name="Alice",
            age=30,
            hobbies=["reading", "coding", "hiking"]
        )
        
        # Call the tool with the Pydantic model
        result = await tool(person=person_input)
        
        # Check that the result is a PersonOutput instance
        assert isinstance(result, PersonOutput)
        assert result.greeting == "Hello Alice!"
        assert result.age_next_year == 31
        assert result.hobby_count == 3
        
        # Check that the tool has the expected attributes
        assert tool.name == "process_person"
        assert "person" in tool.inputs
        assert tool.inputs["person"]["type"] == "object"
        assert "nullable" not in tool.inputs["person"]  # Required parameter
        
    @pytest.mark.asyncio
    @patch("schema_agents.agents.CodeAgent.run")
    async def test_agent_with_pydantic_result_type_and_schema_tool(self, mock_run, mock_model):
        """Test an agent with Pydantic result type and schema tools."""
        # Define a Pydantic model for the agent's result type
        class QueryResult(BaseModel):
            answer: str = Field(description="The answer to the query")
            confidence: float = Field(description="Confidence score (0-1)")
            sources: List[str] = Field(default_factory=list, description="Sources of information")
            
        # Create a QueryResult instance to return from the mock
        expected_result = QueryResult(
            answer="The sum is 15",
            confidence=0.95,
            sources=["Calculator", "Math knowledge"]
        )
        
        # Set up the mock to return the QueryResult instance
        mock_run.return_value = expected_result
        
        # Create an agent with the Pydantic model as result_type
        agent = Agent(
            model=mock_model,
            name="TestAgent",
            result_type=QueryResult,
        )
        
        # Define a tool using the agent.schema_tool decorator
        @agent.schema_tool
        async def add_numbers(
            x: int = Field(..., description="First number"),
            y: int = Field(default=10, description="Second number")
        ) -> int:
            """Add two numbers"""
            return x + y
        
        # Run the agent
        result = await agent.run("What is 5 + 10?")
        
        # Check that the result is a QueryResult instance
        assert isinstance(result, QueryResult)
        assert result.answer == "The sum is 15"
        assert result.confidence == 0.95
        assert result.sources == ["Calculator", "Math knowledge"]
        
        # Verify that the parent run method was called
        mock_run.assert_called_once_with("What is 5 + 10?")
        
        # Check that the final_answer tool is a PydanticFinalAnswerTool
        from schema_agents.pydantic_tools import PydanticFinalAnswerTool
        assert isinstance(agent.tools["final_answer"], PydanticFinalAnswerTool)
        assert agent.tools["final_answer"].model_type == QueryResult
        
    @pytest.mark.asyncio
    async def test_schema_tool_with_nested_pydantic_models(self, mock_model):
        """Test that the schema_tool can work with nested Pydantic models."""
        # Define nested Pydantic models
        class Address(BaseModel):
            street: str = Field(description="Street address")
            city: str = Field(description="City name")
            zip_code: str = Field(description="ZIP code")
            
        class Contact(BaseModel):
            email: str = Field(description="Email address")
            phone: Optional[str] = Field(default=None, description="Phone number")
            
        class Person(BaseModel):
            name: str = Field(description="Person's name")
            age: int = Field(description="Person's age")
            address: Address = Field(description="Person's address")
            contact: Contact = Field(description="Person's contact information")
            tags: List[str] = Field(default_factory=list, description="Tags associated with the person")
            
        class PersonSummary(BaseModel):
            full_name: str = Field(description="Person's full name")
            location: str = Field(description="Person's location")
            contact_info: str = Field(description="Person's contact information")
            age_group: str = Field(description="Person's age group")
            tag_count: int = Field(description="Number of tags")
            
        # Create an agent
        agent = Agent(
            model=mock_model,
            name="TestAgent",
        )
        
        # Define a tool using the agent.schema_tool decorator with nested Pydantic models
        @agent.schema_tool
        async def summarize_person(
            person: Person = Field(..., description="Person information")
        ) -> PersonSummary:
            """Summarize person information."""
            # Determine age group
            if person.age < 18:
                age_group = "Minor"
            elif person.age < 65:
                age_group = "Adult"
            else:
                age_group = "Senior"
                
            # Create a summary
            return PersonSummary(
                full_name=person.name,
                location=f"{person.address.city}, {person.address.zip_code}",
                contact_info=f"Email: {person.contact.email}, Phone: {person.contact.phone or 'N/A'}",
                age_group=age_group,
                tag_count=len(person.tags)
            )
        
        # Execute the tool with nested Pydantic models
        tool = agent.tools["summarize_person"]
        
        # Create a Person instance with nested Address and Contact
        person_input = Person(
            name="John Doe",
            age=35,
            address=Address(
                street="123 Main St",
                city="New York",
                zip_code="10001"
            ),
            contact=Contact(
                email="john.doe@example.com",
                phone="555-123-4567"
            ),
            tags=["customer", "premium", "verified"]
        )
        
        # Call the tool with the nested Pydantic model
        result = await tool(person=person_input)
        
        # Check that the result is a PersonSummary instance
        assert isinstance(result, PersonSummary)
        assert result.full_name == "John Doe"
        assert result.location == "New York, 10001"
        assert result.contact_info == "Email: john.doe@example.com, Phone: 555-123-4567"
        assert result.age_group == "Adult"
        assert result.tag_count == 3
        
        # Check that the tool has the expected attributes
        assert tool.name == "summarize_person"
        assert "person" in tool.inputs
        assert tool.inputs["person"]["type"] == "object"
        assert "nullable" not in tool.inputs["person"]  # Required parameter
        
    @pytest.mark.asyncio
    @patch("schema_agents.agents.CodeAgent.run")
    async def test_run_with_overridden_tools(self, mock_run, mock_model, test_tool):
        """Test running the agent with overridden tools."""
        # Set up the mock to return a string
        mock_run.return_value = "Final Answer: Test result with overridden tools"
        
        # Create an agent with a default tool
        agent = Agent(
            model=mock_model,
            name="TestAgent",
            tools=[test_tool],
        )
        
        # Store the original tools dict
        original_tools = agent.tools.copy()
        
        # Create a new tool for this run only
        from schema_agents.tools import Tool
        new_tool = MagicMock(spec=Tool)
        new_tool.name = "new_tool"
        new_tool.description = "A new tool just for this run"
        new_tool.inputs = {"param": {"type": "string", "description": "A parameter"}}
        new_tool.output_type = "string"
        
        # Run the agent with overridden tools
        result = await agent.run("Test query", tools=[new_tool])
        
        # Check that the result is correct
        assert result == "Final Answer: Test result with overridden tools"
        
        # Verify that the parent run method was called
        assert mock_run.call_count == 1
        
        # Verify that the original agent's tools were not modified
        assert agent.tools == original_tools
        assert "new_tool" not in agent.tools
        assert "test_tool" in agent.tools
        
    @pytest.mark.asyncio
    @patch("schema_agents.agents.CodeAgent.run")
    async def test_run_with_overridden_result_type(self, mock_run, mock_model):
        """Test running the agent with an overridden result type."""
        # Set up the mock to return a string
        mock_run.return_value = "42"
        
        # Create an agent with string result type
        agent = Agent(
            model=mock_model,
            name="TestAgent",
            result_type=str,
        )
        
        # Store the original result_type
        original_result_type = agent.result_type
        
        # Create a custom result type for this run only
        class CustomResult:
            def __init__(self, value):
                self.value = int(value)
        
        # Run the agent with overridden result type
        result = await agent.run("Test query", result_type=CustomResult)
        
        # Check that the result is of the custom type
        assert isinstance(result, CustomResult)
        assert result.value == 42
        
        # Verify that the parent run method was called
        assert mock_run.call_count == 1
        
        # Verify that the original agent's result_type was not modified
        assert agent.result_type == original_result_type
        assert agent.result_type == str
        
    @pytest.mark.asyncio
    @patch("schema_agents.agents.CodeAgent.run")
    async def test_run_with_pydantic_result_type_override(self, mock_run, mock_model):
        """Test running the agent with an overridden Pydantic result type."""
        # Define a Pydantic model for the result
        class QueryResult(BaseModel):
            answer: str = Field(description="The answer to the query")
            confidence: float = Field(description="Confidence score (0-1)")
        
        # Create a QueryResult instance to return from the mock
        expected_result = QueryResult(
            answer="The answer is 42",
            confidence=0.95,
        )
        
        # Set up the mock to return the QueryResult instance
        mock_run.return_value = expected_result
        
        # Create an agent with string result type
        agent = Agent(
            model=mock_model,
            name="TestAgent",
            result_type=str,
        )
        
        # Store the original result_type
        original_result_type = agent.result_type
        
        # Run the agent with overridden result type
        result = await agent.run("Test query", result_type=QueryResult)
        
        # Check that the result is a QueryResult instance
        assert isinstance(result, QueryResult)
        assert result.answer == "The answer is 42"
        assert result.confidence == 0.95
        
        # Verify that the parent run method was called
        assert mock_run.call_count == 1
        
        # Verify that the original agent's result_type was not modified
        assert agent.result_type == original_result_type
        
    @pytest.mark.asyncio
    @patch("schema_agents.agents.CodeAgent.run")
    async def test_run_with_additional_kwargs(self, mock_run, mock_model):
        """Test running the agent with additional kwargs passed to the parent run method."""
        # Set up the mock to return a string
        mock_run.return_value = "Final Answer: Test result with additional kwargs"
        
        # Create an agent
        agent = Agent(
            model=mock_model,
            name="TestAgent",
        )
        
        # Run the agent with additional kwargs
        result = await agent.run(
            "Test query",
            stream=True,
            reset=False,
            images=["image1.png", "image2.png"],
            additional_args={"key1": "value1", "key2": "value2"},
            max_steps=10,
        )
        
        # Check that the result is correct
        assert result == "Final Answer: Test result with additional kwargs"
        
        # Verify that the parent run method was called with the correct arguments
        mock_run.assert_called_once_with(
            "Test query",
            stream=True,
            reset=False,
            images=["image1.png", "image2.png"],
            additional_args={"key1": "value1", "key2": "value2"},
            max_steps=10,
        )
        
    @pytest.mark.asyncio
    @patch("schema_agents.agents.CodeAgent.run")
    async def test_run_with_combined_overrides(self, mock_run, mock_model, test_tool):
        """Test running the agent with both tools and result_type overridden, plus additional kwargs."""
        # Set up the mock to return a string
        mock_run.return_value = "42"
        
        # Create an agent
        agent = Agent(
            model=mock_model,
            name="TestAgent",
            tools=[test_tool],
        )
        
        # Store the original tools and result_type
        original_tools = agent.tools.copy()
        original_result_type = agent.result_type
        
        # Create a new tool for this run only
        from schema_agents.tools import Tool
        new_tool = MagicMock(spec=Tool)
        new_tool.name = "new_tool"
        new_tool.description = "A new tool just for this run"
        new_tool.inputs = {"param": {"type": "string", "description": "A parameter"}}
        new_tool.output_type = "string"
        
        # Create a custom result type for this run only
        class CustomResult:
            def __init__(self, value):
                self.value = int(value)
        
        # Run the agent with overridden tools, result_type, and additional kwargs
        result = await agent.run(
            "Test query",
            tools=[new_tool],
            result_type=CustomResult,
            stream=True,
            reset=False,
            max_steps=10,
        )
        
        # Check that the result is of the custom type
        assert isinstance(result, CustomResult)
        assert result.value == 42
        
        # Verify that the parent run method was called
        assert mock_run.call_count == 1
        
        # Verify that the original agent's tools and result_type were not modified
        assert agent.tools == original_tools
        assert agent.result_type == original_result_type
        assert "new_tool" not in agent.tools
        assert "test_tool" in agent.tools
        
    @pytest.mark.asyncio
    @patch("schema_agents.agents.CodeAgent.run")
    async def test_concurrent_runs_with_different_overrides(self, mock_run, mock_model, test_tool):
        """Test that concurrent runs with different overrides don't interfere with each other."""
        # Set up different return values for different calls
        mock_run.side_effect = ["42", "hello", "true"]
        
        # Create an agent
        agent = Agent(
            model=mock_model,
            name="TestAgent",
            tools=[test_tool],
        )
        
        # Store the original tools and result_type
        original_tools = agent.tools.copy()
        original_result_type = agent.result_type
        
        # Create different tools and result types for concurrent runs
        from schema_agents.tools import Tool
        
        tool1 = MagicMock(spec=Tool)
        tool1.name = "tool1"
        tool1.description = "Tool 1"
        tool1.inputs = {"param": {"type": "string", "description": "A parameter"}}
        tool1.output_type = "string"
        
        tool2 = MagicMock(spec=Tool)
        tool2.name = "tool2"
        tool2.description = "Tool 2"
        tool2.inputs = {"param": {"type": "string", "description": "A parameter"}}
        tool2.output_type = "string"
        
        class IntResult:
            def __init__(self, value):
                self.value = int(value)
                
        class StringResult:
            def __init__(self, value):
                self.value = str(value)
                
        class BoolResult:
            def __init__(self, value):
                self.value = value.lower() == "true"
        
        # Run three concurrent tasks with different overrides
        import asyncio
        tasks = [
            agent.run("Query 1", tools=[tool1], result_type=IntResult),
            agent.run("Query 2", tools=[tool2], result_type=StringResult),
            agent.run("Query 3", result_type=BoolResult),
        ]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Check that each result has the correct type and value
        assert isinstance(results[0], IntResult)
        assert results[0].value == 42
        
        assert isinstance(results[1], StringResult)
        assert results[1].value == "hello"
        
        assert isinstance(results[2], BoolResult)
        assert results[2].value is True
        
        # Verify that the parent run method was called three times
        assert mock_run.call_count == 3
        
        # Verify that the original agent's tools and result_type were not modified
        assert agent.tools == original_tools
        assert agent.result_type == original_result_type
        assert "tool1" not in agent.tools
        assert "tool2" not in agent.tools
        assert "test_tool" in agent.tools 

    @pytest.mark.asyncio
    @patch("schema_agents.agents.CodeAgent.run")
    async def test_run_with_overridden_managed_agents(self, mock_run, mock_model, test_tool):
        """Test running the agent with overridden managed agents."""
        # Set up the mock to return a string
        mock_run.return_value = "Final Answer: Test result with overridden managed agents"
        
        # Create an agent
        agent = Agent(
            model=mock_model,
            name="TestAgent",
            tools=[test_tool],
        )
        
        # Store the original managed_agents
        original_managed_agents = agent.managed_agents.copy()
        
        # Create a mock managed agent
        mock_managed_agent = MagicMock()
        mock_managed_agent.name = "mock_agent"
        mock_managed_agent.description = "A mock managed agent"
        
        # Run the agent with overridden managed agents
        result = await agent.run("Test query", managed_agents=[mock_managed_agent])
        
        # Check that the result is correct
        assert result == "Final Answer: Test result with overridden managed agents"
        
        # Verify that the parent run method was called
        assert mock_run.call_count == 1
        
        # Verify that the original agent's managed_agents were not modified
        assert agent.managed_agents == original_managed_agents
        assert "mock_agent" not in agent.managed_agents
        
    @pytest.mark.asyncio
    @patch("schema_agents.agents.CodeAgent.run")
    async def test_run_with_overridden_state(self, mock_run, mock_model):
        """Test running the agent with overridden state."""
        # Set up the mock to return a string
        mock_run.return_value = "Final Answer: Test result with overridden state"
        
        # Create an agent
        agent = Agent(
            model=mock_model,
            name="TestAgent",
        )
        
        # Set some initial state
        agent.state = {"initial_key": "initial_value"}
        
        # Store the original state
        original_state = agent.state.copy()
        
        # Create a new state for this run only
        new_state = {"new_key": "new_value"}
        
        # Run the agent with overridden state
        result = await agent.run("Test query", state=new_state)
        
        # Check that the result is correct
        assert result == "Final Answer: Test result with overridden state"
        
        # Verify that the parent run method was called
        assert mock_run.call_count == 1
        
        # Verify that the original agent's state was not modified
        assert agent.state == original_state
        assert "new_key" not in agent.state
        assert "initial_key" in agent.state
        
    @pytest.mark.asyncio
    @patch("schema_agents.agents.CodeAgent.run")
    async def test_run_with_reset_memory(self, mock_run, mock_model):
        """Test running the agent with reset_memory parameter."""
        # Set up the mock to return a string
        mock_run.return_value = "Final Answer: Test result with reset_memory"
        
        # Create an agent
        agent = Agent(
            model=mock_model,
            name="TestAgent",
        )
        
        # Run the agent with reset_memory=True
        result = await agent.run("Test query", reset_memory=True)
        
        # Check that the result is correct
        assert result == "Final Answer: Test result with reset_memory"
        
        # Verify that the parent run method was called with reset=True
        mock_run.assert_called_once_with("Test query", reset=True)
        
        # Reset the mock
        mock_run.reset_mock()
        
        # Run the agent with reset_memory=False
        result = await agent.run("Test query", reset_memory=False)
        
        # Check that the result is correct
        assert result == "Final Answer: Test result with reset_memory"
        
        # Verify that the parent run method was called with reset=False
        mock_run.assert_called_once_with("Test query", reset=False)
        
    @pytest.mark.asyncio
    @patch("schema_agents.agents.CodeAgent.run")
    async def test_run_with_all_overrides(self, mock_run, mock_model, test_tool):
        """Test running the agent with all possible overrides."""
        # Set up the mock to return a string
        mock_run.return_value = "42"
        
        # Create an agent
        agent = Agent(
            model=mock_model,
            name="TestAgent",
            tools=[test_tool],
        )
        
        # Set some initial state
        agent.state = {"initial_key": "initial_value"}
        
        # Store the original values
        original_tools = agent.tools.copy()
        original_result_type = agent.result_type
        original_managed_agents = agent.managed_agents.copy()
        original_state = agent.state.copy()
        
        # Create overrides
        from schema_agents.tools import Tool
        new_tool = MagicMock(spec=Tool)
        new_tool.name = "new_tool"
        new_tool.description = "A new tool just for this run"
        new_tool.inputs = {"param": {"type": "string", "description": "A parameter"}}
        new_tool.output_type = "string"
        
        class CustomResult:
            def __init__(self, value):
                self.value = int(value)
                
        mock_managed_agent = MagicMock()
        mock_managed_agent.name = "mock_agent"
        mock_managed_agent.description = "A mock managed agent"
        
        new_state = {"new_key": "new_value"}
        
        # Run the agent with all overrides
        result = await agent.run(
            "Test query",
            tools=[new_tool],
            result_type=CustomResult,
            managed_agents=[mock_managed_agent],
            state=new_state,
            reset_memory=True,
            stream=True,
            max_steps=10,
        )
        
        # Check that the result is of the custom type
        assert isinstance(result, CustomResult)
        assert result.value == 42
        
        # Verify that the parent run method was called with the correct parameters
        mock_run.assert_called_once_with(
            "Test query",
            reset=True,
            stream=True,
            max_steps=10,
        )
        
        # Verify that the original agent's values were not modified
        assert agent.tools == original_tools
        assert agent.result_type == original_result_type
        assert agent.managed_agents == original_managed_agents
        assert agent.state == original_state
        assert "new_tool" not in agent.tools
        assert "test_tool" in agent.tools
        assert "mock_agent" not in agent.managed_agents
        assert "new_key" not in agent.state
        assert "initial_key" in agent.state 