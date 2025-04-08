import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from schema_agents.schema_tools import schema_tool, create_tool_from_schema_function
from schema_agents.tools import Tool


class TestStandaloneSchemaTools:
    """Tests for the standalone schema_tool decorator."""
    
    def test_standalone_schema_tool_decorator(self):
        """Test that the standalone schema_tool decorator creates a valid tool."""
        
        # Define a tool using the standalone schema_tool decorator
        @schema_tool
        async def say_hello(
            name: str = Field(default="world", description="Name of the person to greet")
        ) -> str:
            """Say hello to someone"""
            return f"Hello {name}!"
        
        # Check that the tool was added as an attribute to the function
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
        
    def test_standalone_schema_tool_with_custom_name_description(self):
        """Test that the standalone schema_tool decorator works with custom name and description."""
        
        # Define a tool using the standalone schema_tool decorator with custom name and description
        @schema_tool(name="greeter", description="A tool that greets people")
        async def say_hello(
            name: str = Field(default="world", description="Name of the person to greet")
        ) -> str:
            """Say hello to someone"""
            return f"Hello {name}!"
        
        # Check that the tool was added as an attribute to the function
        assert hasattr(say_hello, "__tool__")
        
        # Check that the tool has the expected attributes
        tool = say_hello.__tool__
        assert isinstance(tool, Tool)
        assert tool.name == "greeter"
        assert tool.description == "A tool that greets people"
        assert "name" in tool.inputs
        assert tool.inputs["name"]["type"] == "string"
        assert tool.inputs["name"]["description"] == "Name of the person to greet"
        
    def test_standalone_schema_tool_with_required_param(self):
        """Test that the standalone schema_tool decorator handles required parameters correctly."""
        
        # Define a tool using the standalone schema_tool decorator with a required parameter
        @schema_tool
        async def calculate(
            x: int = Field(..., description="First number"),
            y: int = Field(default=10, description="Second number")
        ) -> int:
            """Perform a calculation"""
            return x + y
        
        # Check that the tool was added as an attribute to the function
        assert hasattr(calculate, "__tool__")
        
        # Check that the tool has the expected attributes
        tool = calculate.__tool__
        assert isinstance(tool, Tool)
        assert "x" in tool.inputs
        assert "y" in tool.inputs
        assert "nullable" not in tool.inputs["x"]  # Required parameter
        assert tool.inputs["y"]["nullable"] is True  # Optional parameter
        
    @pytest.mark.asyncio
    async def test_standalone_schema_tool_execution(self):
        """Test that the standalone schema_tool can be executed."""
        
        # Define a tool using the standalone schema_tool decorator
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
        
    def test_create_tool_from_schema_function(self):
        """Test that create_tool_from_schema_function creates a valid tool."""
        
        # Define a tool using the standalone schema_tool decorator
        @schema_tool
        async def multiply_numbers(
            x: int = Field(..., description="First number"),
            y: int = Field(default=10, description="Second number")
        ) -> int:
            """Multiply two numbers"""
            return x * y
        
        # Create a tool from the function
        tool = create_tool_from_schema_function(multiply_numbers)
        
        # Check that the tool has the expected attributes
        assert isinstance(tool, Tool)
        assert tool.name == "multiply_numbers"
        assert tool.description == "Multiply two numbers"
        assert "x" in tool.inputs
        assert "y" in tool.inputs
        assert "nullable" not in tool.inputs["x"]  # Required parameter
        assert tool.inputs["y"]["nullable"] is True  # Optional parameter
        
    def test_create_tool_from_non_schema_function(self):
        """Test that create_tool_from_schema_function raises an error for non-schema functions."""
        
        # Define a function without the schema_tool decorator
        async def divide_numbers(x: int, y: int) -> float:
            """Divide two numbers"""
            return x / y
        
        # Try to create a tool from the function
        with pytest.raises(ValueError):
            create_tool_from_schema_function(divide_numbers)
            
    @pytest.mark.asyncio
    async def test_standalone_schema_tool_with_pydantic_models(self):
        """Test that the standalone schema_tool can work with Pydantic models."""
        
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
            
        # Define a tool using the standalone schema_tool decorator with Pydantic models
        @schema_tool
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
        tool = process_person.__tool__
        
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
    async def test_standalone_schema_tool_with_nested_pydantic_models(self):
        """Test that the standalone schema_tool can work with nested Pydantic models."""
        
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
            
        # Define a tool using the standalone schema_tool decorator with nested Pydantic models
        @schema_tool
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
        tool = summarize_person.__tool__
        
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