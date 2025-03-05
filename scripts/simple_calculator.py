"""
Simple Calculator Demo using Schema Agents

This example demonstrates a basic calculator agent with schema-driven tools:
1. Basic arithmetic operations
2. Memory storage and recall
3. Unit conversion
4. History tracking

Features demonstrated:
- Basic agent setup
- Schema-driven tools
- ReAct reasoning
- Structured outputs
"""

import os
import asyncio
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from pydantic_ai import RunContext, models
from pydantic_ai.models.openai import OpenAIModel
from schema_agents import Agent, ReasoningStrategy, ReActConfig
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Ensure OPENAI_API_KEY is set
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Define data structures
class CalculationResult(BaseModel):
    """Result of a calculation operation"""
    operation: str = Field(..., description="The operation performed")
    result: float = Field(..., description="The result of the calculation")
    expression: str = Field(..., description="The original expression")

class MemoryState(BaseModel):
    """Calculator memory state"""
    stored_value: Optional[float] = Field(None, description="Value stored in memory")
    history: List[CalculationResult] = Field(default_factory=list, description="Calculation history")
    
    async def add_to_history(self, prompt: str) -> None:
        """Add a prompt to history - required by the Agent framework"""
        # This method is required by the Agent framework
        pass

class UnitConversion(BaseModel):
    """Unit conversion result"""
    from_unit: str = Field(..., description="Original unit")
    to_unit: str = Field(..., description="Target unit")
    value: float = Field(..., description="Original value")
    result: float = Field(..., description="Converted value")

# Create the Calculator Agent
def create_calculator_agent(model: models.Model) -> Agent:
    agent = Agent(
        model=model,
        name="Calculator",
        deps_type=MemoryState,
        result_type=str,
        role="Mathematical Assistant",
        goal="Perform calculations and unit conversions accurately",
        backstory="I am a sophisticated calculator that can perform arithmetic operations and unit conversions while maintaining calculation history.",
        reasoning_strategy=ReasoningStrategy(
            type="react",
            react_config=ReActConfig(
                max_loops=5,
                min_confidence=0.8
            )
        )
    )
    
    @agent.schema_tool(takes_ctx=True)
    async def calculate(
        ctx: RunContext[MemoryState],
        x: int = Field(..., description="First number"),
        y: int = Field(..., description="Second number"),
        operation: str = Field(..., description="Operation to perform (add/subtract/multiply/divide)")
    ) -> CalculationResult:
        """Perform a basic arithmetic calculation"""
        # Convert to float for calculations
        x_float = float(x)
        y_float = float(y)
        
        result = None
        if operation == "add":
            result = x_float + y_float
        elif operation == "subtract":
            result = x_float - y_float
        elif operation == "multiply":
            result = x_float * y_float
        elif operation == "divide":
            if y_float == 0:
                raise ValueError("Cannot divide by zero")
            result = x_float / y_float
        else:
            raise ValueError(f"Unknown operation: {operation}")
            
        calc_result = CalculationResult(
            operation=operation,
            result=result,
            expression=f"{x_float} {operation} {y_float}"
        )
        
        # Add to history
        if not ctx.deps.history:
            ctx.deps.history = []
        ctx.deps.history.append(calc_result)
        
        return calc_result

    @agent.schema_tool(takes_ctx=True)
    async def store_memory(
        ctx: RunContext[MemoryState],
        value: int = Field(..., description="Value to store in memory")
    ) -> str:
        """Store a value in calculator memory"""
        # Convert to float for storage
        ctx.deps.stored_value = float(value)
        return f"Stored {value} in memory"

    @agent.schema_tool(takes_ctx=True)
    async def recall_memory(
        ctx: RunContext[MemoryState]
    ) -> float:
        """Recall the value stored in memory"""
        if ctx.deps.stored_value is None:
            raise ValueError("No value stored in memory")
        return ctx.deps.stored_value

    @agent.schema_tool(takes_ctx=True)
    async def convert_units(
        ctx: RunContext[MemoryState],
        value: int = Field(..., description="Value to convert"),
        from_unit: str = Field(..., description="Original unit (m/cm/km/in/ft)"),
        to_unit: str = Field(..., description="Target unit (m/cm/km/in/ft)")
    ) -> UnitConversion:
        """Convert between different units of length"""
        # Convert to float for calculations
        value_float = float(value)
        
        # Conversion factors to meters
        to_meters = {
            "m": 1,
            "cm": 0.01,
            "km": 1000,
            "in": 0.0254,
            "ft": 0.3048
        }
        
        if from_unit not in to_meters or to_unit not in to_meters:
            raise ValueError(f"Unsupported units. Supported units: {list(to_meters.keys())}")
            
        # Convert to meters first
        meters = value_float * to_meters[from_unit]
        # Then convert to target unit
        result = meters / to_meters[to_unit]
        
        return UnitConversion(
            from_unit=from_unit,
            to_unit=to_unit,
            value=value_float,
            result=result
        )

    @agent.schema_tool(takes_ctx=True)
    async def get_history(
        ctx: RunContext[MemoryState],
        last_n: Optional[int] = Field(None, description="Number of last entries to return")
    ) -> List[CalculationResult]:
        """Get calculation history"""
        if not ctx.deps.history:
            return []
        if last_n is not None:
            return ctx.deps.history[-last_n:]
        return ctx.deps.history

    return agent

async def main():
    try:
        # Create OpenAI model instance
        logger.info("Creating OpenAI model instance...")
        model = OpenAIModel(
            'gpt-4o-mini',
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Create calculator agent
        logger.info("Creating calculator agent...")
        calculator = create_calculator_agent(model)
        
        # Initialize memory state
        memory_state = MemoryState()
        
        # Example calculations
        examples = [
            "Calculate 15 plus 27",
            "Store the result in memory",
            "What's 5 times the value in memory?",
            "Convert 100 centimeters to feet",
            "Show me the last 3 calculations"
        ]
        
        for prompt in examples:
            logger.info(f"\nProcessing: {prompt}")
            result = await calculator.run(prompt, deps=memory_state)
            logger.info(f"Result: {result.data}")
            
        logger.info("\nDemo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main()) 