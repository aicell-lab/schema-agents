from dataclasses import dataclass
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
import asyncio

# Define complex input schemas for calculator operations
class NumberPair(BaseModel):
    first: float = Field(..., description="First number in the operation")
    second: float = Field(..., description="Second number in the operation")
    label: Optional[str] = Field(None, description="Optional label for the number pair")

class CalculationRequest(BaseModel):
    numbers: List[NumberPair] = Field(..., description="List of number pairs to operate on")
    operation_type: str = Field(..., description="Type of operation to perform (add/multiply)")
    description: Optional[str] = Field(None, description="Optional description of the calculation")

class TextProcessRequest(BaseModel):
    text: str = Field(..., description="Text to process")
    operation: str = Field(..., description="Operation to perform (uppercase/lowercase/reverse)")
    repeat: Optional[int] = Field(1, description="Number of times to repeat the operation")

# Create agent with dependencies
@dataclass
class CalculatorDeps:
    history: list = None
    
    def __init__(self):
        self.history = []

# Initialize OpenAI model
model = OpenAIModel(
    'qwen2.5-coder',
    base_url='https://hypha-ollama.scilifelab-2-dev.sys.kth.se/v1',
    api_key='ollama',
)

# Initialize the agent
calculator_agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=CalculatorDeps,
    result_type=str,
    system_prompt="""You are a helpful assistant that can:
    1. Perform calculations using process_calculation
    2. Process text using process_text
    Choose the appropriate tool based on the user's request."""
)

# Define calculator tools
@calculator_agent.tool
async def process_calculation(
    ctx: RunContext[CalculatorDeps],
    request: CalculationRequest
) -> str:
    """Process a calculation request with multiple number pairs"""
    results = []
    
    for pair in request.numbers:
        if request.operation_type == "add":
            result = pair.first + pair.second
            operation = "+"
        elif request.operation_type == "multiply":
            result = pair.first * pair.second
            operation = "*"
        else:
            raise ValueError(f"Unknown operation: {request.operation_type}")
        
        label = f" ({pair.label})" if pair.label else ""
        calculation = f"{pair.first} {operation} {pair.second} = {result}{label}"
        results.append(calculation)
        ctx.deps.history.append(calculation)
    
    description = f"\n{request.description}" if request.description else ""
    return f"Calculations performed:{description}\n" + "\n".join(results)

@calculator_agent.tool
async def process_text(
    ctx: RunContext[CalculatorDeps],
    request: TextProcessRequest
) -> str:
    """Process text with various operations"""
    result = request.text
    
    for _ in range(request.repeat):
        if request.operation == "uppercase":
            result = result.upper()
        elif request.operation == "lowercase":
            result = result.lower()
        elif request.operation == "reverse":
            result = result[::-1]
        else:
            raise ValueError(f"Unknown operation: {request.operation}")
        
    ctx.deps.history.append(f"Processed text: {result}")
    return f"Text processing result: {result}"

async def main():
    deps = CalculatorDeps()
    
    # Test prompts for different tools
    prompts = [
        """I need to do these calculations:
        1. Add 5302 and 332393 (first pair)
        2. Multiply 546 and 4316546 (second pair)
        Please show all results.""",
        
        """Please process this text:
        Make "Hello World" uppercase and repeat it twice."""
    ]
    
    for prompt in prompts:
        print(f"\nProcessing: {prompt}")
        async with calculator_agent.run_stream(prompt, deps=deps) as response:
            print("\nStreamed updates:")
            async for message, last in response.stream_structured(debounce_by=0.01):
                try:
                    partial_result = await response.validate_structured_result(
                        message,
                        allow_partial=not last
                    )
                    print(f"Partial result: {partial_result}")
                except ValidationError:
                    continue
            
            final_result = await response.get_data()
            print("\nFinal result:")
            print(final_result)
            
            print("\nOperation history:")
            for entry in deps.history:
                print(entry)
        
        print("\n" + "="*50)

if __name__ == "__main__":
    asyncio.run(main())
