import asyncio
import os
from typing import Optional

from pydantic import Field

# Import necessary components from schema_agents
from schema_agents.schema_tools import schema_tool
from schema_agents import Agent
from schema_agents.models import OpenAIServerModel

# --- Tool Definitions ---
# Define tools using the standalone @schema_tool decorator.
# These tools can be used by any agent.

@schema_tool
async def calculator(
    expression: str = Field(..., description="The mathematical expression to evaluate. e.g., '5 + 12 * 3'")
) -> str:
    """
    Evaluates a mathematical expression provided as a string.
    Handles basic arithmetic operations (+, -, *, /).
    """
    # Note: Using eval is generally unsafe due to potential code injection.
    # This is for demonstration purposes only. In a real application,
    # implement a safer evaluation method (e.g., using ast.literal_eval
    # or a dedicated math parsing library).
    print(f"🛠️ Calculator Tool: Evaluating '{expression}'")
    allowed_chars = "0123456789+-*/(). "
    if not all(c in allowed_chars for c in expression):
        return "Error: Invalid characters in expression. Only numbers and operators (+-*/(). ) are allowed."
    try:
        # Basic validation passed, proceed with eval
        result = eval(expression)
        print(f"✅ Calculator Tool: Result = {result}")
        return f"The result of '{expression}' is {float(result)}"
    except Exception as e:
        print(f"❌ Calculator Tool: Error evaluating '{expression}': {e}")
        return f"Error evaluating expression '{expression}': {e}"

@schema_tool
async def get_weather(
    location: str = Field(..., description="The city and state, e.g., San Francisco, CA, or a city like London or Paris"),
    unit: Optional[str] = Field(default="celsius", description="Temperature unit: 'celsius' or 'fahrenheit'")
) -> str:
    """
    Gets the current weather condition for a specified location.
    This is a simulated tool and provides fixed weather data for specific cities.
    """
    print(f"🛠️ Weather Tool: Getting weather for {location} in {unit}")
    unit_symbol = "°C" if unit == "celsius" else "°F"
    location_lower = location.lower()

    # Simulate weather data lookup
    if "london" in location_lower:
        temp = 15 if unit == "celsius" else 59
        condition = "cloudy with a chance of rain"
    elif "paris" in location_lower:
        temp = 22 if unit == "celsius" else 72
        condition = "sunny and pleasant"
    elif "tokyo" in location_lower:
        temp = 28 if unit == "celsius" else 82
        condition = "hot and humid"
    else:
        print(f"❌ Weather Tool: No data for {location}")
        return f"Sorry, I don't have weather information for {location}."

    weather_report = f"The current weather in {location.title()} is {temp}{unit_symbol} and {condition}."
    print(f"✅ Weather Tool: Report = {weather_report}")
    return weather_report

# --- Agent Configuration and Execution ---
async def main():
    """Configures and runs the schema agent."""
    print("--- Schema Agent Example with OpenAI ---")

    # 1. Check for OpenAI API Key
    # The OpenAIServerModel requires the API key to be set as an environment variable.
    if "OPENAI_API_KEY" not in os.environ:
        print("\n❌ Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key to run this example:")
        print("  export OPENAI_API_KEY='your-api-key'\n")
        return

    print("✅ OpenAI API Key found.")

    # 2. Define the Language Model
    # We'll use OpenAI's gpt-4o-mini model.
    print("🧠 Configuring Language Model: gpt-4o-mini")
    llm = OpenAIServerModel(model_id="gpt-4o-mini")

    # 3. Define the Tools
    # Pass the functions decorated with @schema_tool directly.
    # The agent expects the actual Tool objects, which are stored in the __tool__ attribute.
    tool_functions = [
        calculator,
        get_weather,
        # Add other @schema_tool decorated functions here
    ]
    tools = [func.__tool__ for func in tool_functions]
    print(f"🔧 Available Tools: {[tool.name for tool in tools]}")

    # 4. Create the Agent
    # Using the base Agent class instead of ToolCallingAgent
    print("🤖 Creating Agent...")
    agent = Agent(
        model=llm,
        tools=tools,
        verbosity_level=1, # Set to 1 (or higher) to see the agent's reasoning and tool calls
    )

    # 5. Define the Task
    # A task that requires using both the calculator and weather tools.
    task = "What is (5 + 13) * 2? Also, could you tell me the weather in Paris (in Fahrenheit) and Tokyo?"
    print(f"\n🚀 Running Agent for Task:")
    print(f"   '{task}'\n")

    # 6. Run the Agent
    try:
        # The agent will use the LLM to decide which tools to call,
        # execute them, and formulate a final answer.
        result = await agent.run(task)

        print("\n--- Agent Finished ---")
        print(f"💬 Final Answer:\n{result}")

    except Exception as e:
        print(f"\n--- Agent Error ---")
        print(f"❌ An error occurred during the agent run: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the asynchronous main function
    asyncio.run(main()) 