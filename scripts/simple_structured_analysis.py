"""
Simple Structured Data Analysis using Schema Agents with Structured Reasoning

This example demonstrates a single agent for data analysis using structured reasoning:
1. DataPreprocessor Agent: Handles data loading and preprocessing

Features demonstrated:
- Schema-based tools
- Structured reasoning strategy
- Structured outputs
"""

import os
import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import RunContext, models
from pydantic_ai.models.openai import OpenAIModel
from schema_agents import Agent
from schema_agents.reasoning import ReasoningStrategy, StructuredReasoningConfig, ReActConfig
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure all loggers show INFO level messages
logging.getLogger().setLevel(logging.INFO)
for handler in logging.getLogger().handlers:
    handler.setLevel(logging.INFO)

# Load environment variables (ensure OPENAI_API_KEY is set)
from dotenv import load_dotenv
load_dotenv()

# Ensure OPENAI_API_KEY is set
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Define data structures
class DataStats(BaseModel):
    """Statistical summary of a dataset"""
    shape: tuple = Field(..., description="Dataset dimensions (rows, columns)")
    columns: List[str] = Field(..., description="Column names")
    summary: Dict[str, Dict[str, float]] = Field(..., description="Statistical summary per column")
    missing_values: Dict[str, int] = Field(..., description="Missing values count per column")

@dataclass
class AgentDeps:
    """Shared dependencies for all agents"""
    data: Optional[pd.DataFrame] = None
    stats: Optional[DataStats] = None
    history: List[str] = None
    memory: List[str] = None
    
    def __init__(self):
        self.history = []
        self.memory = []
    
    async def add_to_history(self, entry: str):
        self.history.append(entry)
        self.memory.append(entry)  # Add to memory as well
        logger.debug(f"Added to history: {entry}")

# Create the DataPreprocessor Agent
def create_preprocessor_agent(model: models.Model) -> Agent:
    logger.info("Creating preprocessor agent...")
    agent = Agent(
        model=model,
        name="DataPreprocessor",
        deps_type=AgentDeps,
        result_type=DataStats,
        role="Data Preprocessing Specialist",
        goal="Prepare and validate data for analysis",
        backstory="You are an expert in data preprocessing with years of experience in handling various data formats and quality issues.",
        reasoning_strategy=ReasoningStrategy(
            type="react",  # Use ReAct instead of structured reasoning
            react_config=ReActConfig(
                max_iterations=5,
                memory_enabled=True,
                summarize_memory=True
            )
        )
    )
    
    @agent.tool
    async def load_csv(ctx: RunContext[AgentDeps], file_path: str) -> str:
        """Load data from a CSV file"""
        try:
            abs_path = os.path.abspath(file_path)
            logger.info(f"Loading data from {abs_path}")
            
            ctx.deps.data = pd.read_csv(abs_path)
            logger.info(f"Successfully loaded data with shape {ctx.deps.data.shape}")
            return f"Loaded data with shape {ctx.deps.data.shape}"
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            return f"Error loading file: {str(e)}"
    
    logger.info("Registered load_csv tool")
    
    @agent.tool
    async def compute_stats(ctx: RunContext[AgentDeps]) -> DataStats:
        """Compute statistical summary of the dataset"""
        if ctx.deps.data is None:
            logger.error("No data loaded")
            raise ValueError("No data loaded")
        
        df = ctx.deps.data
        logger.info("Computing dataset statistics")
        
        stats = DataStats(
            shape=df.shape,
            columns=df.columns.tolist(),
            summary={col: df[col].describe().to_dict() for col in df.select_dtypes(include=[np.number]).columns},
            missing_values=df.isnull().sum().to_dict()
        )
        ctx.deps.stats = stats
        logger.info("Successfully computed statistics")
        return stats
    
    logger.info("Registered compute_stats tool")
    
    @agent.tool
    async def clean_data(ctx: RunContext[AgentDeps], columns: List[str], strategy: str = "mean") -> str:
        """Clean data by handling missing values. Strategy can be: mean/median/drop"""
        if ctx.deps.data is None:
            logger.error("No data loaded")
            raise ValueError("No data loaded")
        
        df = ctx.deps.data.copy()
        logger.info(f"Cleaning data using {strategy} strategy for columns {columns}")
        
        for col in columns:
            if col not in df.columns:
                logger.error(f"Column {col} not found in dataset")
                return f"Column {col} not found in dataset"
            
            if strategy == "mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == "median":
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == "drop":
                df.dropna(subset=[col], inplace=True)
        
        ctx.deps.data = df
        logger.info("Successfully cleaned data")
        return f"Cleaned data using {strategy} strategy for columns {columns}"
    
    logger.info("Registered clean_data tool")
    logger.info(f"Available tools: {[tool.name for tool in agent._function_tools.values()]}")
    
    return agent

async def main():
    try:
        # Create OpenAI model instance
        print("Creating OpenAI model instance...")
        model = OpenAIModel(
            'gpt-4o-mini',
            api_key=os.getenv('OPENAI_API_KEY')
        )
        print("OpenAI model instance created")
        
        # Initialize shared dependencies
        deps = AgentDeps()
        
        # Create preprocessor agent
        print("Creating agent...")
        preprocessor = create_preprocessor_agent(model)
        print("Agent created successfully")
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        preprocess_result = await preprocessor.run(
            """
            Please analyze the iris dataset:
            1. Load the iris dataset from data/iris.csv
            2. Calculate statistics on the loaded data
            3. Check for missing values and clean if necessary
            4. Return the computed statistics
            """,
            deps=deps
        )
        
        # Print the result
        print(f"Preprocessing result: {preprocess_result.data}")
        
        logger.info("Simple analysis workflow completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main()) 