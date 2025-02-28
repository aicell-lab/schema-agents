"""
Simplified Collaborative Data Analysis System using Schema Agents with ReAct Reasoning

This example demonstrates a multi-agent system for collaborative data analysis using ReAct reasoning:
1. DataPreprocessor Agent: Handles data loading and preprocessing
2. Analyzer Agent: Performs statistical analysis and visualization

Features demonstrated:
- Schema-based tools
- ReAct reasoning strategy
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
from schema_agents.reasoning import ReasoningStrategy, ReActConfig
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

class AnalysisResult(BaseModel):
    """Analysis results including visualizations"""
    title: str = Field(..., description="Analysis title")
    description: str = Field(..., description="Analysis description")
    findings: List[str] = Field(..., description="Key findings")
    plots: List[str] = Field(default_factory=list, description="Base64 encoded plot images")
    correlations: Optional[Dict[str, Dict[str, float]]] = Field(None, description="Correlation analysis results")

@dataclass
class AgentDeps:
    """Shared dependencies for all agents"""
    data: Optional[pd.DataFrame] = None
    stats: Optional[DataStats] = None
    analysis: Optional[AnalysisResult] = None
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
            type="react",
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

# Create the Analyzer Agent
def create_analyzer_agent(model: models.Model) -> Agent:
    logger.info("Creating analyzer agent...")
    agent = Agent(
        model=model,
        name="Analyzer",
        deps_type=AgentDeps,
        result_type=AnalysisResult,
        role="Data Analysis Expert",
        goal="Perform in-depth statistical analysis and create visualizations",
        backstory="You are a skilled data analyst with expertise in statistical analysis and data visualization.",
        reasoning_strategy=ReasoningStrategy(
            type="react",
            react_config=ReActConfig(
                max_iterations=5,
                memory_enabled=True,
                summarize_memory=True
            )
        )
    )
    
    @agent.tool
    async def create_plot(
        ctx: RunContext[AgentDeps],
        plot_type: str,
        x_column: str,
        y_column: Optional[str] = None,
        title: str = "Plot"
    ) -> str:
        """Create a visualization and save it to a file. plot_type can be: scatter/bar/line/hist"""
        if ctx.deps.data is None:
            logger.error("No data loaded for plotting")
            raise ValueError("No data loaded")
        
        logger.info(f"Creating {plot_type} plot: {title}")
        plt.figure(figsize=(6, 4))  # Reduced figure size
        
        try:
            if plot_type == "scatter":
                if y_column is None:
                    logger.error("y_column is required for scatter plots")
                    raise ValueError("y_column is required for scatter plots")
                plt.scatter(ctx.deps.data[x_column], ctx.deps.data[y_column], s=20)  # Reduced marker size
            elif plot_type == "bar":
                ctx.deps.data[x_column].value_counts().plot(kind='bar')
            elif plot_type == "line":
                if y_column is None:
                    logger.error("y_column is required for line plots")
                    raise ValueError("y_column is required for line plots")
                plt.plot(ctx.deps.data[x_column], ctx.deps.data[y_column], linewidth=1)  # Reduced line width
            elif plot_type == "hist":
                plt.hist(ctx.deps.data[x_column], bins=20)  # Reduced number of bins
            else:
                logger.error(f"Unsupported plot type: {plot_type}")
                raise ValueError(f"Unsupported plot type: {plot_type}")
            
            plt.title(title)
            plt.xlabel(x_column)
            if y_column:
                plt.ylabel(y_column)
            
            # Create plots directory if it doesn't exist
            os.makedirs('plots', exist_ok=True)
            
            # Generate a unique filename based on plot type and title
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"plots/{plot_type}_{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            # Save the plot to file
            plt.savefig(filename, format='png', bbox_inches='tight', dpi=100)  # Reduced DPI
            plt.close()
            
            logger.info(f"Successfully saved plot to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error creating plot: {str(e)}")
            plt.close()
            raise
    
    @agent.tool
    async def correlation_analysis(ctx: RunContext[AgentDeps], columns: List[str]) -> Dict[str, Dict[str, float]]:
        """Compute correlation matrix for specified columns"""
        if ctx.deps.data is None:
            logger.error("No data loaded for correlation analysis")
            raise ValueError("No data loaded")
        
        logger.info(f"Computing correlations for columns: {columns}")
        
        # Validate columns
        for col in columns:
            if col not in ctx.deps.data.columns:
                logger.error(f"Column not found: {col}")
                raise ValueError(f"Column not found: {col}")
            if not np.issubdtype(ctx.deps.data[col].dtype, np.number):
                logger.error(f"Column {col} is not numeric")
                raise ValueError(f"Column {col} is not numeric")
        
        # Create a more structured correlation dictionary
        df_corr = ctx.deps.data[columns].corr().round(3)
        
        # Convert to a nested dictionary structure that matches the expected format
        corr_matrix = {}
        for col1 in columns:
            for col2 in columns:
                if col1 != col2:
                    # Create a key for the correlation pair
                    key = f"{col1}_{col2}"
                    # Store the correlation value in a nested dictionary
                    corr_matrix[key] = {"value": df_corr.loc[col1, col2]}
        
        logger.info("Successfully computed correlation matrix")
        return corr_matrix
    
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
        
        # Create agents
        print("Creating agents...")
        preprocessor = create_preprocessor_agent(model)
        analyzer = create_analyzer_agent(model)
        print("All agents created successfully")
        
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
        
        # Perform analysis
        logger.info("Performing analysis...")
        analysis_result = await analyzer.run(
            f"""
            Please analyze the iris dataset using the exact column names:
            - 'sepal length (cm)'
            - 'sepal width (cm)'
            - 'petal length (cm)'
            - 'petal width (cm)'
            - 'species'
            
            1. Create scatter plots for 'sepal length (cm)' vs 'sepal width (cm)' and 'petal length (cm)' vs 'petal width (cm)'
            2. Create histograms for each numeric feature
            3. Perform correlation analysis between all numeric columns
            4. Identify key patterns and insights
            5. Return the analysis results
            """,
            deps=deps
        )
        
        # Print the result
        print(f"Analysis result: {analysis_result.data}")
        
        logger.info("Analysis workflow completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main()) 