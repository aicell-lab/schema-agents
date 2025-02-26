from __future__ import annotations

import json
from typing import List, Dict, Any, Optional
from schema_agents.schema_reasoning import ReasoningState, MemoryType, ThoughtType, ActionType

class MemoryVisualizer:
    """Visualizes agent memory and reasoning state using Mermaid."""
    
    @staticmethod
    def create_memory_graph(state: ReasoningState) -> str:
        """Create a Mermaid graph visualization of memory state."""
        nodes = []
        edges = []
        
        # Add task node
        nodes.append('Task["{task}"]'.format(task=state.task.replace('"', "'")))
        
        # Add plan nodes
        if state.current_plan:
            nodes.append('Plan["Current Plan"]')
            edges.append('Task --> Plan')
            for i, step in enumerate(state.current_plan.steps):
                node_id = f"plan_{i}"
                nodes.append(f'{node_id}["{step}"]')
                edges.append(f'Plan --> {node_id}')
                if i == state.current_plan.current_step:
                    nodes.append(f'{node_id}:::current')
        
        # Add thought nodes
        for i, thought in enumerate(state.thoughts):
            node_id = f"thought_{i}"
            nodes.append(f'{node_id}["{thought.type}: {thought.content[:50]}..."]')
            if i > 0:
                edges.append(f'thought_{i-1} --> {node_id}')
            else:
                edges.append(f'Task --> {node_id}')
            
            # Add supporting facts
            if thought.supporting_facts:
                fact_group = f"fact_group_{i}"
                nodes.append(f'subgraph {fact_group}["Supporting Facts"]')
                for j, fact in enumerate(thought.supporting_facts):
                    fact_id = f"fact_{i}_{j}"
                    nodes.append(f'{fact_id}["{fact[:30]}..."]')
                    edges.append(f'{node_id} -.-> {fact_id}')
                nodes.append('end')
        
        # Add action nodes
        for i, action in enumerate(state.actions):
            node_id = f"action_{i}"
            nodes.append(f'{node_id}["{action.type}: {action.purpose[:50]}..."]')
            edges.append(f'thought_{i} --> {node_id}')
            
            # Add observation if exists
            if i < len(state.observations):
                obs_id = f"obs_{i}"
                nodes.append(f'{obs_id}["{state.observations[i].content[:50]}..."]')
                edges.append(f'{node_id} --> {obs_id}')
        
        # Create Mermaid graph
        mermaid = """graph TD
    classDef current fill:#f96;
    classDef fact fill:#e8f4f8;
    classDef thought fill:#f4e8f8;
    classDef action fill:#f8e8e8;
    classDef observation fill:#e8f8e8;
    """
        mermaid += "\n    " + "\n    ".join(nodes)
        mermaid += "\n    " + "\n    ".join(edges)
        
        return mermaid

    @staticmethod
    def create_memory_timeline(state: ReasoningState) -> str:
        """Create a Mermaid timeline of memory events."""
        timeline = """gantt
    title Memory Timeline
    dateFormat X
    axisFormat %s
    section Memory Events
"""
        
        # Sort memory entries by timestamp
        entries = sorted(state.memory, key=lambda x: x.timestamp)
        start_time = entries[0].timestamp if entries else 0
        
        # Add entries to timeline
        for entry in entries:
            relative_time = entry.timestamp - start_time
            duration = 1  # Fixed duration for visualization
            timeline += f'    {entry.type.value}: {relative_time}, {duration}\n'
            
        return timeline

    @staticmethod
    def create_state_diagram(state: ReasoningState) -> str:
        """Create a Mermaid state diagram of the reasoning process."""
        diagram = """stateDiagram-v2
    [*] --> Task
"""
        
        # Add transitions between thoughts
        for i, thought in enumerate(state.thoughts):
            if i == 0:
                diagram += f"    Task --> {thought.type}\n"
            else:
                diagram += f"    {state.thoughts[i-1].type} --> {thought.type}\n"
            
            # Add actions from thoughts
            if i < len(state.actions):
                action = state.actions[i]
                diagram += f"    {thought.type} --> {action.type}\n"
                
                # Add observations
                if i < len(state.observations):
                    diagram += f"    {action.type} --> Observation{i}\n"
                    if i + 1 < len(state.thoughts):
                        diagram += f"    Observation{i} --> {state.thoughts[i+1].type}\n"
        
        # Add final state if exists
        if state.final_answer:
            diagram += "    Conclude --> [*]\n"
            
        return diagram

    @staticmethod
    def create_confidence_chart(state: ReasoningState) -> str:
        """Create a Mermaid chart of confidence levels over time."""
        chart = """xychart-beta
    title "Confidence Over Time"
    x-axis ["""
        
        # Add x-axis labels (steps)
        steps = [str(i) for i in range(len(state.thoughts))]
        chart += ", ".join(steps)
        chart += """]
    y-axis "Confidence" 0 --> 1
    line ["""
        
        # Add confidence values
        confidences = [str(thought.confidence) for thought in state.thoughts]
        chart += ", ".join(confidences)
        chart += "]\n"
        
        return chart

def visualize_memory_state(state: ReasoningState, format: str = "all") -> Dict[str, str]:
    """Generate visualizations of memory state.
    
    Args:
        state: The reasoning state to visualize
        format: One of "graph", "timeline", "state", "confidence", or "all"
        
    Returns:
        Dictionary of visualization names and their Mermaid markup
    """
    visualizer = MemoryVisualizer()
    
    if format == "graph":
        return {"memory_graph": visualizer.create_memory_graph(state)}
    elif format == "timeline":
        return {"memory_timeline": visualizer.create_memory_timeline(state)}
    elif format == "state":
        return {"state_diagram": visualizer.create_state_diagram(state)}
    elif format == "confidence":
        return {"confidence_chart": visualizer.create_confidence_chart(state)}
    else:
        return {
            "memory_graph": visualizer.create_memory_graph(state),
            "memory_timeline": visualizer.create_memory_timeline(state),
            "state_diagram": visualizer.create_state_diagram(state),
            "confidence_chart": visualizer.create_confidence_chart(state)
        } 