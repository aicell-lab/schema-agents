import json
from graphviz import Digraph

def format_detail_value(value):
    """Format detail values for display in the graph, converting all types to strings."""
    if isinstance(value, str):
        return value
    elif value is None:
        return 'None'
    elif isinstance(value, (list, dict)):
        # For simplicity, convert lists and dicts to JSON strings but keep them concise
        return json.dumps(value, indent=None)
    else:
        # For other types (int, float, bool), convert directly to string
        return str(value)

def add_node_with_details(graph, node_name, details, prefix):
    """Helper function to add a node with detailed label information."""
    label_parts = [f"{prefix}{node_name}"]
    for key, value in details.items():
        # Skip 'steps' in 'kwargs' to avoid redundancy
        if key == 'kwargs' and 'steps' in value:
            value = {k: v for k, v in value.items() if k != 'steps'}
        formatted_value = format_detail_value(value)
        label_parts.append(f"{key}: {formatted_value}")
    graph.node(f"{prefix}{node_name}", label="\n".join(label_parts), shape="box")


def build_graph(steps, graph, parent=None, prefix=''):
    """Recursively traverse the JSON structure and add nodes/edges to the graph."""
    if isinstance(steps, list):
        for i, step in enumerate(steps):
            current_prefix = f"{prefix}{i+1}."
            build_graph(step, graph, parent, current_prefix)
    elif isinstance(steps, dict):
        node_name = steps.get('name', 'Unnamed')
        details = {k: v for k, v in steps.items() if k not in ['name', 'steps', 'result']}
        new_parent = f"{prefix}{node_name}"

        add_node_with_details(graph, node_name, details, prefix)
        
        if parent:
            graph.edge(parent, new_parent)

        nested_prefix = f"{prefix}{node_name}."  # Define this for nested steps
        
        if 'kwargs' in steps and 'steps' in steps['kwargs']:
            build_graph(steps['kwargs']['steps'], graph, new_parent, nested_prefix)
        
        if 'steps' in steps:
            build_graph(steps['steps'], graph, new_parent, nested_prefix)
        
        if 'result' in steps and isinstance(steps['result'], list) and len(steps['result']) > 1 and 'steps' in steps['result'][1]:
            build_graph(steps['result'][1]['steps'], graph, new_parent, nested_prefix)

def visualize_reasoning_chain(json_data, file_path_gv : str, format = 'svg', view = False):
    """Visualize the reasoning chain from the JSON data."""
    graph = Digraph(comment='Reasoning Chain', format = format)
    
    if 'steps' in json_data:
        build_graph(json_data['steps'], graph)
    
    graph.render(file_path_gv, view = view)


if __name__ == "__main__":
    # Load the JSON content from the file
    with open('/Users/gkreder/schema-agents/metadata.json') as f:
        json_data = json.load(f)
    visualize_reasoning_chain(json_data, file_path_gv='/Users/gkreder/schema-agents/reasoning_chain_visualization.gv', view = True)
