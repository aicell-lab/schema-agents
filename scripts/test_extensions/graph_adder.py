import json
import networkx as nx
import matplotlib.pyplot as plt

def add_steps_to_graph(G, steps, parent=None, prev_step=None):
    for i_step, step in enumerate(steps):
        if "name" in step:
            step_name = f'({i_step}) {step["name"]}'
        else:
            step_name = f"Step_{i_step}"
        G.add_node(step_name)
        
        # Connect this step to the previous step, if there is one
        if prev_step is not None:
            G.add_edge(prev_step, step_name)
        elif parent is not None:
            # If this is the first step in a nested list, connect it to the parent
            G.add_edge(parent, step_name)
        
        # Update prev_step for the next iteration
        prev_step = step_name
        
        # Handle "StartNewPlan" with nested steps
        if step_name == "StartNewPlan" and "steps" in step["kwargs"]:
            nested_steps = step["kwargs"]["steps"]
            add_steps_to_graph(G, nested_steps, step_name)
            
    return G  # Return the graph for further use if needed

def plot_metdata(metadata : dict):
    G = nx.DiGraph()
    steps_flattened = steps_flattened = [item for sublist in metadata['steps'] for item in sublist]
    add_steps_to_graph(G, metadata["steps"], None)
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
    plt.show()
