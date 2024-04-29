import json
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt

with open("../../responses.json") as f:
    data = json.load(f)

# Assuming 'data' is your loaded JSON content

def parse_json_for_hierarchy(data, parent_name='', level=0, parent_position=0, positions=dict(), edges=list(), width=1):
    if isinstance(data, dict):
        for index, (key, value) in enumerate(data.items()):
            node_name = f"{parent_name}.{key}" if parent_name else key
            if parent_name:  # Avoid adding edge for the root node
                edges.append((parent_name, node_name))
            # Calculate horizontal position
            horizontal_position = parent_position + (index - len(data)/2) * width/(level+1)
            positions[node_name] = (horizontal_position, -level)
            parse_json_for_hierarchy(value, node_name, level+1, horizontal_position, positions, edges, width/len(data))
    elif isinstance(data, list):
        for index, item in enumerate(data):
            node_name = f"{parent_name}.{index}"
            edges.append((parent_name, node_name))
            horizontal_position = parent_position + (index - len(data)/2) * width/(level+1)
            positions[node_name] = (horizontal_position, -level)
            parse_json_for_hierarchy(item, node_name, level+1, horizontal_position, positions, edges, width/len(data))
    else:
        node_name = f"{parent_name}.{data}"
        positions[node_name] = (parent_position, -level)
        if parent_name:
            edges.append((parent_name, node_name))

positions = {}
edges = []
parse_json_for_hierarchy(data, positions=positions, edges=edges)

# Convert positions to lists for Plotly
node_x = [pos[0] for pos in positions.values()]
node_y = [pos[1] for pos in positions.values()]
edge_x = []
edge_y = []
for edge in edges:
    x0, y0 = positions[edge[0]]
    x1, y1 = positions[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

# Create traces
edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=[name.split('.')[-1] for name in positions.keys()], hoverinfo='text', marker=dict(showscale=False, color='lightblue', size=10), textposition="bottom center")

# Create figure
fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

fig.show()
