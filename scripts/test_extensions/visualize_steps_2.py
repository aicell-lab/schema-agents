import matplotlib.pyplot as plt
import networkx as nx
import json

# Load your JSON data
with open('/Users/gkreder/schema-agents/metadata.json', 'r') as file:
    json_data = json.load(file)

G = nx.DiGraph()

def add_nodes_edges(data, parent=None):
    if isinstance(data, dict):
        G.add_node(json.dumps(data))
        if parent:
            G.add_edge(parent, json.dumps(data))
        for k, v in data.items():
            add_nodes_edges(v, parent=json.dumps(data))
    elif isinstance(data, list):
        for item in data:
            add_nodes_edges(item, parent)

add_nodes_edges(json_data)

pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
nx.draw(G, pos, with_labels=True, arrows=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
plt.show()