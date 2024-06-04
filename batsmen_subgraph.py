import networkx as nx 
import os
import matplotlib.pyplot as plt 
from lib.pib import get_PIB_graph

bowling_career_file = 'data/stats/bowlers_averages.csv'
batsmen_vs_bowler_file = 'data/stats/batsmen_against_bowlers_averages.csv'
subgraph_dir = 'results/subgraph'

# Check if the results directory exists, if not, create it
if not os.path.exists(subgraph_dir):
    os.makedirs(subgraph_dir)

def plot_batsman_interactions(G, batsman, layout_type='circular_layout', top_n=15):
    # Ensure the batsman is in the graph
    if batsman not in G.nodes:
        raise ValueError(f"Batsman {batsman} is not in the graph.")
    
    # Get all interactions (bowlers) for the specified batsman
    interactions = [(neighbor, G[neighbor][batsman]['weight']) for neighbor in G.predecessors(batsman)]
    
    # Sort the interactions based on PIB (weight) and take the top N
    top_interactions = sorted(interactions, key=lambda x: x[1], reverse=True)[:top_n]
    top_bowlers = [bowler for bowler, _ in top_interactions]
    
    if not top_bowlers:
        print(f"No interactions found for batsman {batsman}.")
        return

    # Nodes to include in the subgraph
    nodes_to_include = [batsman] + top_bowlers
    H = G.subgraph(nodes_to_include)

    # Choose layout
    if layout_type == 'spring_layout':
        pos = nx.spring_layout(H)
    elif layout_type == 'circular_layout':
        pos = nx.circular_layout(H)
    elif layout_type == 'kamada_kawai_layout':
        pos = nx.kamada_kawai_layout(H)
    elif layout_type == 'spectral_layout':
        pos = nx.spectral_layout(H)
    elif layout_type == 'random_layout':
        pos = nx.random_layout(H)
    elif layout_type == 'shell_layout':
        pos = nx.shell_layout(H)
    else:
        raise ValueError("Unknown layout type")

    # Modify the positions to place the batsman at the center
    pos[batsman] = [0, 0]  # Center the main batsman

    # Plot the graph
    plt.figure(figsize=(12, 8))
    edges = H.edges(data=True)
    weights = [d['weight'] for (u, v, d) in edges]
    
    # Color the batsman node red and bowlers blue
    node_colors = ['red' if node == batsman else 'skyblue' for node in H.nodes()]
    
    nx.draw(H, pos, with_labels=True, node_size=8000, node_color=node_colors, font_size=10, font_color='black', font_weight='bold', edge_color='gray', width=weights)
    plt.title(f"Top {top_n} Interactions of {batsman} with Bowlers ({layout_type})")
    plt.savefig("results/subgraph/V_Kohli_subgraph.png")

# Example usage
G = get_PIB_graph(bowling_career_file, batsmen_vs_bowler_file)  # Your function to get the original PIB graph

# Specify the batsman you are interested in
batsman = "V Kohli"

# Plot the interactions of the specified batsman with the top 25 bowlers
plot_batsman_interactions(G, batsman, layout_type='circular_layout', top_n=15)