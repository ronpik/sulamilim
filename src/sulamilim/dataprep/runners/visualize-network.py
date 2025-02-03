import argparse
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go
from globalog import LOG


def load_and_layout_network(file_path):
    """
    Load network from GraphML and compute layout using NetworkX's ForceAtlas2.

    Args:
        file_path (str): Path to the GraphML file

    Returns:
        tuple: (networkx.Graph, dict of node positions)
    """
    # Load the network
    LOG.info(f"Loading network from {file_path}")
    G = nx.read_graphml(file_path)

    LOG.info("Computing layout using NetworkX's ForceAtlas2.")
    # Compute layout using NetworkX's ForceAtlas2
    pos = nx.forceatlas2_layout(
        G,
        max_iter=100,
        scaling_ratio=2.0,
        gravity=1.0,
        strong_gravity=False,
        jitter_tolerance=1.0
    )

    LOG.info(f"Finished computing layout using NetworkX's ForceAtlas2.")
    return G, pos

def create_network_visualization(G, pos, output_file='network_viz.html'):
    """
    Create interactive network visualization using Plotly.

    Args:
        G (networkx.Graph): Network graph
        pos (dict): Dictionary of node positions
        output_file (str): Path for output HTML file
    """
    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create node trace
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    # Calculate node degrees for size and color
    node_degrees = dict(G.degree())
    node_sizes = [5 + v * 2 for v in node_degrees.values()]
    node_colors = list(node_degrees.values())

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_sizes,
            color=node_colors,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                # titleside='right'
            ),
            line_width=2))

    # Add node text for hover
    node_text = list(G.nodes())
    node_trace.text = node_text

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Word Network Visualization',
                       # titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002 ) ],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )

    # Save to HTML file
    fig.write_html(output_file)
    print(f"Visualization saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Visualize word network')
    parser.add_argument('--input_file', '-p', type=Path, required=True, help='Path to the GraphML network file')
    parser.add_argument('--output', default='network_viz.html',
                        help='Output HTML file path (default: network_viz.html)')

    args = parser.parse_args()

    # Load network and compute layout
    print("Loading network and computing layout...")
    G, pos = load_and_layout_network(args.input_file)

    # Create visualization
    print("Creating visualization...")
    create_network_visualization(G, pos, args.output)

    # Print some network statistics
    print("\nNetwork Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")


if __name__ == "__main__":
    main()