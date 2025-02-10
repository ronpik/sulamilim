import math
import matplotlib.pyplot as plt
import networkx as nx
from typing import Iterator

from sulamilim.dataprep.utils.graphlets import generate_graphlets
from sulamilim.dataprep.utils.hamilton import all_hamiltonian_paths


def visualize_graphlets_in_grid(graphlets: Iterator[nx.Graph]) -> None:
    """
    Visualize all generated graphlets in a grid.
    Each graphlet is drawn in its own subplot.
    """
    # Convert the iterator to a list
    graphs = list(graphlets)
    num_graphs = len(graphs)

    if num_graphs == 0:
        print("No graphlets to display.")
        return

    # Determine grid dimensions (try to form a roughly square grid)
    cols = math.ceil(math.sqrt(num_graphs))
    rows = math.ceil(num_graphs / cols)

    # Create the subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # If there's only one subplot, wrap it in a list for consistency.
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    # Draw each graphlet in its own subplot
    for ax, G in zip(axes, graphs):
        pos = nx.spring_layout(G)
        nx.draw(
            G, pos, ax=ax,
            with_labels=True,
            node_color='skyblue',
            edge_color='gray',
            node_size=500
        )
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        n_paths = len(all_hamiltonian_paths(G))
        ax.set_title(f"Nodes: {n_nodes}, Edges: {n_edges}, H-Paths: {n_paths}")
        ax.axis('off')

    # Hide any extra subplots if the grid is larger than the number of graphs.
    for ax in axes[num_graphs:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    n_nodes = 4
    graphlets_iter = generate_graphlets(n_nodes)
    visualize_graphlets_in_grid(graphlets_iter)
