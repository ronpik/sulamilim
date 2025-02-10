from typing import Optional

import networkx as nx
import matplotlib.pyplot as plt
import imageio
import os
import itertools

from matplotlib import patches

# -------------------------------
# 1. Create the Full Graph (Word Network)
# -------------------------------
G = nx.Graph()
nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
# nodes = ['A', 'B', 'C', 'D', 'E', 'F']
G.add_nodes_from(nodes)
# Create a basic chain (word ladder)
edges_chain = [('A', 'B'), ('B', 'C'), ('C', 'D'),
               ('D', 'E'), ('E', 'F'), ('F', 'G'), ('G', 'H'), ('E', 'G')
               # ('F', 'D'),
               # , ('E', 'G')
               ]
G.add_edges_from(edges_chain)
# Add an extra edge to add some complexity.
G.add_edge('B', 'D')

# Use a fixed layout for the full graph.
pos_full = nx.spring_layout(G, seed=42)

G.remove_nodes_from(['H'])
G.add_edge('E', 'G')
# G.add_edge('B', 'F')


# -------------------------------
# 2. Define Two Graphlets and Their Layout
# -------------------------------
# Graphlet 1: a graph with one extra edge.
graphlet = nx.Graph()
graphlet_nodes = [0, 1, 2, 3, 4]
graphlet.add_nodes_from(graphlet_nodes)
graphlet_edges = [(0, 1), (1, 2), (2, 3), (1, 3), (3, 4)]
graphlet.add_edges_from(graphlet_edges)
pos_graphlet = nx.spring_layout(graphlet, seed=42)

# Graphlet 2: remove one edge from graphlet.
graphlet2 = graphlet.copy()
graphlet2.remove_edge(1, 3)
# They will use the same pos_graphlet layout.

# -------------------------------
# 3. Prepare for GIF Frame Saving
# -------------------------------
frames = []
output_dir = "gif_frames"
os.makedirs(output_dir, exist_ok=True)


def create_and_save_gif_frame(frame_num: int, title: str, candidate_nodes=None, candidate_color=None,
                              valid_sequences: Optional[list[str]] = None,
                              gl1_match: bool = False, gl2_match: bool = False):
    """
    Draws the main graph (optionally with candidate nodes highlighted),
    adds two inset panels showing the two graphlets side by side, adds a title
    and sequence text, then saves the frame.
    """
    plt.figure(figsize=(10, 8))
    # Draw the full graph.
    nx.draw(G, pos_full, with_labels=True, node_color="skyblue",
            edge_color="gray", node_size=800)

    # If candidate_nodes is provided, draw them in the specified candidate_color.
    if candidate_nodes and candidate_color:
        # Highlight candidate nodes and their induced edges.
        candidate_edges = list(G.subgraph(candidate_nodes).edges())
        nx.draw_networkx_nodes(G, pos_full, nodelist=candidate_nodes, node_color=candidate_color, node_size=800)
        nx.draw_networkx_edges(G, pos_full, edgelist=candidate_edges, edge_color=candidate_color, width=3)

    plt.title(title)

    # Add accumulated sequence text (if any)
    if valid_sequences:
        sequence_text = "\n".join(valid_sequences)
        plt.figtext(0.5, 0.02, sequence_text, ha="center", fontsize=12, color="green")

    # Draw two inset axes for the graphlets, placed side by side at the top right.
    # Inset for Graphlet 1.
    inset_ax1 = plt.axes([0.65, 0.65, 0.3, 0.3])
    gl1_color = 'lightgreen' if gl1_match else 'lightgray'
    nx.draw_networkx(graphlet, pos=pos_graphlet, ax=inset_ax1,
                     with_labels=False, node_color=gl1_color,
                     edge_color="black", node_size=300)
    if gl1_match:
        rect = patches.Rectangle((0, 0), 1, 1, transform=inset_ax1.transAxes,
                                 fill=False, edgecolor='black', linewidth=2)
        inset_ax1.add_patch(rect)
    inset_ax1.set_title("Graphlet 1", fontsize=10)
    inset_ax1.axis('off')

    # Inset for Graphlet 2.
    inset_ax2 = plt.axes([0.35, 0.65, 0.3, 0.3])
    gl2_color = 'lightgreen' if gl2_match else 'lightgray'
    nx.draw_networkx(graphlet2, pos=pos_graphlet, ax=inset_ax2,
                     with_labels=False, node_color=gl2_color,
                     edge_color="black", node_size=300)
    if gl2_match:
        rect = patches.Rectangle((0, 0), 1, 1, transform=inset_ax2.transAxes,
                                 fill=False, edgecolor='black', linewidth=2)
        inset_ax2.add_patch(rect)
    inset_ax2.set_title("Graphlet 2", fontsize=10)
    inset_ax2.axis('off')

    fname = os.path.join(output_dir, f"frame_{frame_num}.png")
    plt.savefig(fname, dpi=300)
    frames.append(fname)
    plt.close()


# -------------------------------
# 4. Process a Candidate 5-Node Subgraph
# -------------------------------
def process_candidate_subgraph(G, candidate_nodes, frame_counter, valid_sequences):
    """
    For a candidate (a tuple of 5 nodes), first add a frame with the candidate
    highlighted in yellow. Then check for isomorphism against graphlet or graphlet2.
    If isomorphic, add a frame with the candidate colored green and update valid_sequences;
    otherwise add a frame with the candidate colored red.

    Returns the next available frame counter.
    """
    candidate_str = " → ".join(candidate_nodes)

    # Frame A: Show candidate in yellow.
    create_and_save_gif_frame(
        frame_num=frame_counter,
        title=f"Candidate: {candidate_str} (Examining)",
        candidate_nodes=candidate_nodes,
        candidate_color="yellow",
        valid_sequences=valid_sequences
    )
    frame_counter += 1

    # Extract the candidate subgraph.
    subG = G.subgraph(candidate_nodes)
    # Check isomorphism against both graphlets.
    iso1 = nx.is_isomorphic(subG, graphlet)
    iso2 = nx.is_isomorphic(subG, graphlet2)
    if iso1 or iso2:
        valid_sequences.append(candidate_str)
        # Create a multi-line sequence text if there are previous valid sequences.
        create_and_save_gif_frame(
            frame_num=frame_counter,
            title=f"Candidate: {candidate_str} (Valid)",
            candidate_nodes=candidate_nodes,
            candidate_color="green",
            valid_sequences=valid_sequences,
            gl1_match=iso1, gl2_match=iso2
        )
    else:
        # Not valid candidate.
        sequence_text = "\n".join(valid_sequences)
        create_and_save_gif_frame(
            frame_num=frame_counter,
            title=f"Candidate: {candidate_str} (Not Valid)",
            candidate_nodes=candidate_nodes,
            candidate_color="red",
            valid_sequences=valid_sequences
        )
    frame_counter += 1
    return frame_counter


# -------------------------------
# 5. Iterate All Connected 5-Node Subgraphs
# -------------------------------
def iterate_connected_subgraphs(G):
    """
    Iterate over all connected subgraphs of 5 nodes in G.
    For each, process the candidate.
    """
    valid_sequences = []
    frame_counter = 0
    create_and_save_gif_frame(
        frame_num=frame_counter,
        title=f"",
        candidate_nodes=[],
        candidate_color="yellow",
        valid_sequences=valid_sequences
    )
    frame_counter += 1
    # Use itertools.combinations to generate 5-node combinations.
    for candidate in itertools.combinations(G.nodes(), 5):
        print(candidate)
        subG = G.subgraph(candidate)
        if nx.is_connected(subG):
            frame_counter = process_candidate_subgraph(G, candidate, frame_counter, valid_sequences)
    return frame_counter


# -------------------------------
# Run the Iteration and Create the GIF
# -------------------------------
final_frame = iterate_connected_subgraphs(G)

# Compile all frames into a GIF.
images = [imageio.imread(frame) for frame in frames]
gif_filename = "valid_sequence_simulation_full.gif"
imageio.mimsave(gif_filename, images, duration=1000)

print(f"GIF saved as {gif_filename}")
