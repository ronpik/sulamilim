import networkx as nx
import matplotlib.pyplot as plt
import imageio
import os

# -------------------------------
# 1. Create the Main Graph (Word Network)
# -------------------------------
G = nx.Graph()
nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
G.add_nodes_from(nodes)
# Create a basic chain (word ladder)
edges_chain = [('A', 'B'), ('B', 'C'), ('C', 'D'),
               ('D', 'E'), ('E', 'F'), ('F', 'G'), ('G', 'H')]
G.add_edges_from(edges_chain)
# Add an extra edge to increase complexity (this will affect some candidates)
G.add_edge('B', 'D')

# Use a fixed layout for the main graph for consistency.
pos = nx.spring_layout(G, seed=42)

# -------------------------------
# 2. Define the Valid Graphlet
# -------------------------------
# The graphlet represents the ideal valid subgraph: a simple path on 5 nodes.
graphlet = nx.Graph()
graphlet_nodes = [0, 1, 2, 3, 4]
graphlet.add_nodes_from(graphlet_nodes)
graphlet_edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
graphlet.add_edges_from(graphlet_edges)
# Define a fixed layout for the inset drawing (a horizontal line).
pos_graphlet = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0), 4: (4, 0)}

# -------------------------------
# 3. Prepare for GIF Frame Saving
# -------------------------------
frames = []
output_dir = "gif_frames"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# Frame 0: Display the Complete Word Network
# -------------------------------
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color="skyblue",
        edge_color="gray", node_size=800)
plt.title("Complete Word Network")
fname = os.path.join(output_dir, "frame_0.png")
plt.savefig(fname)
frames.append(fname)
plt.close()

# -------------------------------
# Frame 1: Process Candidate Subgraph 2 (Not Matching)
# -------------------------------
# Candidate subgraph 2: nodes = ['A','B','C','D','E']
candidate2 = ['A', 'B', 'C', 'D', 'E']
candidate2_edges = list(G.subgraph(candidate2).edges())

# Check isomorphism against the valid graphlet.
subgraph2 = G.subgraph(candidate2)
is_iso2 = nx.is_isomorphic(subgraph2, graphlet)
# (For this candidate, because of the extra edge B–D, is_iso2 is expected to be False.)

plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color="skyblue",
        edge_color="gray", node_size=800)
# First, highlight candidate nodes (we start with pink, then change to red if not valid).
nx.draw_networkx_nodes(G, pos, nodelist=candidate2, node_color="red", node_size=800)
nx.draw_networkx_edges(G, pos, edgelist=candidate2_edges, edge_color="red", width=3)
plt.title("Candidate: A → B → C → D → E (Not Valid)")

# Add a small inset in the top right to display the valid graphlet.
ax = plt.gca()
inset_ax = plt.axes([0.65, 0.65, 0.3, 0.3])
nx.draw_networkx(graphlet, pos=pos_graphlet, ax=inset_ax,
                 with_labels=False, node_color="lightgray",
                 edge_color="black", node_size=300)
inset_ax.set_title("Graphlet", fontsize=10)
inset_ax.axis('off')

fname = os.path.join(output_dir, "frame_1.png")
plt.savefig(fname)
frames.append(fname)
plt.close()

# -------------------------------
# Frame 2: Process Candidate Subgraph 1 (Matching)
# -------------------------------
# Candidate subgraph 1: nodes = ['C','D','E','F','G']
candidate1 = ['C', 'D', 'E', 'F', 'G']
candidate1_edges = list(G.subgraph(candidate1).edges())

subgraph1 = G.subgraph(candidate1)
is_iso1 = nx.is_isomorphic(subgraph1, graphlet)
# Here, candidate 1 is expected to be isomorphic to our valid graphlet (a simple path).

plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color="skyblue",
        edge_color="gray", node_size=800)
# Mark candidate1 in green to indicate a match.
nx.draw_networkx_nodes(G, pos, nodelist=candidate1, node_color="green", node_size=800)
nx.draw_networkx_edges(G, pos, edgelist=candidate1_edges, edge_color="green", width=3)
plt.title("Candidate: C → D → E → F → G (Valid)")

# Add the same inset for the graphlet.
ax = plt.gca()
inset_ax = plt.axes([0.65, 0.65, 0.3, 0.3])
nx.draw_networkx(graphlet, pos=pos_graphlet, ax=inset_ax,
                 with_labels=False, node_color="lightgray",
                 edge_color="black", node_size=300)
inset_ax.set_title("Graphlet", fontsize=10)
inset_ax.axis('off')

# Add the valid sequence at the bottom of the image.
plt.figtext(0.5, 0.05, "C → D → E → F → G", ha="center", fontsize=12, color="green")

fname = os.path.join(output_dir, "frame_2.png")
plt.savefig(fname)
frames.append(fname)
plt.close()

# -------------------------------
# 4. Compile Frames into a GIF
# -------------------------------
images = [imageio.imread(frame) for frame in frames]
gif_filename = "valid_sequence_simulation_extended.gif"
imageio.mimsave(gif_filename, images, duration=2)

print(f"GIF saved as {gif_filename}")
