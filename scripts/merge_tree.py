import os
import matplotlib.pyplot as plt
import networkx as nx

def load_merge_tree_edges(file_path):
    merge_tree = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            parent, child = int(parts[0]), int(parts[1])
            merge_tree.setdefault(parent, []).append(child)
    return merge_tree

def plot_merge_tree_with_dot(
    file_path,
    output_dir="images",
    output_file="merge_tree.png"
):
    # ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, output_file)

    # build the directed graph from edge list
    G = nx.DiGraph()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            parent, child = int(parts[0]), int(parts[1])
            G.add_edge(parent, child)

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except Exception as e:
        print("Warning: graphviz layout failed, using spring_layout:", e)
        pos = nx.spring_layout(G)

    # Draw
    plt.figure(figsize=(12, 12))
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=50,
        font_size=8,
        arrows=True
    )
    plt.title("Filtered Merge Tree Visualization")
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Merge tree visualization saved to {full_path}")

if __name__ == "__main__":
    file_path = "merge_tree_edges_filtered.txt"
    plot_merge_tree_with_dot(file_path)
