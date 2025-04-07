import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def plot_persistence_diagram(file_path, output_file, output_dir="output"):
    try:
        persistence_pairs = np.loadtxt(file_path)
        births = persistence_pairs[:, 0]
        deaths = persistence_pairs[:, 1]
        persistence = deaths - births
        max_value = max(np.max(births), np.max(deaths))

        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(births, deaths, c=persistence, cmap='viridis', alpha=0.7, s=10)
        plt.plot([0, max_value], [0, max_value], color='red', linestyle='--', label='Birth = Death')

        plt.title("Persistence Diagram", fontsize=14)
        plt.xlabel("Birth", fontsize=12)
        plt.ylabel("Death", fontsize=12)
        plt.legend()
        plt.grid(True)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Persistence (Death - Birth)')

        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {output_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python persistence_diagram.py <persistence_pairs.txt> <output_file> [output_dir]")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "output_plots"
    plot_persistence_diagram(input_file, output_file, output_dir)
