import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def plot_persistence_diagram(file_path, output_file_name, output_dir="output_plots", figsize=(10,10), dpi=600, vector=False):
    # load data
    persistence_pairs = np.loadtxt(file_path)
    births = persistence_pairs[:, 0]
    deaths = persistence_pairs[:, 1]
    persistence = deaths - births
    max_value = max(np.max(births), np.max(deaths))

    # make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file_name)

    # create figure with high‐res settings
    if vector:
        # vector formats ignore DPI but will respect figsize
        fmt = os.path.splitext(output_file_name)[1].lower().lstrip('.')
        plt.figure(figsize=figsize)
    else:
        # for raster formats, set figsize and dpi
        plt.figure(figsize=figsize, dpi=dpi)

    scatter = plt.scatter(births, deaths, c=persistence, cmap='viridis', alpha=0.8, s=20, edgecolors='none')
    plt.plot([0, max_value], [0, max_value], color='red', linestyle='--', lw=1, label='Birth = Death')

    plt.title("Persistence Diagram", fontsize=16)
    plt.xlabel("Birth", fontsize=14)
    plt.ylabel("Death", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Persistence (Death - Birth)', fontsize=12)

    plt.tight_layout()

    save_kwargs = {}
    if vector:
        save_kwargs['format'] = fmt
    else:
        save_kwargs['dpi'] = dpi

    plt.savefig(output_path, **save_kwargs, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"Plot saved to: {output_path}  (figsize={figsize}, dpi={dpi}, vector={vector})")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python persistence_diagram.py <pairs.txt> <output.png> [output_dir] [vector]")
        sys.exit(1)

    input_file      = sys.argv[1]
    output_filename = sys.argv[2]
    output_dir      = sys.argv[3] if len(sys.argv) > 3 else "output_plots"
    vector_flag     = (len(sys.argv) > 4 and sys.argv[4].lower() in ("svg","pdf","eps","vector","v"))

    plot_persistence_diagram(input_file,
                             output_filename,
                             output_dir=output_dir,
                             figsize=(12,12),
                             dpi=600,
                             vector=vector_flag)
