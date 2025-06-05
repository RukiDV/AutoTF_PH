import matplotlib.pyplot as plt
import numpy as np
import os
import sys

SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.pdf', '.svg', '.eps', '.tiff', '.tif', '.webp'}

def plot_persistence_diagram(file_path, output_file_name, output_dir="output_plots", figsize=(10,10), dpi=600, vector=False):
    # check that the input file exists
    if not os.path.isfile(file_path):
        print(f"Error: input file not found: {file_path}")
        sys.exit(1)

    root, ext = os.path.splitext(output_file_name)
    ext = ext.lower()
    if ext not in SUPPORTED_EXTS:
        print(f"Error: unsupported output format '{ext}'.")
        print(f"Please use one of: {', '.join(sorted(SUPPORTED_EXTS))}")
        sys.exit(1)

    try:
        persistence_pairs = np.loadtxt(file_path)
    except Exception as e:
        # For example: malformed file, wrong columns, etc.
        print(f"Error: could not parse '{file_path}': {e}")
        sys.exit(1)

    if persistence_pairs.ndim != 2 or persistence_pairs.shape[1] < 2:
        print(f"Error: '{file_path}' does not appear to contain two columns of numbers.")
        sys.exit(1)

    births = persistence_pairs[:, 0]
    deaths = persistence_pairs[:, 1]
    persistence = deaths - births
    max_value = max(np.max(births), np.max(deaths))

    # make sure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error: could not create output directory '{output_dir}': {e}")
        sys.exit(1)
    output_path = os.path.join(output_dir, output_file_name)

    if vector:
        plt.figure(figsize=figsize)
    else:
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
        save_kwargs['format'] = ext.lstrip('.')
    else:
        save_kwargs['dpi'] = dpi

    try:
        plt.savefig(output_path, **save_kwargs, bbox_inches='tight', pad_inches=0.02)
        plt.close()
    except Exception as e:
        print(f"Error: could not write figure to '{output_path}': {e}")
        sys.exit(1)

    print(f"Plot saved to: {output_path}  (figsize={figsize}, dpi={dpi}, vector={vector})")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python persistence_diagram.py <input_pairs.txt> <output_image.png> [output_dir] [vector]")
        sys.exit(1)

    input_file      = sys.argv[1]
    output_filename = sys.argv[2]
    output_dir      = sys.argv[3] if len(sys.argv) > 3 else "output_plots"
    vector_flag     = (len(sys.argv) > 4 and sys.argv[4].lower() in ("svg","pdf","eps","vector","v"))

    plot_persistence_diagram(input_file, output_filename, output_dir=output_dir, figsize=(12,12), dpi=600, vector=vector_flag)
