import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def plot_persistence_diagram(file_path, output_dir="output", model_name=None):
    try:
        # use the provided model name or extract it from the file path if not provided
        if model_name is None:
            model_name = os.path.splitext(os.path.basename(file_path))[0]
        else:
            model_name = model_name.strip()

        # load persistence pairs from file
        persistence_pairs = np.loadtxt(file_path)

        # separate birth and death values
        births = persistence_pairs[:, 0]
        deaths = persistence_pairs[:, 1]

        # calculate persistence (lifetime) for each feature
        persistence = deaths - births

        # determine the maximum value for setting the diagonal
        max_value = max(np.max(births), np.max(deaths))

        plt.figure(figsize=(8, 8))
        
        # use a colormap to map persistence values to colors
        scatter = plt.scatter(births, deaths, c=persistence, cmap='viridis', alpha=0.7, s=10)

        # add diagonal (birth = death)
        plt.plot([0, max_value], [0, max_value], color='red', linestyle='--', label='Birth = Death')

        plt.title(f"Persistence Diagram - {model_name}", fontsize=14)
        plt.xlabel("Birth", fontsize=12)
        plt.ylabel("Death", fontsize=12)

        plt.legend()
        plt.grid(True)

        # add a color bar to indicate persistence scale
        cbar = plt.colorbar(scatter)
        cbar.set_label('Persistence (Death - Birth)')

        # ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"persistence_diagram_{model_name}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

        plt.show()

        print(f"Plot saved to: {output_file}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"Unknown error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python persistance_diagagram.py <input_file> [model_name] [output_directory]")
        sys.exit(1)

    input_file = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else None
    output_directory = sys.argv[3] if len(sys.argv) > 3 else "output_plots"
    # Call the plotting function
    plot_persistence_diagram(input_file, output_dir=output_directory, model_name=model_name)
