import matplotlib.pyplot as plt
import numpy as np

def plot_persistence_diagram(file_path):
    try:
        # load persistence pairs using numpy
        persistence_pairs = np.loadtxt(file_path)

        # separate birth and death values into separate arrays
        births = persistence_pairs[:, 0]
        deaths = persistence_pairs[:, 1]

        # determine maximum values
        max_value = max(np.max(births), np.max(deaths))

        plt.figure(figsize=(8, 8))
        plt.scatter(births, deaths, color='blue', alpha=0.7, s=10)  # Reduced point size

        # add diagonal (birth = death)
        plt.plot([0, max_value], [0, max_value], color='red', linestyle='--', label='Birth = Death')

        plt.title("Persistence Diagram")
        plt.xlabel("Birth")
        plt.ylabel("Death")
        plt.legend()
        plt.grid(True)
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except ValueError:
        print(f"Error: The file '{file_path}' has an incorrect format. Ensure it contains persistence pairs in the format '(birth, death)'.")
    except Exception as e:
        print(f"Unknown error: {e}")

plot_persistence_diagram("persistence_pairs.txt")