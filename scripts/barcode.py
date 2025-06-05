import matplotlib.pyplot as plt
import numpy as np

def plot_persistence_barcodes(file_path, max_bars=10000):
    # load the data
    persistence_pairs = np.loadtxt(file_path)

    # limit the number of bars
    if len(persistence_pairs) > max_bars:
        persistence_pairs = persistence_pairs[:max_bars]

    # create barcodes
    plt.figure(figsize=(12, 6))
    for i, (birth, death) in enumerate(persistence_pairs):
        plt.plot([birth, death], [i, i], color='blue', linewidth=0.5)

    plt.title("Persistence Barcodes")
    plt.xlabel("Filtration Value")
    plt.ylabel("Feature Index")
    plt.grid(True)
    plt.show()

plot_persistence_barcodes("volume_data/persistence_pairs.txt")
