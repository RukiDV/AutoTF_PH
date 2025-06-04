import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def main():

    parser = argparse.ArgumentParser(description="Plot Persistence Curves for a given volume.")
    parser.add_argument("volume_label", type=str, help="Label to describe the volume (e.g., 'Silicon')")
    args = parser.parse_args()
    label = args.volume_label

    scalar_df = pd.read_csv('scalar_pairs.csv')
    gradient_df = pd.read_csv('gradient_pairs.csv')

    # compute persistence = death - birth
    scalar_df['p'] = scalar_df['death'] - scalar_df['birth']
    gradient_df['p'] = gradient_df['death'] - gradient_df['birth']

    # build the counts for thresholds 0 - 100
    thresholds = list(range(0, 101, 10))
    scalar_counts = [(scalar_df['p'] >= T).sum() for T in thresholds]
    gradient_counts = [(gradient_df['p'] >= T).sum() for T in thresholds]

    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, scalar_counts, marker='o', label='Scalar-PH')
    plt.plot(thresholds, gradient_counts, marker='o', label='Gradient-PH')
    plt.xlabel('Persistence threshold T')
    plt.ylabel('Number of features ≥ T')
    plt.title(f'Persistence curves: {label}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # save the figure to output_plots
    os.makedirs("output_plots", exist_ok=True)
    save_path = f"output_plots/persistence_curve_{label}.png"
    plt.savefig(save_path, dpi=600)

    plt.show()

    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    main()
