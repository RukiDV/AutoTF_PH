import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Compare Scalar vs. Gradient Persistence at a given threshold.")
    parser.add_argument("volume_label", type=str, help="Label to describe the volume (e.g., 'Silicon')")
    parser.add_argument( "-T", "--threshold", type=int, default=10, help="Persistence threshold (default: 10)")
    args = parser.parse_args()
    label = args.volume_label
    T = args.threshold

    scalar_df = pd.read_csv('scalar_pairs.csv')
    gradient_df = pd.read_csv('gradient_pairs.csv')

    # compute persistence
    scalar_df['persistence'] = scalar_df['death'] - scalar_df['birth']
    gradient_df['persistence'] = gradient_df['death'] - gradient_df['birth']

    # count features ≥ T
    scalar_count = (scalar_df['persistence'] >= T).sum()
    gradient_count = (gradient_df['persistence'] >= T).sum()

    modes = ['Scalar-Persistence', 'Gradient-Persistence']
    counts = [scalar_count, gradient_count]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.bar(modes, counts)
    ax.bar_label(bars, padding=4)
    ax.set_ylabel(f'Number of features (persistence ≥ {T})')
    ax.set_title(f'Scalar vs. Gradient Persistence (T={T}): {label}')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    os.makedirs("output_plots", exist_ok=True)
    save_path = f"output_plots/compare_persistence_{label}_T{T}.png"
    plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    
    plt.show()

    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    main()