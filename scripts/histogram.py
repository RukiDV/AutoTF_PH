#!/usr/bin/env python3
import os
import nrrd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_volume_data(file_path):
    """
    Loads volume data from an NHDR file using pynrrd.
    The returned data is flattened for histogram computation.
    """
    try:
        data, header = nrrd.read(file_path)
        # Flatten the volume data in case it's multi-dimensional.
        return data.flatten()
    except Exception as e:
        print(f"Error loading volume data from {file_path}: {e}")
        exit(1)

def plot_histogram(volume_data, output_path="images/volume_histogram.png"):
    """
    Computes and plots the histogram of the volume's intensity values.
    Saves it as a PNG to the specified output_path under images/.
    """
    bins = 256
    hist, bin_edges = np.histogram(volume_data, bins=bins, range=(0, 255))
    
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], hist, width=1.0, edgecolor='black')
    plt.title("Volume Intensity Histogram")
    plt.xlabel("Intensity Value")
    plt.ylabel("Frequency")
    plt.xlim([0, 255])
    plt.tight_layout()

    # Ensure the images/ directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save the figure
    plt.savefig(output_path, dpi=300)
    print(f"Histogram saved to {output_path}")
    
    # Optionally display it interactively
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot and save histogram of volume data")
    parser.add_argument("file", help="Path to volume data file (e.g., input.nhdr)")
    parser.add_argument(
        "--out", "-o",
        default="images/volume_histogram.png",
        help="Output PNG path (default: images/volume_histogram.png)"
    )
    args = parser.parse_args()

    volume_data = load_volume_data(args.file)
    plot_histogram(volume_data, output_path=args.out)

if __name__ == '__main__':
    main()

