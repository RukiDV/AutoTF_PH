import numpy as np
import os
import matplotlib.pyplot as plt

def create_donut_array(dim, tube_radius, ring_radius):
    """
    Create a 3D donut (torus) scalar field array with improved noise and gradient distribution.
    """
    # Create a 3D coordinate grid
    x = np.linspace(-1, 1, dim)
    y = np.linspace(-1, 1, dim)
    z = np.linspace(-1, 1, dim)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Calculate the distance from the ring center
    ring_distance = np.sqrt(X**2 + Y**2) - ring_radius
    distance_from_donut = np.sqrt(ring_distance**2 + Z**2)

    # Initialize 3D array with NaNs (unsichtbarer Hintergrund)
    donut = np.full((dim, dim, dim), np.nan, dtype=np.float32)

    # Fill the donut array based on the radii
    inside_donut = distance_from_donut < tube_radius
    donut[inside_donut] = 1 - (distance_from_donut[inside_donut] / tube_radius)  # Normalize to [0, 1]

    # Improved noise: Combination of uniform & normal distribution
    donut[inside_donut] += (
        np.random.uniform(-20, 20, size=donut[inside_donut].shape) +
        np.random.normal(scale=10, size=donut[inside_donut].shape)
    )

    return donut, inside_donut

def add_gradient_all_axes(scalar_field, mask):
    """
    Add a gradient to the scalar field along all axes (X, Y, Z), but only inside the donut.
    """
    dim = scalar_field.shape[0]

    # Create gradients along X, Y, and Z axes
    x_gradient = np.linspace(0, 100, dim).reshape(-1, 1, 1)  
    y_gradient = np.linspace(0, 100, dim).reshape(1, -1, 1)  
    z_gradient = np.linspace(0, 100, dim).reshape(1, 1, -1)  

    # Combine gradients (average)
    combined_gradient = (x_gradient + y_gradient + z_gradient) / 3  

    # Apply gradient only to the donut voxels
    scalar_field[mask] += combined_gradient[mask]

    return scalar_field

def normalize_to_uint8(scalar_field, mask):
    """
    Normalize the scalar field to the range [0, 255] and apply gamma correction.
    """
    # Hintergrund (NaN) entfernen, nur Donut normalisieren
    valid_values = scalar_field[mask]

    # Normalize values between 0 and 1
    min_val, max_val = valid_values.min(), valid_values.max()
    scalar_field[mask] = (valid_values - min_val) / (max_val - min_val)

    # Apply gamma correction (softens intensity jumps)
    scalar_field[mask] = np.clip(scalar_field[mask] ** 0.8, 0, 1) * 255

    # Hintergrund bleibt 0
    scalar_field[~mask] = 0  

    return scalar_field.astype(np.uint8)

def visualize_scalar_distribution(scalar_field, mask, output_dir="images"):
    """
    Visualize the distribution of scalar values using a histogram.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Nur Donut-Werte visualisieren
    donut_values = scalar_field[mask]

    # Plot the histogram of scalar values
    plt.hist(donut_values.flatten(), bins=100, color='blue', alpha=0.7)
    plt.title("Scalar Field Distribution (Donut only)")
    plt.xlabel("Scalar Value")
    plt.ylabel("Frequency")
    output_path = os.path.join(output_dir, "scalar_distribution.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Visualization saved to {output_path}")

def visualize_cross_section(scalar_field, mask, output_dir="images"):
    """
    Visualize a middle slice (cross-section) of the scalar field.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Mittel-Slice nehmen & als float32 casten, damit NaN möglich ist
    mid_slice = scalar_field[scalar_field.shape[0] // 2, :, :].astype(np.float32)
    mid_mask = mask[mask.shape[0] // 2, :, :]

    # Hintergrund transparent machen
    mid_slice[~mid_mask] = np.nan  

    plt.imshow(mid_slice, cmap='viridis')
    plt.colorbar(label="Scalar Value")
    plt.title("Middle Slice of Donut (No Background)")
    output_path = os.path.join(output_dir, "donut_cross_section.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Donut cross-section visualization saved to {output_path}")


# Parameters for the donut
dim = 256
tube_radius = 0.3
ring_radius = 0.5

# Create the donut scalar field and mask
donut_array, donut_mask = create_donut_array(dim, tube_radius, ring_radius)

# Add gradient to the donut array along all axes, but only inside the donut
donut_array = add_gradient_all_axes(donut_array, donut_mask)

# Normalize the scalar field to [0, 255]
donut_array = normalize_to_uint8(donut_array, donut_mask)

donut_array = donut_array.astype('<u1')
print("First 50 values after conversion to uint8:")
print(donut_array[donut_mask][:50])

print("Daten-Typ von donut_array:", donut_array.dtype)

# Debugging: Check scalar values
print(f"Donut values after normalization: Min={donut_array.min()}, Max={donut_array.max()}")
print("First 50 non-zero values in the volume data:")
print(donut_array[donut_mask][:50])

# Visualize the scalar distribution
visualize_scalar_distribution(donut_array, donut_mask)

# Visualize a cross-section of the donut
visualize_cross_section(donut_array, donut_mask)

raw_file_path = 'data/volume/donut.raw'

# Daten aus der .raw-Datei laden
data = np.fromfile(raw_file_path, dtype=np.uint8)

# Erste 50 Werte ausgeben
print("Print data from .raw file:")
print(data[:50])

# Output file paths
output_dir = 'data/volume'
output_raw_filename = os.path.join(output_dir, 'donut.raw')
output_nhdr_filename = os.path.join(output_dir, 'donut.nhdr')

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Write the raw data to a file
donut_array.tofile(output_raw_filename)
print(f"File size of {output_raw_filename}: {os.path.getsize(output_raw_filename)} bytes")

# Define header data for NRRD file
header = {
    'type': 'uint8',
    'dimension': 3,
    'sizes': donut_array.shape,  
    'encoding': 'raw',
    'data file': 'donut.raw',
    'space directions': '(1,0,0) (0,1,0) (0,0,1)',
}

# Write the header file
with open(output_nhdr_filename, 'w') as nhdr_file:
    nhdr_file.write("NRRD0005\n")
    for key, value in header.items():
        if isinstance(value, (list, tuple)):
            value = ' '.join(map(str, value))
        nhdr_file.write(f"{key}: {value}\n")

print(f"Donut NRRD file has been saved as {output_nhdr_filename}")



