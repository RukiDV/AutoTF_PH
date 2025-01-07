import numpy as np
import os

def create_donut_array(dim, tube_radius, ring_radius):
    
    # create a 3D coordinate grid
    x = np.linspace(-1, 1, dim)
    y = np.linspace(-1, 1, dim)
    z = np.linspace(-1, 1, dim)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # calculate the distance from the ring center
    ring_distance = np.sqrt(X**2 + Y**2) - ring_radius
    distance_from_donut = np.sqrt(ring_distance**2 + Z**2)

    # initialize 3D array
    donut = np.zeros((dim, dim, dim), dtype=np.uint8)

    # fill the donut array based on the radii
    inside_donut = distance_from_donut < tube_radius
    donut[inside_donut] = 255

    return donut

# parameters for the donut
dim = 256
tube_radius = 0.3
ring_radius = 0.5

donut_array = create_donut_array(dim, tube_radius, ring_radius)

# output file paths
output_dir = 'data/volume'
output_raw_filename = os.path.join(output_dir, 'donut.raw')
output_nhdr_filename = os.path.join(output_dir, 'donut.nhdr')

# ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# write the raw data to a file
donut_array.tofile(output_raw_filename)

# define header data
header = {
    'type': 'uint8',  # data type of the voxels
    'dimension': 3,  # 3D volume
    'sizes': donut_array.shape,  # dimensions of the array
    'encoding': 'raw',  # uncompressed raw data
    'data file': 'donut.raw',  # name of the raw data file
    'space directions': '(1,0,0) (0,1,0) (0,0,1)',  # equal spacing in all directions
}

# write the header file
with open(output_nhdr_filename, 'w') as nhdr_file:
    nhdr_file.write("NRRD0005\n")
    for key, value in header.items():
        if isinstance(value, (list, tuple)):
            value = ' '.join(map(str, value))
        nhdr_file.write(f"{key}: {value}\n")

print(f'Donut NRRD file has been saved as {output_nhdr_filename}')