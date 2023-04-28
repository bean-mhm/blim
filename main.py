"""

3D LUT Generator for blim

Repo:
https://github.com/bean-mhm/blim

"""


import numpy as np
import colour
import os
import time

from blim import apply_transform, vt_version


# Parameters


compress_lg2_min = -12
compress_lg2_max = 5

parallel_processing = True

lut_dims = 65
lut_filename = 'blim'
lut_comments = [
    '-------------------------------------------------',
    '',
    f'blim v{vt_version} - Bean\'s Unprofessional View Transform',
    '',
    'LUT input is expected to be in Linear BT.709 I-D65 and gone through an AllocationTransform like the following:',
    f'!<AllocationTransform> {{allocation: lg2, vars: [{compress_lg2_min}, {compress_lg2_max}, {2**compress_lg2_min}]}}',
    '',
    'Output will be in sRGB 2.2.',
    '',
    'Repo:',
    'https://github.com/bean-mhm/blim',
    '',
    'Read more:',
    'https://opencolorio.readthedocs.io/en/latest/guides/authoring/authoring.html#how-to-configure-colorspace-allocation',
    '',
    '-------------------------------------------------'
]


# Print the parameters
print(f'{vt_version = }')
print(f'{compress_lg2_min = }')
print(f'{compress_lg2_max = }')
print(f'{parallel_processing = }')
print(f'{lut_dims = }')
print(f'{lut_filename = }')
print(f'{lut_comments = }')
print('')

t_start = time.time()

# Make a linear 3D LUT
print('Making a linear 3D LUT...')
lut = colour.LUT3D(
    table = colour.LUT3D.linear_table(lut_dims),
    name = 'blim',
    domain = np.array([[0, 0, 0], [1, 1, 1]]),
    size = lut_dims,
    comments = lut_comments
)

# Apply transform on the LUT
print('Applying the transform...')
lut.table = apply_transform(lut.table, compress_lg2_min, compress_lg2_max, parallel_processing)

# Write the LUT
print('Writing the LUT...')
script_dir = os.path.realpath(os.path.dirname(__file__))
colour.write_LUT(
    LUT = lut,
    path = f"{script_dir}/{lut_filename}.spi3d",
    decimals = 7,
    method = 'Sony SPI3D'
)

t_end = time.time()

print(f'Done ({t_end - t_start:.1f} s)')
