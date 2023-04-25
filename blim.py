"""

blim - Bean's Unprofessional View Transform

Input Color Space:   Linear BT.709 I-D65
Output Color Space:  sRGB 2.2

Repo:
https://github.com/bean-mhm/blim

"""


import numpy as np
import colour
import joblib

from utils import *


vt_name = 'blim'
vt_version = '0.3.0'


# Transform a 3D LUT
def apply_transform(table: np.ndarray, compress_lg2_min, compress_lg2_max, parallel):
    if len(table.shape) != 4:
        raise Exception('table must have 4 dimensions (3 for xyz, 1 for the color channels)')
    
    if table.shape[3] != 3:
        raise Exception('the fourth axis must have a size of 3 (RGB)')
    
    # Decompress: Map Range
    table = colour.algebra.linear_conversion(
        table,
        np.array([0.0, 1.0]),
        np.array([compress_lg2_min, compress_lg2_max])
    )
    
    # Decompress: Exponent
    colour.algebra.set_spow_enable(True)
    table = colour.algebra.spow(2.0, table)
    
    # Decompress: Black Point
    offset = (2.0**compress_lg2_min)
    table -= offset
    
    # Eliminate negative values
    table = np.maximum(table, 0.0)
    
    # Pre-Exposure
    pre_exposure = 1.0
    table *= 2**pre_exposure
    
    # Apply transform on each RGB triplet
    if parallel:
        print('Starting parallel element-wise transform...')
        num_points = table.shape[0] * table.shape[1] * table.shape[2]
        stride_y = table.shape[0]
        stride_z = table.shape[0] * table.shape[1]
        results = joblib.Parallel(n_jobs=8)(joblib.delayed(run_parallel)(table, (i % stride_y, (i % stride_z) // stride_y, i // stride_z)) for i in range(num_points))
    
        # Arrange the results
        print('Arranging the results...')
        for z in range(table.shape[2]):
            for y in range(table.shape[1]):
                for x in range(table.shape[0]):
                    index = x + (y * stride_y) + (z * stride_z)
                    table[x, y, z] = results[index]
    else:
        print('Starting serial element-wise transform...')
        for z in range(table.shape[2]):
            for y in range(table.shape[1]):
                print(f'at [0, {y}, {z}]')
                for x in range(table.shape[0]):
                    table[x, y, z] = transform_rgb(table[x, y, z])
    
    # OETF (Gamma 2.2)
    table = colour.algebra.spow(table, 1.0 / 2.2)
    
    return table


def run_parallel(table, indices):
    result = transform_rgb(table[indices])
    print(f'{indices} done')
    return result


# Transform a single RGB triplet
# This function is used by apply_transform.
def transform_rgb(inp):
    # Skip pure black
    if not np.any(inp):
        return inp
    
    # Power
    mono = rgb_mag_over_sqrt3(inp)
    inp = inp * (mono**1.3) / mono
    
    # Color Filter
    inp = rgb_monotone_in_Oklab(inp, np.array([1.0, 0.15, 0.01]), 0.015)
    
    # Selective HSV
    inp = rgb_selective_hsv(
        inp = inp,

        hue_red = 0.501,
        hue_yellow = 0.5,
        hue_green = 0.496,
        hue_cyan = 0.5,
        hue_blue = 0.499,
        hue_magenta = 0.502,

        sat_red = 1.0,
        sat_yellow = 1.04,
        sat_green = 1.05,
        sat_cyan = 1.02,
        sat_blue = 1.0,
        sat_magenta = 1.05,

        val_red = 1.0,
        val_yellow = 1.0,
        val_green = 1.02,
        val_cyan = 1.0,
        val_blue = 1.0,
        val_magenta = 1.0
    )
    
    # Hue Shifts
    inp = rgb_hue_shift(inp, np.array([1.0, -1.1, -1.3]), 0.05, 0.001, 0.0, 0.0)
    inp = rgb_hue_shift(inp, np.array([-1.0, 1.0, -1.0]), 0.05, -0.005, 0.0, 0.0)
    inp = rgb_hue_shift(inp, np.array([-1.0, -1.0, 1.0]), 0.01, -0.003, 0.0, -0.04)
    
    # Compress Highlights
    inp = rgb_compress_highlights(inp)
    
    # Brighten and Clamp
    inp = np.clip(inp * 1.01, 0, 1)
    inp = blender_hue_sat(inp, 0.5, 1.01, 1.0)
    
    # Enhance Curve
    mono = rgb_mag_over_sqrt3(inp)
    inp = inp * enhance_curve(mono, 1.01, 1.5, 0.6) / mono
    
    # Clip and return
    return np.clip(inp, 0, 1)
