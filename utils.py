"""

Helper functions for blim

Repo:
https://github.com/bean-mhm/blim

"""


import numpy as np
import colour


def lerp(a, b, t):
    return a + t * (b - a)


def safe_divide(a, b):
    if (b == 0.0):
        return 0.0
    return a / b


def safe_pow(a, b):
    return np.sign(a) * (np.abs(a)**b)


def pivot_pow(a, b, pivot):
    return pivot * ((a / pivot)**b)


def smootherstep(x, edge0, edge1):
    x = np.clip(safe_divide((x - edge0), (edge1 - edge0)), 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def map_range(inp, inp_start, inp_end, out_start, out_end):
    return out_start + ((out_end - out_start) / (inp_end - inp_start)) * (inp - inp_start)


def map_range_clamp(inp, inp_start, inp_end, out_start, out_end):
    v = out_start + ((out_end - out_start) / (inp_end - inp_start)) * (inp - inp_start)
    
    if v < out_start:
        return out_start
    
    if v > out_end:
        return out_end
    
    return v


def map_range_smootherstep(inp, inp_start, inp_end, out_start, out_end):
    if (inp_start == inp_end):
        return 0.0
    
    fac = \
        (1.0 - smootherstep(inp, inp_end, inp_start)) \
        if inp_start > inp_end \
        else smootherstep(inp, inp_start, inp_end)
    
    return out_start + fac * (out_end - out_start)


def blender_rgb_to_hsv(inp):
    h = 0.0
    s = 0.0
    v = 0.0

    cmax = max(inp[0], max(inp[1], inp[2]))
    cmin = min(inp[0], min(inp[1], inp[2]))
    cdelta = cmax - cmin

    v = cmax
    
    if cmax != 0.0:
        s = cdelta / cmax
    
    if s != 0.0:
        c = (-inp + cmax) / cdelta

        if inp[0] == cmax:
            h = c[2] - c[1]
        elif inp[1] == cmax:
            h = 2.0 + c[0] - c[2]
        else:
            h = 4.0 + c[1] - c[0]

        h /= 6.0

        if h < 0.0:
            h += 1.0

    return np.array([h, s, v])


def blender_hsv_to_rgb(inp):
    h = inp[0]
    s = inp[1]
    v = inp[2]

    if s == 0.0:
        return np.array([v, v, v])
    else:
        if h == 1.0:
            h = 0.0

        h *= 6.0
        i = np.floor(h)
        f = h - i
        p = v * (1.0 - s)
        q = v * (1.0 - (s * f))
        t = v * (1.0 - (s * (1.0 - f)))

        if i == 0.0:
            return np.array([v, t, p])
        elif i == 1.0:
            return np.array([q, v, p])
        elif i == 2.0:
            return np.array([p, v, t])
        elif i == 3.0:
            return np.array([p, q, v])
        elif i == 4.0:
            return np.array([t, p, v])
        else:
            return np.array([v, p, q])


def blender_hue_sat(inp, hue, sat, value):
    hsv = blender_rgb_to_hsv(inp)
    
    hsv[0] = np.modf(hsv[0] + hue + 0.5)[0]
    hsv[1] = np.clip(hsv[1] * sat, 0, 1)
    hsv[2] = hsv[2] * value
    
    return blender_hsv_to_rgb(hsv)


def BT_709_to_XYZ(inp):
    mat = np.array([
        [ 0.4123908 ,  0.35758434,  0.18048079],
        [ 0.21263901,  0.71516868,  0.07219232],
        [ 0.01933082,  0.11919478,  0.95053215]
    ])
    return np.matmul(mat, inp)


def XYZ_to_BT_709(inp):
    mat = np.array([
        [ 3.24096994, -1.53738318, -0.49861076],
        [-0.96924364,  1.8759675 ,  0.04155506],
        [ 0.05563008, -0.20397696,  1.05697151]
    ])
    return np.matmul(mat, inp)


# Oklab implementation
# See: https://bottosson.github.io/posts/oklab

mat_Oklab_M1 = np.array([
    [ 0.8189330101,  0.3618667424, -0.1288597137],
    [ 0.0329845436,  0.9293118715,  0.0361456387],
    [ 0.0482003018,  0.2643662691,  0.6338517070]
])
mat_Oklab_M1_inv = np.linalg.inv(mat_Oklab_M1)

mat_Oklab_M2 = np.array([
    [ 0.2104542553,  0.7936177850, -0.0040720468],
    [ 1.9779984951, -2.4285922050,  0.4505937099],
    [ 0.0259040371,  0.7827717662, -0.8086757660]
])
mat_Oklab_M2_inv = np.linalg.inv(mat_Oklab_M2)

def BT_709_to_Oklab(inp):
    inp = BT_709_to_XYZ(inp)

    inp = np.matmul(mat_Oklab_M1, inp)
    
    inp = safe_pow(inp, 1.0 / 3.0)
    
    inp = np.matmul(mat_Oklab_M2, inp)
    
    return inp

def Oklab_to_BT_709(inp):
    inp = np.matmul(mat_Oklab_M2_inv, inp)
    
    inp = safe_pow(inp, 3.0)
    
    inp = np.matmul(mat_Oklab_M1_inv, inp)
    
    inp = XYZ_to_BT_709(inp)
    
    return inp


luminance_BT_709_I_D65 = np.array([0.299, 0.587, 0.114])
def rgb_lum(inp):
    return np.dot(inp, luminance_BT_709_I_D65)


def rgb_avg(inp):
    return (inp[0] + inp[1] + inp[2]) / 3.0


def rgb_sum(inp):
    return inp[0] + inp[1] + inp[2]


def rgb_max(inp):
    return max(max(inp[0], inp[1]), inp[2])


def rgb_min(inp):
    return min(min(inp[0], inp[1]), inp[2])


def rgb_mag(inp):
    return np.linalg.norm(inp)


def rgb_mag_over_sqrt3(inp):
    return np.linalg.norm(inp) / 1.7320508075688772935274463415059


def rgb_hue(inp):
    x1 = (inp[1] - inp[2]) * np.sqrt(3)
    x2 = inp[0]*2 - inp[1] - inp[2]
    
    hue = np.rad2deg(np.arctan2(x1, x2))
    
    if hue < 0.0:
        hue += 360.0
    
    if hue > 360.0:
        hue -= 360.0
    
    return hue


def rgb_sat(inp):
    inp_norm = inp / rgb_max(inp)
    return np.clip(rgb_max(inp_norm) - rgb_min(inp_norm), 0, 1)


def rgb_monotone(inp, col, amount):
    inp_mag = rgb_mag(inp)
    
    inp_norm = inp / inp_mag
    col_norm = col / rgb_mag(col)
    
    dot = np.dot(inp_norm, col_norm)
    
    out = col_norm * (inp_mag * dot)
    
    return inp + amount * (out - inp)


def rgb_monotone_in_Oklab(inp, col, amount):
    # Convert to Oklab
    inp = BT_709_to_Oklab(inp)
    col = BT_709_to_Oklab(col)
    
    # Dot product
    dot = np.dot(inp / inp[0], col / col[0])
    dot = max(0.0, dot)
    dot = dot**3.0
    
    # Target color
    out = col * (inp[0] / col[0]) * dot
    
    # Mix
    out = lerp(inp, out, amount)
    
    # Convert from Oklab
    out = Oklab_to_BT_709(out)
    out = np.maximum(out, 0.0)
    
    return out


def rgb_hue_selection(inp, hue, max_distance):
    inp_hue = rgb_hue(inp)
    
    dist1 = np.abs(inp_hue - hue)
    dist2 = np.abs(inp_hue - 360 - hue)
    dist3 = np.abs(inp_hue + 360 - hue)
    
    min_dist = min(min(dist1, dist2), dist3)
    
    return 1 - np.clip(min_dist / max_distance, 0, 1)


def rgb_selective_hsv(
    inp,
    
    hue_red,
    hue_yellow,
    hue_green,
    hue_cyan,
    hue_blue,
    hue_magenta,
    
    sat_red,
    sat_yellow,
    sat_green,
    sat_cyan,
    sat_blue,
    sat_magenta,
    
    val_red,
    val_yellow,
    val_green,
    val_cyan,
    val_blue,
    val_magenta
):
    hue_max_distance = 60
    
    reds = inp * rgb_hue_selection(inp, 0, hue_max_distance)
    yellows = inp * rgb_hue_selection(inp, 60, hue_max_distance)
    greens = inp * rgb_hue_selection(inp, 120, hue_max_distance)
    cyans = inp * rgb_hue_selection(inp, 180, hue_max_distance)
    blues = inp * rgb_hue_selection(inp, 240, hue_max_distance)
    magentas = inp * rgb_hue_selection(inp, 300, hue_max_distance)
    
    reds = blender_hue_sat(reds, hue_red, sat_red, val_red)
    yellows = blender_hue_sat(yellows, hue_yellow, sat_yellow, val_yellow)
    greens = blender_hue_sat(greens, hue_green, sat_green, val_green)
    cyans = blender_hue_sat(cyans, hue_cyan, sat_cyan, val_cyan)
    blues = blender_hue_sat(blues, hue_blue, sat_blue, val_blue)
    magentas = blender_hue_sat(magentas, hue_magenta, sat_magenta, val_magenta)
    
    return reds + yellows + greens + cyans + blues + magentas


# Shift a certain tone as it gets logarithmically brighter
def rgb_hue_shift(inp, channel, threshold, hue, sat, val):
    # Only take more saturated colors into account
    sat_fac = rgb_sat(inp)**2
    sat_fac = map_range_clamp(sat_fac, 0.6, 1.0, 0.0, 1.0)
    
    # See how much the input matches the channel
    mask = np.dot(channel, np.log2(inp + 1.0))
    
    # Saturation factor
    mask = max(0.0, mask * sat_fac)
    
    # Threshold
    mask = max(0.0, mask - threshold)
    
    # Define HSV adjustments
    hue = mask * hue + 0.5
    sat = mask * sat + 1.0
    val = mask * val + 1.0
    
    # HSV
    return blender_hue_sat(inp, hue, sat, val)


def fractional_hue_transform(inp, amount):
    out = safe_pow(inp*2.0 - 1.0, amount*0.3 + 1.0)
    return np.clip((out + 1.0) / 2.0, 0, 1)


magenta_mul_norm = np.array([1, -1, 1]) / rgb_mag(np.array([1, -1, 1]))
blue = np.array([0, 0, 1])

def rgb_perceptual_hue_shifts(inp):
    sat_mask = np.clip(rgb_sat(inp)**5.0 - 0.1, 0, 1)
    
    magenta_mask = max(0.0, np.dot(inp / rgb_mag(inp), magenta_mul_norm))
    magenta_mask = magenta_mask ** 3.0
    magenta_mask = map_range_smootherstep(magenta_mask, 0.0, 0.5, 1.0, 0.0)
    
    amount = rgb_mag(rgb_monotone(inp, blue, 0.8))
    amount = np.log2(amount + 1.0)
    
    amount = pivot_pow(amount, 1.0, 1.73205)
    amount *= 3.0
    
    amount *= sat_mask
    amount *= magenta_mask
    
    hsv = blender_rgb_to_hsv(inp)
    
    hue_fract, hue_int = np.modf(hsv[0] * 3.0)
    
    new_hue = (hue_int + fractional_hue_transform(hue_fract, amount)) / 3.0
    
    return blender_hsv_to_rgb(np.array([new_hue, hsv[1], hsv[2]]))


def rgb_path_to_white_mask(mono, min_exp, max_exp, mask_pow):
    mask = np.log2(mono)
    
    mask = map_range_clamp(mask, min_exp, max_exp, 0.0, 1.0)
    
    mask = mask**mask_pow
    
    return mask


white_Oklab = BT_709_to_Oklab(np.array([1,1,1]))

# Compress the highlights so that the output values fit into
# the [0, 1] range.
def rgb_compress_highlights(inp):
    inp_sum = rgb_sum(inp)
    inp_avg = rgb_avg(inp)
    inp_max = rgb_max(inp)
    
    # Path-to-white factors
    white_mix_1 = rgb_path_to_white_mask(inp_avg, min_exp = -3.0, max_exp = 7.0, mask_pow = 1.732)
    white_mix_2 = rgb_path_to_white_mask(inp_avg, min_exp = -1.0, max_exp = 7.0, mask_pow = 4.0)
    white_mix_3 = rgb_path_to_white_mask(inp_max, min_exp =  0.0, max_exp = 6.9, mask_pow = 2.0)
    
    # Reinhard
    inp = inp / (inp_sum + 1.0)
    
    # The Reinhard transform scales the input uniformly. This might
    # turn down bright colors extensively and make them look
    # saturated, leading to uncanny results. That's why we mix with
    # white based on the exposure. The mixing is done in Oklab to
    # try to preserve the hue in a perceptual way, although it's
    # just an approximation.
    
    # Convert to Oklab
    inp = BT_709_to_Oklab(inp)
    
    # Mix with white
    inp = lerp(inp, white_Oklab, white_mix_1)
    inp = lerp(inp, white_Oklab, white_mix_2)
    inp = lerp(inp, white_Oklab, white_mix_3)
    
    # Convert from Oklab
    inp = Oklab_to_BT_709(inp)
    
    # Clip
    inp = np.clip(inp, 0, 1)
    
    return inp


cmfs = (
    colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    .copy()
    .align(colour.SpectralShape(360, 780, 20))
)
illuminant_d65_sd = colour.SDS_ILLUMINANTS["D65"].copy().align(cmfs.shape)
illuminant_d65_sd.normalise()
illuminant_d65_xy = colour.temperature.CCT_to_xy_CIE_D(6500.0 * 1.438776877 / 1.4380)

def rgb_compress_highlights_spectral(inp, mono):
    inp_reinhard = inp * (1.0 / (mono + 1.0))
    
    white_mix = np.log2(mono + 1.0)
    white_mix = map_range_smootherstep(white_mix, -0.7, 9.0, 0.0, 1.0)
    white_mix = white_mix**2.0
    
    xyz = colour.RGB_to_XYZ(
        RGB = inp_reinhard,
        illuminant_RGB = illuminant_d65_xy,
        illuminant_XYZ = illuminant_d65_xy,
        matrix_RGB_to_XYZ = colour.models.RGB_COLOURSPACE_BT709.matrix_RGB_to_XYZ
    )
    
    sd = colour.XYZ_to_sd(xyz, method="Jakob 2019", cmfs=cmfs, illuminant=illuminant_d65_sd)
    
    sd.values = sd.values + white_mix * (illuminant_d65_sd.values - sd.values)
    
    xyz = colour.sd_to_XYZ(sd, cmfs=cmfs, illuminant=illuminant_d65_sd, method="Integration")
    
    return colour.XYZ_to_RGB(
        XYZ = xyz,
        illuminant_XYZ = illuminant_d65_xy,
        illuminant_RGB = illuminant_d65_xy,
        matrix_XYZ_to_RGB = colour.models.RGB_COLOURSPACE_BT709.matrix_XYZ_to_RGB
    ) / 100.0


# Adjust the highlights and the shadows in a smooth way.
# Note: All arguments must be scalars.
def enhance_curve(inp, shadow_pow, highlight_pow, mix_pow):
    a = inp**shadow_pow
    b = 1.0 - (1.0 - inp)**highlight_pow
    mix = inp**mix_pow
    
    return lerp(a, b, mix)
