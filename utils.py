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


def rgb_lum(inp):
    return inp[0] * 0.299 + inp[1] * 0.587 + inp[2] * 0.114


def rgb_avg(inp):
    return (inp[0] + inp[1] + inp[2]) / 3.0


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


def rgb_hue_shift(inp, channel, threshold, hue, sat, val):
    sat_fac = rgb_sat(inp)**2
    sat_fac = map_range_clamp(sat_fac, 0.6, 1.0, 0.0, 1.0)
    
    mask = np.dot(channel, np.log2(inp + 1.0))
    mask = max(0.0, mask * sat_fac)
    mask = max(0.0, mask - threshold)
    
    hue = mask * hue + 0.5
    sat = mask * sat + 1.0
    val = mask * val + 1.0
    
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


def rgb_path_to_white_mask(inp_mag, min_exp, max_exp, mask_pow):
    mask = np.log2(rgb_mag(inp_mag))
    mask = map_range_clamp(mask, min_exp, max_exp, 0.0, 1.0)
    mask = mask**mask_pow
    
    return mask


white = np.array([1,1,1])
def rgb_compress_highlights(inp):
    inp_mag = rgb_mag(inp)
    
    # Reinhard
    inp = inp * (1.0 / (inp_mag + 1.0))
    
    white_mix = rgb_path_to_white_mask(inp_mag, -1.0, 8.0, 2.4)
    inp = lerp(inp, white, white_mix)
    
    white_mix = rgb_path_to_white_mask(inp_mag, -1.0, 10.0, 4.0)
    inp = lerp(inp, white, white_mix)
    
    white_mix = rgb_path_to_white_mask(inp_mag, 4.0, 7.2, 3.0)
    inp = lerp(inp, white, white_mix)
    
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


def enhance_curve(inp, shadow_pow, highlight_pow, mix_pow):
    a = inp**shadow_pow
    b = 1.0 - (1.0 - inp)**highlight_pow
    mix = inp**mix_pow
    
    return a + mix * (b - a)
