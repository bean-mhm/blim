# blim - Bean's Unprofessional View Transform

## Introduction

blim is an experimental view transform I've been working on for a few months, intended to produce relatively saturated, bright, and contrasty colors out of the box, while avoiding clipping and unwanted hue skews. blim is by no means production-ready, nor is it based on physically accurate phenomena (no film emulation for example).

## Eye Candy

See comparisons between blim and other view transforms in the [releases](https://github.com/bean-mhm/blim/releases) section.

## Scripts

The code is structured in the following way:

| Script | Role | Uses |
|---|---|---|
| main.py | Generates a 3D LUT for blim | blim.py  |
| blim.py | Transforms a given linear 3D LUT table | utils.py |
| utils.py | Contains helper functions | - |

Here are the external libraries required to run the scripts:

 - [Colour](https://www.colour-science.org/)
 
 - [NumPy](https://numpy.org/)
 
 - [Joblib](https://joblib.readthedocs.io/en/latest)

## Using the LUT

First, a few notes:

 - blim's 3D LUT is designed to be used in an [OpenColorIO](https://opencolorio.org/) environment.
 
 - blim only supports the sRGB display format as of now.

If `main.py` runs successfully, you should see a file named `blim.spi3d` in the same directory. Alternatively, you can look up the latest LUT - no pun intended - in the [releases](https://github.com/bean-mhm/blim/releases) section, which may be outdated.

The LUT's expected input and output formats are mentioned in the LUT comments at end of the file, but they can also be seen in the code.

Here's an example of the LUT comments (note that this might not match the latest version):

```
# -------------------------------------------------
# 
# blim v0.3.2 - Bean's Unprofessional View Transform
# 
# LUT input is expected to be in Linear BT.709 I-D65 and gone through an AllocationTransform like the following:
# !<AllocationTransform> {allocation: lg2, vars: [-12, 5, 0.000244140625]}
# 
# Output will be in sRGB 2.2.
# 
# Repo:
# https://github.com/bean-mhm/blim
# 
# Read more:
# https://opencolorio.readthedocs.io/en/latest/guides/authoring/authoring.html#how-to-configure-colorspace-allocation
# 
# -------------------------------------------------
```

Here's an example of how you can add blim to an OCIO config:

```yaml
colorspaces:
  - !<ColorSpace>
    name: blim
    family: Image Formation
    equalitygroup: ""
    bitdepth: unknown
    description: blim - Bean's Unprofessional View Transform
    isdata: false
    allocation: uniform
    from_scene_reference: !<GroupTransform>
      children:
        - !<ColorSpaceTransform> {src: Linear CIE-XYZ I-E, dst: Linear BT.709 I-D65}
        - !<AllocationTransform> {allocation: lg2, vars: [-12, 5, 0.000244140625]}
        - !<FileTransform> {src: blim.spi3d, interpolation: linear}
  ...
```

Paying attention to the transforms, you will notice a `ColorSpaceTransform` from CIE-XYZ I-E to Linear BT.709 I-D65. This is because the example OCIO config has its reference color space (the `reference` role) set to CIE-XYZ I-E. If your config already uses Linear BT.709 I-D65 as its reference this is not needed. If your config uses another color space as its reference, you should manually do a conversion to Linear BT.709 I-D65. You can get the conversion matrices using the [Colour](https://www.colour-science.org/) library.

Next, we have an `AllocationTransform`, which can be directly copied from the LUT comments. The `AllocationTransform` here literally just takes the log2 of the tristimulus (RGB) values and maps them from a specified range (the first two values after `vars`) to the [0, 1] range. The third value in `vars` is the offset applied to the values before mapping. This is done to keep the blacks.

Finally, a `FileTransform` references the 3D LUT.

Here's an example of how you can add blim as a view transform in an OCIO config:

```yaml
displays:
  sRGB:
    - !<View> {name: blim, colorspace: blim}
    ...
```

> `...` refers to the other view transforms in the config. `...` is generally used as a placeholder for the other parts of the code. I cannot believe I had to mention this, but someone was actually confused by it.
