"""Pseudo-spectral image processing. All bands are derived from RGB pixels."""
import numpy as np

WAVELENGTHS = np.linspace(450, 950, 12).astype(int)

def pseudo_cube(rgb):
    image = np.asarray(rgb, dtype=np.float32) / 255.0
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    luminance = .299*r + .587*g + .114*b
    bands = []
    for i, _ in enumerate(WAVELENGTHS):
        t = i / (len(WAVELENGTHS)-1)
        # Smooth mixtures and weak deterministic spatial modulation emulate band diversity.
        curve = (1-t)*b + t*r
        band = np.clip(.62*curve + .38*g + .08*np.sin((luminance+t)*np.pi), 0, 1)
        bands.append(band)
    return np.stack(bands, axis=-1)

def false_color(cube):
    idx = [9, 5, 1]
    out = cube[..., idx]
    lo, hi = np.percentile(out, [2, 98])
    return np.clip((out-lo)/max(hi-lo, 1e-5), 0, 1)
