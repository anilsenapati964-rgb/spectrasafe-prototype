import numpy as np
from .spectral_simulation import WAVELENGTHS

def extract_features(cube, mask=None):
    if mask is None or mask.sum() < 20: mask = np.ones(cube.shape[:2], dtype=bool)
    px = cube[mask]
    mean, std = px.mean(0), px.std(0)
    ratios = mean[1:] / (mean[:-1] + 1e-5)
    normdiff = (mean[1:] - mean[:-1]) / (mean[1:] + mean[:-1] + 1e-5)
    rgb = cube[..., [9,5,1]]
    gray = rgb.mean(2)
    gy, gx = np.gradient(gray)
    edge = np.mean(np.hypot(gx,gy) > .025)
    texture = [float(gray.std()), float(edge), float(mask.mean())]
    return np.r_[mean, std, ratios, normdiff, texture].astype(np.float32)

def signature(cube, mask=None):
    if mask is None or mask.sum() < 20: mask = np.ones(cube.shape[:2], dtype=bool)
    return cube[mask].mean(0)
