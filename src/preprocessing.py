import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

def preprocess(image, size=(480, 320)):
    raw = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    denoised = raw.filter(ImageFilter.MedianFilter(3))
    normalized = ImageEnhance.Contrast(denoised).enhance(1.12)
    arr = np.asarray(normalized)
    # Simple foreground ROI from pixels differing from the image-border background.
    border = np.concatenate([arr[0], arr[-1], arr[:,0], arr[:,-1]])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(arr.astype(float)-bg, axis=2)
    mask = dist > max(18, float(np.percentile(dist, 45)))
    if mask.mean() < .08: mask[:] = True
    return raw, denoised, normalized, mask
