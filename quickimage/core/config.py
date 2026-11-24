"""Tuning constants shared by the image-processing modules."""


import numpy as np


BG_TOL = 90
SOFT_TOL = 330
MAX_BG_FRAC = 0.97
DEFAULT_SVG_COLORS = 4
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".ico"}
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)
PREVIEW_MAX_SIDE = 1600  # live previews run on a proxy no larger than this
DIST_CHUNK = 1 << 18  # pixels per chunk when assigning pixels to palette colors
SEED_REACH = 3  # px around a picked region where the picked color acts as background

BOX3 = np.ones((3, 3), dtype=bool)
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
