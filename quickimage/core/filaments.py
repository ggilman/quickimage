"""Bambu Lab filament colors, and mapping a binned palette onto the filaments you have loaded."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "bambu_filaments.json"
DEFAULT_MATERIAL = "PLA Basic"


def hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


@lru_cache(maxsize=1)
def load_catalog(path=CATALOG_PATH):
    """All solid Bambu filament colors: tuple of dicts with material, name, hex, rgb, code."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(dict(f, rgb=hex_to_rgb(f["hex"])) for f in data["filaments"])


def materials(catalog=None):
    """Material names in catalog order (PLA Basic first, as in Bambu's list)."""
    seen = []
    for f in catalog if catalog is not None else load_catalog():
        if f["material"] not in seen:
            seen.append(f["material"])
    return seen


def material_filaments(material, catalog=None):
    """The colors of one material, without repeated colors (the candidates for automatic matching)."""
    out, seen = [], set()
    for f in catalog if catalog is not None else load_catalog():
        if f["material"] == material and f["rgb"] not in seen:
            seen.add(f["rgb"])
            out.append(f)
    return out


def find_filament(material, name, catalog=None):
    for f in catalog if catalog is not None else load_catalog():
        if f["material"] == material and f["name"] == name:
            return f
    return None


# OKLab with the color axes (a, b) weighted twice as much as lightness. Plain CIELAB pulls vivid
# blues toward purple (e.g. pure blue matched Bambu's "Purple"), and plain OKLab lets lightness
# outweigh color, so near-black or gray could match a dark or muted colored filament.
OKLAB_WEIGHTS = (1.0, 2.0, 2.0)


def _perceptual(rgb):
    """(N, 3) sRGB 0..255 -> weighted OKLab, where Euclidean distance tracks how different colors look."""
    c = np.asarray(rgb, dtype=np.float64)[:, :3] / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m1 = np.array(
        [[0.4122214708, 0.5363325363, 0.0514459929],
         [0.2119034982, 0.6806995451, 0.1073969566],
         [0.0883024619, 0.2817188376, 0.6299787005]]
    )
    m2 = np.array(
        [[0.2104542553, 0.7936177850, -0.0040720468],
         [1.9779984951, -2.4285922050, 0.4505937099],
         [0.0259040371, 0.7827717662, -0.8086757660]]
    )
    return (np.cbrt(c @ m1.T) @ m2.T) * np.asarray(OKLAB_WEIGHTS)


def filament_overrides(raw_palette, filament_rgbs):
    """Map each binned color to one of the chosen filaments: `{palette index: filament rgb}`.

    Colors and filaments are paired one-to-one so that the total color difference
    is smallest (see `_perceptual`), which keeps distinct regions on distinct filaments. With more filaments than
    colors this picks the best-matching subset, so passing a whole material's colors chooses
    the best N filaments automatically. With more colors than filaments, the leftover colors
    take their nearest filament.
    """
    if not raw_palette or not filament_rgbs:
        return {}
    pal = _perceptual(np.asarray(raw_palette))
    fil = _perceptual(np.asarray(filament_rgbs))
    cost = np.linalg.norm(pal[:, None, :] - fil[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    out = {int(r): tuple(int(v) for v in filament_rgbs[c]) for r, c in zip(rows, cols)}
    for i in range(len(raw_palette)):
        if i not in out:
            out[i] = tuple(int(v) for v in filament_rgbs[int(np.argmin(cost[i]))])
    return out
