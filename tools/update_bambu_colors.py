"""Rebuild quickimage/data/bambu_filaments.json from Bambu Studio's official filament color list.

Source: resources/profiles/BBL/filament/filaments_color_codes.json in
https://github.com/bambulab/BambuStudio (the list Bambu Studio uses for its filament colors).
Only solid (single-color) filaments are kept: gradient and multi-color spools have no single
flat color to match against.

Run:  .venv\\Scripts\\python tools\\update_bambu_colors.py [path\\to\\filaments_color_codes.json]
Without a path, the file is downloaded from GitHub.
"""

import json
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/bambulab/BambuStudio/master/"
    "resources/profiles/BBL/filament/filaments_color_codes.json"
)
OUT = Path(__file__).resolve().parent.parent / "quickimage" / "data" / "bambu_filaments.json"
SOLID = "单色"  # Bambu's fila_color_type for single-color filaments


def build(raw):
    rows = []
    seen = set()
    for r in raw["data"]:
        colors = r.get("fila_color") or []
        if r.get("fila_color_type") != SOLID or len(colors) != 1:
            continue
        hex_rgba = colors[0].lstrip("#")
        name = (r.get("fila_color_name") or {}).get("en") or r.get("color_code") or hex_rgba
        key = (r["fila_type"], name)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "material": r["fila_type"],
                "name": name,
                "hex": "#" + hex_rgba[:6].upper(),
                "code": r.get("fila_color_code", ""),
            }
        )
    return {"source": URL, "filaments": rows}


def main():
    if len(sys.argv) > 1:
        raw = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(URL, timeout=30) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    data = build(raw)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(data['filaments'])} filament colors to {OUT}")


if __name__ == "__main__":
    main()
