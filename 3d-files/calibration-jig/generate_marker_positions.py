"""Generate marker_positions.json from calibration_jig.scad.

Parses simple `name = value;` parameter lines out of the .scad file so the
sidecar coordinates are always in sync with the CAD model. Re-run after any
parameter edit.

Coordinate frame (matches the .scad header):
  Origin: midpoint of base top surface, between panel inner faces.
  +X: Panel A → Panel B.
  +Y: lateral (panel width).
  +Z: up.
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
SCAD_PATH = HERE / "calibration_jig.scad"
JSON_PATH = HERE / "marker_positions.json"

PARAM_LINE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*([-+]?\d+(?:\.\d+)?)\s*;")


def parse_scad_params(text: str) -> dict[str, float]:
    params: dict[str, float] = {}
    for line in text.splitlines():
        m = PARAM_LINE.match(line)
        if m:
            params[m.group(1)] = float(m.group(2))
    return params


def slot_centers(span: float, count: int) -> list[float]:
    if count <= 1:
        return [0.0]
    return [-span / 2 + i * span / (count - 1) for i in range(count)]


def check_no_overlap(axis: str, span: float, count: int, ms: float) -> None:
    """Raise if adjacent markers along this axis would overlap.

    `span` is the distance between the centers of the outermost markers
    (panel_extent - 2*margin - ms). Pitch is span/(count-1).
    """
    if count <= 1:
        return
    pitch = span / (count - 1)
    if pitch < ms:
        raise ValueError(
            f"{count} markers of size {ms} mm along {axis} would overlap "
            f"(pitch {pitch:.2f} mm < marker {ms} mm). Increase panel size, "
            f"shrink marker_size, or reduce grid count."
        )


def main() -> None:
    p = parse_scad_params(SCAD_PATH.read_text())

    sep    = p["panel_separation_x"]
    pw     = p["panel_width"]
    ph     = p["panel_height"]
    bt     = p["base_thickness"]
    ms     = p["marker_size"]
    cols   = int(p["marker_grid_cols"])
    rows   = int(p["marker_grid_rows"])
    margin = p["marker_margin"]

    y_span = pw - 2 * margin - ms
    z_span_extent = ph - 2 * margin - ms  # only used for the overlap check
    check_no_overlap("Y", y_span, cols, ms)
    check_no_overlap("Z", z_span_extent, rows, ms)
    ys = slot_centers(y_span, cols)
    z_min = bt + margin + ms / 2
    z_max = bt + ph - margin - ms / 2
    if rows <= 1:
        zs = [(z_min + z_max) / 2]
    else:
        zs = [z_min + i * (z_max - z_min) / (rows - 1) for i in range(rows)]

    markers = []
    marker_id = 0
    # (panel label, inner-face x, outward normal x sign — i.e. direction the
    # marker faces, which is *toward the origin* from the panel)
    panels = [("A", -sep / 2, +1), ("B", +sep / 2, -1)]
    for label, x_face, normal_x in panels:
        for ri, z_c in enumerate(zs):
            for ci, y_c in enumerate(ys):
                # Corners in (Y, Z) with +Y = "right" and +Z = "up" in the
                # panel's local frame. Order: TL, TR, BR, BL.
                corners = {
                    "TL": [x_face, y_c - ms / 2, z_c + ms / 2],
                    "TR": [x_face, y_c + ms / 2, z_c + ms / 2],
                    "BR": [x_face, y_c + ms / 2, z_c - ms / 2],
                    "BL": [x_face, y_c - ms / 2, z_c - ms / 2],
                }
                markers.append({
                    "id": marker_id,
                    "panel": label,
                    "row": ri,
                    "col": ci,
                    "size_mm": ms,
                    "center_mm": [x_face, y_c, z_c],
                    "normal": [normal_x, 0, 0],
                    "corners_mm": corners,
                })
                marker_id += 1

    out = {
        "units": "millimeters",
        "coordinate_frame": {
            "origin": "midpoint of base top surface, between panel inner faces",
            "+X": "from Panel A toward Panel B",
            "+Y": "lateral (along panel width)",
            "+Z": "up",
        },
        "panel_separation_mm": sep,
        "corner_order": "TL, TR, BR, BL in panel-local (Y, Z); flip if your detector uses a different convention",
        "source_params": {
            "panel_separation_x": sep,
            "panel_width": pw,
            "panel_height": ph,
            "base_thickness": bt,
            "marker_size": ms,
            "marker_grid_cols": cols,
            "marker_grid_rows": rows,
            "marker_margin": margin,
        },
        "markers": markers,
    }
    JSON_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(markers)} markers to {JSON_PATH}")


if __name__ == "__main__":
    main()
