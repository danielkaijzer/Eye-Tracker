"""Generate printable ChArUco boards at several square sizes for the
dual-camera calibration jig.

Each board tiles as many squares as fit on a US-Letter page so the
pattern fills the sheet. Since only a portion of the board needs to be
visible to the camera at any time, filling the page maximises the
calibration points in every partial view.

Print all three sizes, tape them up, and check which square size each
camera detects reliably at the real working distance before laminating.

PRINTING
  Print the PDFs at 100% / "Actual size". Do NOT use "Fit to page" or
  "Shrink to fit" — that rescales the geometry silently. After printing,
  measure the 50 mm reference line in the bottom margin with a ruler to
  confirm the scale is correct.

Re-run after editing BOARDS or page geometry:
    python3 generate_charuco_boards.py
"""

import math
import cv2
import numpy as np
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

HERE = Path(__file__).parent

DICT         = cv2.aruco.DICT_5X5_1000
MARKER_RATIO = 0.72   # marker edge as fraction of square edge
DPI          = 600

# Page margins (mm). Bottom margin is taller to fit the reference ruler.
MARGIN_TOP    = 8.0
MARGIN_SIDE   = 8.0
MARGIN_BOTTOM = 14.0  # ruler lives here

BOARDS = [
    {"label": "small",  "square_mm": 15.0, "id_offset": 0},
    {"label": "medium", "square_mm": 22.0, "id_offset": 200},
    {"label": "large",  "square_mm": 30.0, "id_offset": 400},
]

PAGE_W_MM = 215.9  # US Letter
PAGE_H_MM = 279.4


def tile_counts(square_mm: float) -> tuple[int, int]:
    usable_w = PAGE_W_MM - 2 * MARGIN_SIDE
    usable_h = PAGE_H_MM - MARGIN_TOP - MARGIN_BOTTOM
    return math.floor(usable_w / square_mm), math.floor(usable_h / square_mm)


def make_board(cols: int, rows: int, square_mm: float, id_offset: int):
    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    marker_mm  = square_mm * MARKER_RATIO
    n_markers  = (cols * rows) // 2
    ids        = np.arange(id_offset, id_offset + n_markers, dtype=np.int32)
    board      = cv2.aruco.CharucoBoard(
        (cols, rows), square_mm, marker_mm, dictionary, ids
    )
    return board, marker_mm, ids


def render_png(board, cols: int, rows: int, square_mm: float,
               path: Path) -> tuple[float, float]:
    """Render board at exact size, no margins. Returns (width_mm, height_mm)."""
    px_per_mm = DPI / 25.4
    w_px = int(round(cols * square_mm * px_per_mm))
    h_px = int(round(rows * square_mm * px_per_mm))
    img  = board.generateImage((w_px, h_px), marginSize=0, borderBits=1)
    cv2.imwrite(str(path), img)
    return cols * square_mm, rows * square_mm


def write_pdf(png_path: Path, pdf_path: Path, board_w_mm: float,
              board_h_mm: float, spec: dict, cols: int, rows: int,
              marker_mm: float, ids) -> None:
    page_w, page_h = letter
    c = canvas.Canvas(str(pdf_path), pagesize=letter)

    # Board anchored at top margin, centred horizontally.
    draw_w = board_w_mm * mm
    draw_h = board_h_mm * mm
    x = (page_w - draw_w) / 2
    y = page_h - MARGIN_TOP * mm - draw_h
    c.drawImage(ImageReader(str(png_path)), x, y, width=draw_w, height=draw_h)

    # 50 mm reference ruler in the bottom margin.
    rx = MARGIN_SIDE * mm
    ry = 6 * mm
    c.setLineWidth(0.8)
    c.line(rx, ry, rx + 50 * mm, ry)
    for tick in range(0, 51, 10):
        c.line(rx + tick * mm, ry, rx + tick * mm, ry + 2.5 * mm)
    c.setFont("Helvetica", 7)
    c.drawString(rx + 52 * mm, ry, "50 mm — verify with ruler before use")

    # Tiny label at very bottom-left.
    c.setFont("Helvetica", 7)
    c.drawString(rx, 2 * mm,
                 f"{spec['label'].upper()}  sq={spec['square_mm']:.0f}mm  "
                 f"mk={marker_mm:.1f}mm  {cols}x{rows}  "
                 f"IDs {int(ids[0])}-{int(ids[-1])}  DICT_5X5_1000  "
                 f"100%/Actual size only")

    c.showPage()
    c.save()


def main() -> None:
    for spec in BOARDS:
        cols, rows = tile_counts(spec["square_mm"])
        board, marker_mm, ids = make_board(
            cols, rows, spec["square_mm"], spec["id_offset"]
        )
        png = HERE / f"charuco_{spec['label']}.png"
        pdf = HERE / f"charuco_{spec['label']}.pdf"
        w_mm, h_mm = render_png(board, cols, rows, spec["square_mm"], png)
        write_pdf(png, pdf, w_mm, h_mm, spec, cols, rows, marker_mm, ids)
        print(f"{spec['label']:>6}: {cols}x{rows} grid, square {spec['square_mm']:.0f} mm, "
              f"marker {marker_mm:.1f} mm, "
              f"board {w_mm:.0f}x{h_mm:.0f} mm, "
              f"{(cols*rows)//2} markers (IDs {int(ids[0])}-{int(ids[-1])})"
              f" -> {pdf.name}")


if __name__ == "__main__":
    main()
