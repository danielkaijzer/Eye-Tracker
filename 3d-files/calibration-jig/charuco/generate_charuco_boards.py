"""Generate printable ChArUco boards at several physical sizes for the
dual-camera calibration jig.

WHY THIS EXISTS
  We don't yet know the smallest ChArUco square the headset cameras can
  resolve at the jig's ~150 mm working distance (panels sit ~305 mm apart,
  glasses in the middle). Rather than guess, print these size variants, tape
  one to a panel, and check which detects reliably before committing to a
  laminated final board.

OUTPUT (per size, in this directory)
  charuco_<label>.pdf  — US-Letter page, board drawn at EXACT physical size,
                         with a printed 50 mm reference ruler for scale check.
  charuco_<label>.png  — raw board image (for on-screen reference only).

PRINTING (at the library)
  Print the PDFs at 100% / "Actual size". Do NOT use "Fit to page" / "Shrink
  to fit" — that rescales the board and silently breaks the geometry. After
  printing, measure the 50 mm reference line with a ruler; if it isn't 50 mm,
  the print was scaled and the board is unusable for metric calibration.

IDS
  Every board uses DICT_5X5_1000 but a distinct ID range (id_offset) so no two
  boards share marker IDs. That means you can test them without collisions, and
  later the two jig panels can carry boards with different ID ranges so the
  solver can tell Panel A from Panel B.

Re-run after editing BOARDS below:
    python3 generate_charuco_boards.py
"""

import cv2
import numpy as np
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

HERE = Path(__file__).parent

# Shared layout. 5x7 squares keeps even the largest board on a Letter page and
# guarantees several markers fall inside any partial camera view.
SQUARES_X = 5
SQUARES_Y = 7
DICT = cv2.aruco.DICT_5X5_1000
MARKER_RATIO = 0.72          # marker edge as fraction of square edge
DPI = 600                    # render resolution for the embedded image
QUIET_MM = 6.0               # white border baked into the image (detection)

# Size variants to print and test (square edge in mm). Brackets the plausible
# range for a ~150 mm working distance with an unknown low-res eye camera.
BOARDS = [
    {"label": "small",  "square_mm": 15.0, "id_offset": 0},
    {"label": "medium", "square_mm": 22.0, "id_offset": 100},
    {"label": "large",  "square_mm": 30.0, "id_offset": 200},
]


def make_board(square_mm: float, id_offset: int):
    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    marker_mm = square_mm * MARKER_RATIO
    n_markers = (SQUARES_X * SQUARES_Y) // 2
    ids = np.arange(id_offset, id_offset + n_markers, dtype=np.int32)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), square_mm, marker_mm, dictionary, ids
    )
    return board, marker_mm, ids


def render_png(board, square_mm: float, path: Path) -> tuple[float, float]:
    """Render board to PNG with a baked-in white quiet zone. Returns the
    image's full physical (width_mm, height_mm) including the quiet zone."""
    board_w_mm = SQUARES_X * square_mm
    board_h_mm = SQUARES_Y * square_mm
    px_per_mm = DPI / 25.4
    margin_px = int(round(QUIET_MM * px_per_mm))
    w_px = int(round(board_w_mm * px_per_mm)) + 2 * margin_px
    h_px = int(round(board_h_mm * px_per_mm)) + 2 * margin_px
    img = board.generateImage((w_px, h_px), marginSize=margin_px, borderBits=1)
    cv2.imwrite(str(path), img)
    full_w_mm = board_w_mm + 2 * QUIET_MM
    full_h_mm = board_h_mm + 2 * QUIET_MM
    return full_w_mm, full_h_mm


def write_pdf(png_path: Path, pdf_path: Path, full_w_mm: float,
              full_h_mm: float, spec: dict, marker_mm: float, ids) -> None:
    page_w, page_h = letter
    c = canvas.Canvas(str(pdf_path), pagesize=letter)

    # Anchor the board at the bottom margin so the label + reference ruler get
    # the clear space at the top of the page (the largest board nearly fills
    # the sheet, so there's no room below it).
    bottom_margin = 12 * mm
    draw_w = full_w_mm * mm
    draw_h = full_h_mm * mm
    x = (page_w - draw_w) / 2
    y = bottom_margin
    board_top = y + draw_h
    c.drawImage(ImageReader(str(png_path)), x, y, width=draw_w, height=draw_h)

    # Label block, top-down from the page top.
    c.setFont("Helvetica-Bold", 13)
    c.drawString(18 * mm, page_h - 15 * mm,
                 f"ChArUco test board — {spec['label'].upper()}")
    c.setFont("Helvetica", 10)
    lines = [
        f"square = {spec['square_mm']:.1f} mm    marker = {marker_mm:.1f} mm"
        f"    grid = {SQUARES_X} x {SQUARES_Y}",
        f"dict = DICT_5X5_1000    marker IDs {int(ids[0])}-{int(ids[-1])}",
        "Print at 100% / Actual size. Verify the 50 mm line below with a ruler.",
    ]
    ty = page_h - 22 * mm
    for ln in lines:
        c.drawString(18 * mm, ty, ln)
        ty -= 5 * mm

    # 50 mm reference ruler (scale sanity check after printing), placed just
    # below the label text and above the board.
    ruler_y = ty - 4 * mm
    if ruler_y < board_top + 4 * mm:
        raise ValueError(
            f"{spec['label']}: board too tall for label+ruler on Letter "
            f"(board_top={board_top/mm:.1f}mm, ruler_y={ruler_y/mm:.1f}mm). "
            f"Reduce square size, grid, or QUIET_MM."
        )
    rx = 18 * mm
    c.setLineWidth(1)
    c.line(rx, ruler_y, rx + 50 * mm, ruler_y)
    for tick in range(0, 51, 10):
        c.line(rx + tick * mm, ruler_y, rx + tick * mm, ruler_y + 3 * mm)
    c.setFont("Helvetica", 8)
    c.drawString(rx + 52 * mm, ruler_y, "50 mm reference")

    c.showPage()
    c.save()


def main() -> None:
    for spec in BOARDS:
        board, marker_mm, ids = make_board(spec["square_mm"], spec["id_offset"])
        png = HERE / f"charuco_{spec['label']}.png"
        pdf = HERE / f"charuco_{spec['label']}.pdf"
        full_w, full_h = render_png(board, spec["square_mm"], png)
        write_pdf(png, pdf, full_w, full_h, spec, marker_mm, ids)
        print(f"{spec['label']:>6}: square {spec['square_mm']:.0f} mm, "
              f"marker {marker_mm:.1f} mm, board "
              f"{SQUARES_X*spec['square_mm']:.0f}x{SQUARES_Y*spec['square_mm']:.0f} mm "
              f"-> {pdf.name}")


if __name__ == "__main__":
    main()
