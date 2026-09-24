"""Generate ChArUco boards. Board specs live in charuco_boards.py.

--board screen (default)
    PNG to display fullscreen on a screen the scene cam can see (a second
    monitor is the cleanest setup), then run calibrate_scene_intrinsics.py.
    Output: scripts/extras/charuco_board.png (configurable via --out).

--board small | medium | large | print
    Exact-size US Letter PDFs for the dual-camera calibration jig ("print"
    makes all three). Print all sizes and check which one each camera detects
    reliably at the real working distance before mounting them on the jig.
    Output: scripts/extras/charuco_print/charuco_<size>.pdf (--out is the folder).
    Needs reportlab (`pip install reportlab`).

PRINTING
  Print the PDFs at 100% / "Actual size". "Fit to page" or "Shrink to fit"
  silently rescales the geometry. After printing, measure the 50 mm reference
  line in the top margin with a ruler to confirm the scale.
"""
import argparse
import io
from pathlib import Path

import cv2

from scripts.extras.charuco_boards import (
    MARGIN_BOTTOM_MM, MARGIN_SIDE_MM, PRINT_BOARDS, PRINT_DICT_NAME,
    SCREEN_DICT_NAME, build_print_board, build_screen_board,
)

PRINT_DPI = 600


def write_screen_png(out_path: Path, px_per_square: int) -> None:
    board = build_screen_board()
    squares_x, squares_y = board.getChessboardSize()
    out_w = squares_x * px_per_square
    out_h = squares_y * px_per_square
    img = board.generateImage((out_w, out_h),
                              marginSize=px_per_square // 4,
                              borderBits=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)

    print(f"Saved {out_path}  ({out_w}x{out_h})")
    print(f"Board: {squares_x}x{squares_y} squares, {SCREEN_DICT_NAME}")
    print()
    print("Display fullscreen on a screen the scene cam can see, then run:")
    print("  python -m scripts.extras.calibrate_scene_intrinsics")


def write_print_pdf(name: str, out_dir: Path) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    board = build_print_board(name)
    squares_x, squares_y = board.getChessboardSize()
    square_mm = board.getSquareLength()
    marker_mm = board.getMarkerLength()
    ids = board.getIds().ravel()
    board_w_mm = squares_x * square_mm
    board_h_mm = squares_y * square_mm

    # Render at exact size with no margin; the PDF places it on the page.
    px_per_mm = PRINT_DPI / 25.4
    img = board.generateImage((int(round(board_w_mm * px_per_mm)),
                               int(round(board_h_mm * px_per_mm))),
                              marginSize=0, borderBits=1)
    ok, png = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"failed to encode {name} board image")

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"charuco_{name}.pdf"
    page_w, page_h = letter
    c = canvas.Canvas(str(pdf_path), pagesize=letter)

    # Board anchored at the bottom margin, centred horizontally.
    draw_w = board_w_mm * mm
    draw_h = board_h_mm * mm
    c.drawImage(ImageReader(io.BytesIO(png.tobytes())),
                (page_w - draw_w) / 2, MARGIN_BOTTOM_MM * mm,
                width=draw_w, height=draw_h)

    # 50 mm reference ruler in the top margin.
    ruler_x = MARGIN_SIDE_MM * mm
    ruler_y = page_h - 8 * mm
    c.setLineWidth(0.8)
    c.line(ruler_x, ruler_y, ruler_x + 50 * mm, ruler_y)
    for tick in range(0, 51, 10):
        c.line(ruler_x + tick * mm, ruler_y, ruler_x + tick * mm, ruler_y - 2.5 * mm)
    c.setFont("Helvetica", 7)
    c.drawString(ruler_x + 52 * mm, ruler_y - 2 * mm,
                 "50 mm — verify with ruler before use")

    # Label below the ruler.
    c.drawString(ruler_x, page_h - 12 * mm,
                 f"{name.upper()}  sq={square_mm:.0f}mm  mk={marker_mm:.1f}mm  "
                 f"{squares_x}x{squares_y}  IDs {int(ids[0])}-{int(ids[-1])}  "
                 f"{PRINT_DICT_NAME}  100%/Actual size only")
    c.showPage()
    c.save()

    print(f"{name:>6}: {squares_x}x{squares_y} grid, square {square_mm:.0f} mm, "
          f"marker {marker_mm:.1f} mm, board {board_w_mm:.0f}x{board_h_mm:.0f} mm, "
          f"{len(ids)} markers (IDs {int(ids[0])}-{int(ids[-1])}) -> {pdf_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--board", default="screen",
                        choices=["screen", *PRINT_BOARDS, "print"])
    parser.add_argument("--out", default=None,
                        help="PNG path for --board screen; output folder for printed boards")
    parser.add_argument("--ppx", type=int, default=200,
                        help="screen board: output pixels per square")
    args = parser.parse_args()

    if args.board == "screen":
        write_screen_png(Path(args.out or "scripts/extras/charuco_board.png"), args.ppx)
        return

    out_dir = Path(args.out or "scripts/extras/charuco_print")
    names = list(PRINT_BOARDS) if args.board == "print" else [args.board]
    for name in names:
        write_print_pdf(name, out_dir)


if __name__ == "__main__":
    main()
