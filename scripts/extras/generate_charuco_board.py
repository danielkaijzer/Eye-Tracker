"""Generate a printable / displayable ChArUco board for scene-cam intrinsic calibration.

Output: scripts/extras/charuco_board.png (configurable via --out).

Display fullscreen on a screen the scene cam can see (a second monitor is the
cleanest setup), then run calibrate_scene_intrinsics.py.

The board's absolute size doesn't affect K — ChArUco intrinsic calibration is
scale-invariant in the board geometry. So we don't bother with metric units.

Uses DICT_5X5_100 (different from the screen-corner DICT_4X4_50 markers) so
nothing in the existing pipeline can confuse the two.
"""
import argparse
from pathlib import Path

import cv2

DICT_NAME = "DICT_5X5_100"
SQUARES_X = 10
SQUARES_Y = 7
SQUARE_LEN = 1.0
MARKER_LEN = 0.75


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="scripts/extras/charuco_board.png")
    parser.add_argument("--squares-x", type=int, default=SQUARES_X)
    parser.add_argument("--squares-y", type=int, default=SQUARES_Y)
    parser.add_argument("--ppx", type=int, default=200,
                        help="output pixels per square")
    args = parser.parse_args()

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        SQUARE_LEN, MARKER_LEN, dictionary,
    )
    out_w = args.squares_x * args.ppx
    out_h = args.squares_y * args.ppx
    img = board.generateImage((out_w, out_h),
                              marginSize=args.ppx // 4,
                              borderBits=1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)

    print(f"Saved {out_path}  ({out_w}x{out_h})")
    print(f"Board: {args.squares_x}x{args.squares_y} squares, {DICT_NAME}")
    print()
    print("Display fullscreen on a screen the scene cam can see, then run:")
    print("  python -m scripts.extras.calibrate_scene_intrinsics")


if __name__ == "__main__":
    main()
