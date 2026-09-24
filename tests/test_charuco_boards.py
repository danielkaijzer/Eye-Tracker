"""Printed jig boards share one ArUco dictionary, so their marker-ID ranges
must fit in it and never overlap (a camera could otherwise detect markers from
the wrong board)."""
import cv2

from scripts.extras.charuco_boards import PRINT_BOARDS, PRINT_DICT_NAME, build_print_board


def test_print_board_ids_fit_dictionary_and_do_not_overlap():
    dict_size = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, PRINT_DICT_NAME)).bytesList.shape[0]
    seen = {}
    for name in PRINT_BOARDS:
        ids = build_print_board(name).getIds().ravel()
        assert ids.max() < dict_size, f"{name} uses IDs past {dict_size - 1}"
        for other, other_ids in seen.items():
            assert not set(ids) & other_ids, f"{name} and {other} share marker IDs"
        seen[name] = set(ids)
