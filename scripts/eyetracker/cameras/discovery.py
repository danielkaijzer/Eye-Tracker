"""Probe local /dev/video* (or platform equivalent) indexes for openable cameras."""
import sys
from typing import List, Optional

import cv2

from scripts.eyetracker.cameras.v4l2 import index_for_usb_id, list_capture_devices

IS_LINUX = sys.platform.startswith("linux")


def detect_cameras(max_cams: int = 10) -> list[int]:
    if IS_LINUX:
        # sysfs lists capture nodes directly; blind-probing indexes would also
        # hit each camera's metadata node and spam V4L2 open warnings.
        return [idx for idx, _, _ in list_capture_devices()]
    available = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def eye_first(cameras: List[int], eye_usb_id: Optional[str]) -> List[int]:
    """Reorder so the eye cam (matched by USB id, Linux only) comes first —
    the picker defaults to the first entry."""
    if IS_LINUX and eye_usb_id:
        idx = index_for_usb_id(eye_usb_id)
        if idx in cameras:
            return [idx] + [c for c in cameras if c != idx]
    return cameras


def pick_scene_index(eye_index: int, cameras: List[int],
                     scene_usb_id: Optional[str]) -> int:
    """Choose the scene camera given the user-picked eye camera. On Linux,
    match the scene cam's USB vendor:product; otherwise (or if it isn't found)
    take the first other detected camera."""
    if IS_LINUX and scene_usb_id:
        idx = index_for_usb_id(scene_usb_id, exclude=eye_index)
        if idx is not None:
            return idx
    others = [c for c in cameras if c != eye_index]
    if others:
        return others[0]
    return 1 if eye_index == 0 else 0
