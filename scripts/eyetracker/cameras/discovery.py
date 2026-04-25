"""Probe local /dev/video* (or platform equivalent) indexes for openable cameras."""
import cv2


def detect_cameras(max_cams: int = 10) -> list[int]:
    available = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available
