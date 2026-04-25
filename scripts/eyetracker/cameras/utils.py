"""Frame-shape helpers shared across camera sources."""
import cv2
import numpy as np


def crop_to_aspect_ratio(image: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height

    if current_ratio > desired_ratio:
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped = image[:, offset:offset + new_width]
    else:
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped = image[offset:offset + new_height, :]

    return cv2.resize(cropped, (width, height))
