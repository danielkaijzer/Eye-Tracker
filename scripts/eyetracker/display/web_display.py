"""Flask-based Display that streams the eye and scene frames as MJPEG.

Mirrors the framing/annotation behavior of CvDisplay (scene downscale +
gaze dot drawing) but pushes JPEG-encoded bytes to two HTTP endpoints
instead of cv2.imshow windows. Used by the Next.js dashboard.

Also exposes a small set of command endpoints (e.g. /load) that inject
a key char into a queue; poll_key drains the queue so the App's existing
key handler (`l` -> load calibration) does the work without any new
coupling between the Display and the App.
"""
import queue
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from flask import Flask, Response

from scripts.eyetracker.config import DISPLAY_HEIGHT, DISPLAY_WIDTH
from scripts.eyetracker.display.base import Display, XY


_JPEG_QUALITY = 70


class _FrameSlot:
    """Latest-frame slot. Producers write bytes; consumers wait on `event`
    for a new frame, read `data`, and clear the event before re-waiting."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Optional[bytes] = None
        self.event = threading.Event()

    def put(self, data: bytes) -> None:
        with self._lock:
            self._data = data
        self.event.set()

    def get(self) -> Optional[bytes]:
        with self._lock:
            return self._data


class WebDisplay(Display):
    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int = 5001,
                 display_size: Tuple[int, int] = (DISPLAY_WIDTH, DISPLAY_HEIGHT)):
        self.host = host
        self.port = port
        self.display_w, self.display_h = display_size

        self._eye = _FrameSlot()
        self._scene = _FrameSlot()
        self._key_queue: "queue.Queue[str]" = queue.Queue()
        self._app = Flask(__name__)
        self._register_routes()
        self._server_thread: Optional[threading.Thread] = None
        self._opened = False

    def _register_routes(self) -> None:
        app = self._app

        @app.after_request  # pyright: ignore[reportUnusedFunction]
        def _cors(resp):
            resp.headers["Access-Control-Allow-Origin"] = "*"
            resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            return resp

        @app.route("/eye.mjpg")  # pyright: ignore[reportUnusedFunction]
        def eye_feed():
            return Response(_mjpeg_generator(self._eye),
                            mimetype="multipart/x-mixed-replace; boundary=frame")

        @app.route("/scene.mjpg")  # pyright: ignore[reportUnusedFunction]
        def scene_feed():
            return Response(_mjpeg_generator(self._scene),
                            mimetype="multipart/x-mixed-replace; boundary=frame")

        # Plain GET so browsers don't preflight; idempotent enough for MVP.
        @app.route("/load", methods=["GET", "POST"])  # pyright: ignore[reportUnusedFunction]
        def load_calibration():
            self._key_queue.put("l")
            return ("", 204)

    # ---- Display interface -------------------------------------------------

    def open(self) -> None:
        if self._opened:
            return
        self._server_thread = threading.Thread(
            target=lambda: self._app.run(host=self.host, port=self.port,
                                         threaded=True, debug=False,
                                         use_reloader=False),
            daemon=True,
        )
        self._server_thread.start()
        self._opened = True
        print(f"WebDisplay: streaming at http://{self.host}:{self.port}/"
              f"{{eye,scene}}.mjpg")

    def close(self) -> None:
        # Daemon thread dies with the process; nothing to do.
        self._opened = False

    def show_eye(self, frame: np.ndarray) -> None:
        encoded = _encode_jpeg(frame)
        if encoded is not None:
            self._eye.put(encoded)

    def show_scene(self, frame: np.ndarray, gaze_xy: Optional[XY]) -> None:
        scene_h, scene_w = frame.shape[:2]
        resized = cv2.resize(frame, (self.display_w, self.display_h))
        if gaze_xy is not None and scene_w > 0 and scene_h > 0:
            disp_x = int(gaze_xy[0] * self.display_w / scene_w)
            disp_y = int(gaze_xy[1] * self.display_h / scene_h)
            cv2.circle(resized, (disp_x, disp_y), 10, (0, 0, 255), 2)
        encoded = _encode_jpeg(resized)
        if encoded is not None:
            self._scene.put(encoded)

    def poll_key(self) -> Tuple[Optional[str], int]:
        # Web display has no real keyboard; HTTP endpoints push command keys
        # (e.g. 'l' from /load) into the queue and we surface them here so
        # App._handle_key dispatches them like any other keypress.
        time.sleep(0.001)
        try:
            ch = self._key_queue.get_nowait()
        except queue.Empty:
            return None, 255
        raw = ord(ch) if len(ch) == 1 else 255
        return ch, raw

    def wait_for_pause(self) -> None:
        time.sleep(0.05)


def _encode_jpeg(frame: np.ndarray) -> Optional[bytes]:
    ok, buf = cv2.imencode(".jpg", frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
    if not ok:
        return None
    return buf.tobytes()


def _mjpeg_generator(slot: _FrameSlot):
    """Yield multipart frames whenever the slot gets a new frame.
    Waits on the slot's event so we don't busy-loop or send duplicates."""
    while True:
        slot.event.wait()
        slot.event.clear()
        data = slot.get()
        if data is None:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
               + data + b"\r\n")
