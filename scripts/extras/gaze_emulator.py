"""Gaze emulator — streams fake gaze data over UDP at 30Hz.

For teammates without eye-tracker hardware: lets them develop frontend
features against a plausible gaze stream. Run this on the same machine
as the consumer (defaults to localhost:9999).

Modes:
  (default)  gaze point follows the OS mouse cursor
  --auto     gaze point sweeps a Lissajous pattern across the screen

Each tick emits one JSON message:
  {
    "timestamp": <unix seconds, float>,
    "gaze_direction": [x, y, z],   # 3D unit vector, +Z toward viewer
    "gaze_point": [px, py],        # screen pixel
  }

Pass --ws to also serve the same payload over WebSocket so a browser
client (e.g. the Next.js frontend) can subscribe directly:
    const ws = new WebSocket("ws://localhost:9998")
    ws.onmessage = (e) => console.log(JSON.parse(e.data))
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import socket
import struct
import threading
import time
import tkinter as tk
from typing import Callable, List, Optional, Tuple


# Virtual viewer geometry — only used to turn a 2D gaze point into a
# 3D unit vector. Numbers don't have to be exact; the frontend just
# needs a stable, plausible direction.
ASSUMED_DISTANCE_MM = 600.0
SCREEN_WIDTH_MM = 344.0
SCREEN_HEIGHT_MM = 193.0

GazeSource = Callable[[float], Tuple[float, float]]


def gaze_direction(px: float, py: float, w: int,
                   h: int) -> Tuple[float, float, float]:
    """Unit vector from the virtual eye toward the gaze point on the
    screen. Eye sits at (0, 0, +D); screen lies in the z=0 plane,
    centered at the origin, +Y up."""
    cx = (px - w / 2.0) / w * SCREEN_WIDTH_MM
    cy = (h / 2.0 - py) / h * SCREEN_HEIGHT_MM
    cz = -ASSUMED_DISTANCE_MM
    norm = math.sqrt(cx * cx + cy * cy + cz * cz)
    return (cx / norm, cy / norm, cz / norm)


def make_mouse_source(width: int, height: int) -> GazeSource:
    root = tk.Tk()
    root.withdraw()

    def sample(_t: float) -> Tuple[float, float]:
        x = float(root.winfo_pointerx())
        y = float(root.winfo_pointery())
        return (max(0.0, min(width - 1.0, x)),
                max(0.0, min(height - 1.0, y)))
    return sample


def make_lissajous_source(width: int, height: int) -> GazeSource:
    def sample(t: float) -> Tuple[float, float]:
        x = (math.sin(0.7 * t) + 1.0) * 0.5 * (width - 1)
        y = (math.sin(1.1 * t + 0.5) + 1.0) * 0.5 * (height - 1)
        return (x, y)
    return sample


# ---- WebSocket broadcaster ------------------------------------------------
#
# Hand-rolled because we only need a server-to-client text-frame stream and
# don't want to drag in a dep. Inbound client traffic is ignored; dead
# clients are reaped lazily on the next failed sendall().

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _ws_handshake(conn: socket.socket) -> bool:
    data = b""
    while b"\r\n\r\n" not in data and len(data) < 8192:
        chunk = conn.recv(1024)
        if not chunk:
            return False
        data += chunk

    key: Optional[str] = None
    for line in data.split(b"\r\n")[1:]:
        k, sep, v = line.partition(b":")
        if sep and k.strip().lower() == b"sec-websocket-key":
            key = v.strip().decode()
            break
    if not key:
        return False

    accept = base64.b64encode(
        hashlib.sha1((key + _WS_GUID).encode()).digest()
    ).decode()
    response = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    )
    conn.sendall(response.encode())
    return True


def _ws_text_frame(payload: bytes) -> bytes:
    n = len(payload)
    if n < 126:
        return bytes([0x81, n]) + payload
    if n < 65536:
        return bytes([0x81, 126]) + struct.pack(">H", n) + payload
    return bytes([0x81, 127]) + struct.pack(">Q", n) + payload


class WsBroadcaster:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._clients: List[socket.socket] = []
        self._lock = threading.Lock()
        self._srv: Optional[socket.socket] = None

    def start(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen()
        self._srv = srv
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self) -> None:
        assert self._srv is not None
        while True:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            try:
                ok = _ws_handshake(conn)
            except OSError:
                ok = False
            if ok:
                with self._lock:
                    self._clients.append(conn)
            else:
                try:
                    conn.close()
                except OSError:
                    pass

    def broadcast(self, msg: str) -> None:
        frame = _ws_text_frame(msg.encode("utf-8"))
        with self._lock:
            survivors: List[socket.socket] = []
            for c in self._clients:
                try:
                    c.sendall(frame)
                    survivors.append(c)
                except OSError:
                    try:
                        c.close()
                    except OSError:
                        pass
            self._clients = survivors

    def close(self) -> None:
        if self._srv is not None:
            try:
                self._srv.close()
            except OSError:
                pass
        with self._lock:
            for c in self._clients:
                try:
                    c.close()
                except OSError:
                    pass
            self._clients.clear()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--rate", type=float, default=30.0,
                        help="stream rate in Hz (default 30)")
    parser.add_argument("--auto", action="store_true",
                        help="Lissajous sweep instead of mouse-follow")
    parser.add_argument("--ws", action="store_true",
                        help="also serve gaze over WebSocket (browser-friendly)")
    parser.add_argument("--ws-port", type=int, default=9998,
                        help="WebSocket port when --ws is set (default 9998)")
    args = parser.parse_args()

    if args.auto:
        source = make_lissajous_source(args.width, args.height)
        mode = "auto"
    else:
        source = make_mouse_source(args.width, args.height)
        mode = "mouse"

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period = 1.0 / args.rate
    addr = (args.host, args.port)

    ws: Optional[WsBroadcaster] = None
    if args.ws:
        ws = WsBroadcaster(args.host, args.ws_port)
        ws.start()

    print(f"Streaming gaze at {args.rate:.1f} Hz ({mode} mode):")
    print(f"  udp://{args.host}:{args.port}")
    if ws is not None:
        print(f"  ws://{args.host}:{args.ws_port}")
    print("Ctrl-C to stop.")

    t0 = time.monotonic()
    next_tick = t0
    try:
        while True:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
            t = time.monotonic() - t0
            px, py = source(t)
            dx, dy, dz = gaze_direction(px, py, args.width, args.height)
            payload = {
                "timestamp": time.time(),
                "gaze_direction": [round(dx, 6), round(dy, 6), round(dz, 6)],
                "gaze_point": [round(px, 1), round(py, 1)],
            }
            # print(payload)
            msg = json.dumps(payload)
            sock.sendto(msg.encode("utf-8"), addr)
            if ws is not None:
                ws.broadcast(msg)
            next_tick += period
            # If we fell behind (e.g. process suspended), don't burn a
            # burst of catch-up packets — resync from now.
            if next_tick < time.monotonic() - period:
                next_tick = time.monotonic() + period
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()
        if ws is not None:
            ws.close()


if __name__ == "__main__":
    main()
