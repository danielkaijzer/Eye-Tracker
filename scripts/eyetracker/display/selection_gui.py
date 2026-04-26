"""Initial Tk dialog: pick a camera index or browse to a video file.

Returns a small (kind, value) tuple so the caller can dispatch to either
the live-camera or video-file code path without leaking Tk widgets across
function boundaries.
"""
import tkinter as tk
from tkinter import filedialog, ttk
from typing import List, Optional, Tuple, Union


# Result is one of:
#   ("camera", int)   — chosen cv2 index
#   ("video", str)    — chosen video file path
#   None              — user closed without picking
SelectionResult = Optional[Tuple[str, Union[int, str]]]


class SelectionGui:
    def __init__(self, title: str = "Pupil-only Eye Tracker"):
        self.title = title

    def pick(self, cameras: List[int]) -> SelectionResult:
        result_holder: list = [None]

        root = tk.Tk()
        root.title("Select Input Source")
        root.eval("tk::PlaceWindow . center")
        root.attributes("-topmost", True)
        root.update()
        root.attributes("-topmost", False)

        tk.Label(root, text=self.title, font=("Arial", 12, "bold")).pack(pady=10)
        tk.Label(root, text="Select Camera:").pack(pady=5)

        var = tk.StringVar()
        var.set(str(cameras[0]) if cameras else "No cameras found")
        ttk.Combobox(root, textvariable=var,
                     values=[str(c) for c in cameras]).pack(pady=5)

        def _start_camera():
            try:
                idx = int(var.get())
            except ValueError:
                idx = None
            if idx is not None:
                result_holder[0] = ("camera", idx)
            root.destroy()

        def _browse_video():
            path = filedialog.askopenfilename(
                filetypes=[("Video Files", "*.mp4;*.avi")]
            )
            if path:
                result_holder[0] = ("video", path)
            root.destroy()

        tk.Button(root, text="Start Camera", command=_start_camera).pack(pady=5)
        tk.Button(root, text="Browse Video", command=_browse_video).pack(pady=5)

        root.mainloop()
        return result_holder[0]
