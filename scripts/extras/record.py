import cv2
import time
import threading
from flask import Flask, Response

# --- Flask App Setup ---
app = Flask(__name__)
latest_frame = None
frame_lock = threading.Lock() # Ensures Flask and OpenCV don't read/write the frame at the exact same microsecond

def generate_frames():
    global latest_frame, frame_lock
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            # Encode the current frame to JPEG
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Small sleep to prevent the web stream from hogging CPU
        time.sleep(0.03) 

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    # use_reloader=False is critical when running Flask inside a thread
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# --- Threaded Camera Setup ---
class ThreadedCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.ret, self.frame = self.cap.read()
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.stopped = False
        self.new_frame_ready = False 

    def start(self):
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True 
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()
            self.new_frame_ready = True 

    def read(self):
        is_new = self.new_frame_ready
        self.new_frame_ready = False 
        return self.ret, self.frame, is_new

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

# --- Main Execution ---
def main():
    global latest_frame, frame_lock

    FLIP_IR_CAMERA = True  
    EYE_CAMERA_INDEX = 0

    print("Initializing eye camera...")
    eye_cam = ThreadedCamera(EYE_CAMERA_INDEX)

    if not eye_cam.ret:
        print(f"Error: Could not open camera {EYE_CAMERA_INDEX}.")
        return

    eye_cam.start()

    # Start Flask Web Server in a daemon thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    print("Web stream started! Go to http://<your-jetson-ip>:5000 in your Mac's browser.")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_name = f"eye_cam_{timestamp}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    fps = 30.0 
    out = cv2.VideoWriter(file_name, fourcc, fps, (eye_cam.width, eye_cam.height))

    print(f"\nRecording started...\n Saving video to: {file_name}")
    print("Press Ctrl+C in this terminal to STOP recording and save the file.")

    start_time = time.time()
    frames_recorded = 0

    try:
        while True:
            ret, frame, is_new = eye_cam.read()

            if not is_new:
                time.sleep(0.001)
                continue

            if not ret:
                print("Error: Lost connection to the camera stream.")
                break

            if FLIP_IR_CAMERA:
                frame = cv2.flip(frame, 0)

            # 1. Write to the file
            out.write(frame)
            frames_recorded += 1

            # 2. Update the global frame for Flask
            with frame_lock:
                latest_frame = frame.copy()

    except KeyboardInterrupt:
        # This catches the Ctrl+C from the user
        print("\n\nInterrupted by user. Stopping recording...")

    # --- Stop Timer & Calculate ---
    end_time = time.time()
    elapsed_time = end_time - start_time
    actual_fps = frames_recorded / elapsed_time if elapsed_time > 0 else 0

    print("\n--- Recording Session Stats ---")
    print(f"Total time elapsed: {elapsed_time:.2f} seconds")
    print(f"Total frames saved: {frames_recorded}")
    print(f"True Capture FPS: {actual_fps:.2f} fps")
    print("-------------------------------\n")

    eye_cam.stop()
    out.release()
    print("Recording successfully saved and stream closed.")

if __name__ == "__main__":
    main()