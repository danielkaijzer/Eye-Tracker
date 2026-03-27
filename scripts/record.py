import cv2
import time
import threading

class ThreadedCamera:
    """
    A class that continuously reads frames from a VideoCapture object
    in a dedicated background thread to prevent I/O blocking.
    """
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Read the first frame to ensure the camera is working
        self.ret, self.frame = self.cap.read()
        
        # Get actual dimensions assigned by OpenCV
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.stopped = False
        self.new_frame_ready = False # NEW: Track when a fresh frame arrives

    def start(self):
        # Start the background thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True # Ensures thread closes when main script exits
        self.thread.start()
        return self

    def update(self):
        # Keep looping and grabbing the latest frame from the buffer
        while not self.stopped:
            self.ret, self.frame = self.cap.read()
            self.new_frame_ready = True # NEW: Signal that the frame changed

    def read(self):
        # Return the frame along with the readiness flag
        is_new = self.new_frame_ready
        self.new_frame_ready = False # Lower the flag once the main thread sees it
        return self.ret, self.frame, is_new

    def stop(self):
        # Stop the thread and release the camera
        self.stopped = True
        self.thread.join()
        self.cap.release()

def main():
    # --- Configuration ---
    FLIP_IR_CAMERA = True  
    EYE_CAMERA_INDEX = 0 # Set to your GC0308 index 

    print("Initializing eye camera... (this may take a second)")
    
    # Initialize and start the threaded camera
    eye_cam = ThreadedCamera(EYE_CAMERA_INDEX)

    if not eye_cam.ret:
        print(f"Error: Could not open camera with index {EYE_CAMERA_INDEX}.")
        print("Please make sure your camera is connected and the correct index is used.")
        return

    # Start the background thread
    eye_cam.start()

    # --- Setup Video Recording ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_name = f"eye_cam_{timestamp}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    fps = 30.0 
    
    out = cv2.VideoWriter(file_name, fourcc, fps, (eye_cam.width, eye_cam.height))

    print(f"Recording started...\n Saving video to: {file_name}")
    print("Press 'q' to stop.")

    start_time = time.time()
    frames_recorded = 0

    while True:
        # Check the background thread
        ret, frame, is_new = eye_cam.read()

        # NEW: If the background thread hasn't pulled a new frame yet, wait and try again
        if not is_new:
            time.sleep(0.001) # Sleep for 1ms to prevent hammering the CPU
            continue

        if not ret:
            print("Error: Lost connection to the camera stream. Exiting...")
            break

        # --- Flip IR Camera Feed if Enabled ---
        if FLIP_IR_CAMERA:
            frame = cv2.flip(frame, 0) # 0 for vertical flip

        # --- Write to Video File ---
        out.write(frame)
        frames_recorded += 1

        # --- Display Logic ---
        cv2.imshow('GC0308 Eye Camera Test', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # --- Stop Timer & Calculate ---
    end_time = time.time()
    elapsed_time = end_time - start_time
    actual_fps = frames_recorded / elapsed_time if elapsed_time > 0 else 0

    print("\n--- Recording Session Stats ---")
    print(f"Total time elapsed: {elapsed_time:.2f} seconds")
    print(f"Total frames saved: {frames_recorded}")
    print(f"True Capture FPS: {actual_fps:.2f} fps")
    print("-------------------------------\n")

    # Clean up thread, writer, and window
    eye_cam.stop()
    out.release()
    cv2.destroyAllWindows()
    print("Recording successfully saved and stream closed.")

if __name__ == "__main__":
    main()