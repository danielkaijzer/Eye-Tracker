
import cv2
import numpy as np

def main():
    # --- Configuration ---
    # CAMERA_MODE = 'LOW_RES_120HZ' 
    CAMERA_MODE = 'HIGH_RES_30HZ'
    FLIP_IR_CAMERA = True  # Set to True to flip the IR camera feed upside down
    
    # The front facing (OV5640) defaults to 0 and eye tracker cam (GC0308) to 1
    camera_index_1 = 0
    camera_index_2 = 1 # IR camera feed
    
    cap1 = cv2.VideoCapture(camera_index_1)
    cap2 = cv2.VideoCapture(camera_index_2) # IR camera feed

    if not cap1.isOpened():
        print(f"Error: Could not open camera with index {camera_index_1}.")
        print("Please make sure your camera is connected and the correct index is used.")
        return

    if not cap2.isOpened():
        print(f"Error: Could not open camera with index {camera_index_2}.")
        print("Please make sure your camera is connected and the correct index is used.")
        return

    # --- Set Camera Properties based on Mode ---
    if CAMERA_MODE == 'HIGH_RES_30HZ':
        # Assuming high resolution is 640x480, a standard resolution for many cameras.
        # This might need adjustment based on the specific camera's capabilities.
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap2.set(cv2.CAP_PROP_FPS, 30)
    elif CAMERA_MODE == 'LOW_RES_120HZ':
        # A common low resolution that might support high frame rates.
        # This might need adjustment based on the specific camera's capabilities.
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap2.set(cv2.CAP_PROP_FPS, 120)

    while True:
        # Capture frame-by-frame
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read() # IR camera feed

        # if frame is read correctly ret is True
        if not ret1:
            print("Error: Can't receive frame from camera 1 (stream end?). Exiting ...")
            break
        
        if not ret2:
            print("Error: Can't receive frame from camera 2 (stream end?). Exiting ...")
            break

        # --- Flip IR Camera Feed if Enabled ---
        if FLIP_IR_CAMERA:
            frame2 = cv2.flip(frame2, 0) # 0 for vertical flip

        # --- Scale up low-res feed for better visibility ---
        if CAMERA_MODE == 'LOW_RES_120HZ':
            # Scale the IR feed to be larger for easier viewing
            frame2 = cv2.resize(frame2, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        # Resize frames to the same height if necessary
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2] # IR camera feed

        if h1 != h2:
            if h1 > h2:
                new_w1 = int(w1 * h2 / h1)
                frame1 = cv2.resize(frame1, (new_w1, h2), interpolation=cv2.INTER_AREA)
            else:
                new_w2 = int(w2 * h1 / h2)
                frame2 = cv2.resize(frame2, (new_w2, h1), interpolation=cv2.INTER_AREA)

        # Combine the frames side by side
        combined_frame = np.hstack((frame1, frame2))

        # Display the resulting frame
        cv2.imshow('GC0308 & OV5640 Camera Test', combined_frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
