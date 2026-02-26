
import cv2
import numpy as np

def main():
    # The front facing (OV5640) defaults to 0 and eye tracker cam (GC0308) to 1
    camera_index_1 = 0
    camera_index_2 = 1
    
    cap1 = cv2.VideoCapture(camera_index_1)
    cap2 = cv2.VideoCapture(camera_index_2)

    if not cap1.isOpened():
        print(f"Error: Could not open camera with index {camera_index_1}.")
        print("Please make sure your camera is connected and the correct index is used.")
        return

    if not cap2.isOpened():
        print(f"Error: Could not open camera with index {camera_index_2}.")
        print("Please make sure your camera is connected and the correct index is used.")
        return

    while True:
        # Capture frame-by-frame
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # if frame is read correctly ret is True
        if not ret1:
            print("Error: Can't receive frame from camera 1 (stream end?). Exiting ...")
            break
        
        if not ret2:
            print("Error: Can't receive frame from camera 2 (stream end?). Exiting ...")
            break

        # Resize frames to the same height if necessary
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]

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
