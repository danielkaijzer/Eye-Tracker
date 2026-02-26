
import cv2

def main():
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}.")
        print("Please make sure your camera is connected and the correct index is used.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # # Print the frame shape on the first frame
        # if 'frame_shape_printed' not in locals():
        #     print(f"Frame shape: {frame.shape}")
        #     frame_shape_printed = True

        # Display the resulting frame
        cv2.imshow('GC0308 Camera Test', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
