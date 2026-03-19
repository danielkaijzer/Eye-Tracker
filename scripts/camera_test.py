
import cv2
import numpy as np


def _open_cameras(index1, index2):
    cap1 = cv2.VideoCapture(index1)
    cap2 = cv2.VideoCapture(index2)
    if not cap1.isOpened():
        print(f"Error: Could not open camera with index {index1}.")
        return None, None
    if not cap2.isOpened():
        print(f"Error: Could not open camera with index {index2}.")
        return None, None
    return cap1, cap2


def _configure_camera(cap2, mode):
    if mode == 'HIGH_RES_30HZ':
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap2.set(cv2.CAP_PROP_FPS, 30)
    elif mode == 'LOW_RES_120HZ':
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap2.set(cv2.CAP_PROP_FPS, 120)


def _read_frames(cap1, cap2):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1:
        print("Error: Can't receive frame from camera 1 (stream end?). Exiting ...")
        return None, None
    if not ret2:
        print("Error: Can't receive frame from camera 2 (stream end?). Exiting ...")
        return None, None
    return frame1, frame2


def _prepare_ir_frame(frame2, mode, flip):
    if flip:
        frame2 = cv2.flip(frame2, 0)
    if mode == 'LOW_RES_120HZ':
        frame2 = cv2.resize(frame2, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    return frame2


def _match_heights(frame1, frame2):
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    if h1 > h2:
        frame1 = cv2.resize(frame1, (int(w1 * h2 / h1), h2), interpolation=cv2.INTER_AREA)
    elif h2 > h1:
        frame2 = cv2.resize(frame2, (int(w2 * h1 / h2), h1), interpolation=cv2.INTER_AREA)
    return frame1, frame2


def main():
    CAMERA_MODE = 'HIGH_RES_30HZ'
    FLIP_IR_CAMERA = True
    camera_index_1 = 0
    camera_index_2 = 1

    cap1, cap2 = _open_cameras(camera_index_1, camera_index_2)
    if cap1 is None:
        return

    _configure_camera(cap2, CAMERA_MODE)

    while True:
        frame1, frame2 = _read_frames(cap1, cap2)
        if frame1 is None:
            break

        frame2 = _prepare_ir_frame(frame2, CAMERA_MODE, FLIP_IR_CAMERA)
        frame1, frame2 = _match_heights(frame1, frame2)

        cv2.imshow('GC0308 & OV5640 Camera Test', np.hstack((frame1, frame2)))

        if cv2.waitKey(1) == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

    # # --- Configuration ---
    # # CAMERA_MODE = 'LOW_RES_120HZ'
    # CAMERA_MODE = 'HIGH_RES_30HZ'
    # FLIP_IR_CAMERA = True  # Set to True to flip the IR camera feed upside down
    #
    # # The front facing (OV5640) defaults to 0 and eye tracker cam (GC0308) to 1
    # camera_index_1 = 0
    # camera_index_2 = 1 # IR camera feed
    #
    # cap1 = cv2.VideoCapture(camera_index_1)
    # cap2 = cv2.VideoCapture(camera_index_2) # IR camera feed
    #
    # if not cap1.isOpened():
    #     print(f"Error: Could not open camera with index {camera_index_1}.")
    #     print("Please make sure your camera is connected and the correct index is used.")
    #     return
    #
    # if not cap2.isOpened():
    #     print(f"Error: Could not open camera with index {camera_index_2}.")
    #     print("Please make sure your camera is connected and the correct index is used.")
    #     return
    #
    # # --- Set Camera Properties based on Mode ---
    # if CAMERA_MODE == 'HIGH_RES_30HZ':
    #     # Assuming high resolution is 640x480, a standard resolution for many cameras.
    #     # This might need adjustment based on the specific camera's capabilities.
    #     cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #     cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    #     cap2.set(cv2.CAP_PROP_FPS, 30)
    # elif CAMERA_MODE == 'LOW_RES_120HZ':
    #     # A common low resolution that might support high frame rates.
    #     # This might need adjustment based on the specific camera's capabilities.
    #     cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    #     cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    #     cap2.set(cv2.CAP_PROP_FPS, 120)
    #
    # while True:
    #     # Capture frame-by-frame
    #     ret1, frame1 = cap1.read()
    #     ret2, frame2 = cap2.read() # IR camera feed
    #
    #     # if frame is read correctly ret is True
    #     if not ret1:
    #         print("Error: Can't receive frame from camera 1 (stream end?). Exiting ...")
    #         break
    #
    #     if not ret2:
    #         print("Error: Can't receive frame from camera 2 (stream end?). Exiting ...")
    #         break
    #
    #     # --- Flip IR Camera Feed if Enabled ---
    #     if FLIP_IR_CAMERA:
    #         frame2 = cv2.flip(frame2, 0) # 0 for vertical flip
    #
    #     # --- Scale up low-res feed for better visibility ---
    #     if CAMERA_MODE == 'LOW_RES_120HZ':
    #         # Scale the IR feed to be larger for easier viewing
    #         frame2 = cv2.resize(frame2, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    #
    #     # Resize frames to the same height if necessary
    #     h1, w1 = frame1.shape[:2]
    #     h2, w2 = frame2.shape[:2] # IR camera feed
    #
    #     if h1 != h2:
    #         if h1 > h2:
    #             new_w1 = int(w1 * h2 / h1)
    #             frame1 = cv2.resize(frame1, (new_w1, h2), interpolation=cv2.INTER_AREA)
    #         else:
    #             new_w2 = int(w2 * h1 / h2)
    #             frame2 = cv2.resize(frame2, (new_w2, h1), interpolation=cv2.INTER_AREA)
    #
    #     # Combine the frames side by side
    #     combined_frame = np.hstack((frame1, frame2))
    #
    #     # Display the resulting frame
    #     cv2.imshow('GC0308 & OV5640 Camera Test', combined_frame)
    #
    #     # Exit if 'q' is pressed
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    #
    # cap1.release()
    # cap2.release()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
