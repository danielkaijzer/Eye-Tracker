import cv2
import os

def analyze_video(video_path, real_world_seconds=None):
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    # 1. Get Metadata
    meta_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 2. Manually count frames to ensure accuracy
    print(f"Analyzing '{video_path}'... (this might take a second)")
    actual_frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        actual_frame_count += 1
        
    cap.release()

    # 3. Calculate metrics
    meta_duration = actual_frame_count / meta_fps if meta_fps > 0 else 0

    print("-" * 40)
    print(f"File: {video_path}")
    print(f"Metadata FPS: {meta_fps}")
    print(f"Total Frames Counted: {actual_frame_count}")
    print(f"Video Playback Duration: {meta_duration:.2f} seconds")
    print("-" * 40)

    # 4. Calculate True Effective FPS if real-world time is known
    if real_world_seconds:
        true_fps = actual_frame_count / real_world_seconds
        print(f"Real-World Duration: {real_world_seconds} seconds")
        print(f"True Effective FPS: {true_fps:.2f} frames/sec")
        
        if abs(true_fps - 30.0) > 2:
            print("⚠️ Warning: Effective FPS is lower than 30.")
            print("    This means OpenCV dropped frames while reading from the cameras.")
        else:
            print("✅ Success: Recording effectively captured at ~30 FPS!")
    else:
        print("To calculate TRUE FPS, provide the 'real_world_seconds' parameter.")
        print("Formula: True FPS = Total Frames Counted / Real-World Seconds")
    print("-" * 40)

if __name__ == "__main__":
    # --- Configuration ---
    # Replace these with the actual filenames from your recording session
    video_1 = "ir_cam_20260326_213600.mp4" 
    video_2 = "front_cam_20260326_213600.mp4"
    
    # IMPORTANT: How long did you actually let the script run? 
    # For example, if you ran the script for exactly 15 seconds, set this to 15.
    # If you don't know, leave it as None, and the script will just count your frames.
    known_recording_time_in_seconds = 10 
    
    analyze_video(video_1, real_world_seconds=known_recording_time_in_seconds)
    print("\n")
    analyze_video(video_2, real_world_seconds=known_recording_time_in_seconds)