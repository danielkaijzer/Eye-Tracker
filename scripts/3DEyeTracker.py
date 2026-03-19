import cv2
import random
import math
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, filedialog
import sys
import time

gaze_buffer = [] # Moving average buffer
screen_buffer = []
BUFFER_SIZE = 5  # Small enough to be responsive, large enough to stop jitter

calib_points_screen = []  # [(x,y), ...] in screen coords
calib_vectors_eye = []    # [np.array([dx, dy, dz]), ...]
calib_state = 0           # 0: Idle, >0: current target index (1-based)
calib_total_points = 12   # 3x4 grid

# Multi-sample collection state
calib_collecting = False
calib_collect_frames = []
CALIB_SAMPLES = 15        # frames to collect per point
CALIB_STD_THRESH = 0.015  # max std dev for a valid capture

# Polynomial calibration coefficients (6 per axis)
poly_coeffs_x = None
poly_coeffs_y = None

try:
    import gl_sphere
    GL_SPHERE_AVAILABLE = False # TODO: FIX GL_SPHERE LOGIC
except ImportError:
    GL_SPHERE_AVAILABLE = False
    print("gl_sphere module not found. OpenGL rendering will be disabled.")

# Toggle for 120Hz resolution/blurring logic
HIGH_FPS_MODE = False

ray_lines = [] 
model_centers = []
max_rays = 100
prev_model_center_avg = (320,240)
max_observed_distance = 0  

# --- Gaze → external camera projection globals ---
last_sphere_center = None
last_gaze_dir = None

calibrated = False
R_gaze_to_cam = np.eye(3, dtype=np.float32)  
calibrated_sphere_center = None 

sphere_center_locked_2d = False
locked_model_center_avg = prev_model_center_avg

# External camera / screen params (for 640x480)
EXT_WIDTH = 640
EXT_HEIGHT = 480
EXT_CX = EXT_WIDTH // 2
EXT_CY = EXT_HEIGHT // 2

circle_x = EXT_CX
circle_y = EXT_CY

# Approximate focal length in pixels (simple pinhole model)
EXT_FX = 950.0
EXT_FY = 950.0

# Function to detect available cameras 
def detect_cameras(max_cams=10):
    available_cameras = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def crop_to_aspect_ratio(image, width=640, height=480):
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height

    if current_ratio > desired_ratio:
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset + new_width]
    else:
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset + new_height, :]

    return cv2.resize(cropped_img, (width, height))

def apply_binary_threshold(image, darkestPixelValue, addedThreshold):
    threshold = int(darkestPixelValue) + addedThreshold
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    return thresholded_image

def get_darkest_area(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray, (20, 20))
    margin = 20
    roi = blurred[margin:-margin, margin:-margin]
    min_loc = cv2.minMaxLoc(roi)[3]
    return (min_loc[0] + margin, min_loc[1] + margin)

def mask_outside_square(image, center, size):
    x, y = center
    half_size = size // 2
    mask = np.zeros_like(image)
    top_left_x = max(0, x - half_size)
    top_left_y = max(0, y - half_size)
    bottom_right_x = min(image.shape[1], x + half_size)
    bottom_right_y = min(image.shape[0], y + half_size)
    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255
    return cv2.bitwise_and(image, mask)

def optimize_contours_by_angle(contours, image):
    if len(contours) < 1:
        return contours

    all_contours = np.concatenate(contours[0], axis=0)
    spacing = int(len(all_contours)/25) 
    filtered_points = []
    centroid = np.mean(all_contours, axis=0)
    
    for i in range(0, len(all_contours), 1):
        current_point = all_contours[i]
        prev_point = all_contours[i - spacing] if i - spacing >= 0 else all_contours[-spacing]
        next_point = all_contours[i + spacing] if i + spacing < len(all_contours) else all_contours[spacing]
        
        vec1 = prev_point - current_point
        vec2 = next_point - current_point
        
        with np.errstate(invalid='ignore'):
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        
        vec_to_centroid = centroid - current_point
        cos_threshold = np.cos(np.radians(60)) 
        
        if np.dot(vec_to_centroid, (vec1+vec2)/2) >= cos_threshold:
            filtered_points.append(current_point)
    
    return np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))

def filter_contours_by_area_and_return_largest(contours, pixel_thresh, ratio_thresh):
    max_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= pixel_thresh:
            x, y, w, h = cv2.boundingRect(contour)
            length_to_width_ratio = max(w / h, h / w)
            if length_to_width_ratio <= ratio_thresh:
                if area > max_area:
                    max_area = area
                    largest_contour = contour
    return [largest_contour] if largest_contour is not None else []

def fit_and_draw_ellipses(image, optimized_contours, color):
    if len(optimized_contours) >= 5:
        contour = np.array(optimized_contours, dtype=np.int32).reshape((-1, 1, 2))
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(image, ellipse, color, 2) 
        return image
    else:
        return image

def check_contour_pixels(contour, image_shape, debug_mode_on):
    if len(contour) < 5:
        return [0, 0, None] 
    
    contour_mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, (255), 1)
   
    ellipse_mask_thick = np.zeros(image_shape, dtype=np.uint8)
    ellipse_mask_thin = np.zeros(image_shape, dtype=np.uint8)
    ellipse = cv2.fitEllipse(contour)
    
    cv2.ellipse(ellipse_mask_thick, ellipse, (255), 10) 
    cv2.ellipse(ellipse_mask_thin, ellipse, (255), 4) 

    overlap_thick = cv2.bitwise_and(contour_mask, ellipse_mask_thick)
    overlap_thin = cv2.bitwise_and(contour_mask, ellipse_mask_thin)
    
    absolute_pixel_total_thick = np.sum(overlap_thick > 0)
    absolute_pixel_total_thin = np.sum(overlap_thin > 0)
    
    total_border_pixels = np.sum(contour_mask > 0)
    ratio_under_ellipse = absolute_pixel_total_thin / total_border_pixels if total_border_pixels > 0 else 0
    
    return [absolute_pixel_total_thick, ratio_under_ellipse, overlap_thin]

def check_ellipse_goodness(binary_image, contour, debug_mode_on):
    ellipse_goodness = [0,0,0] 
    if len(contour) < 5:
        return ellipse_goodness  
    
    ellipse = cv2.fitEllipse(contour)
    mask = np.zeros_like(binary_image)
    cv2.ellipse(mask, ellipse, (255), -1)
    
    ellipse_area = np.sum(mask == 255)
    covered_pixels = np.sum((binary_image == 255) & (mask == 255))
    
    if ellipse_area == 0:
        return ellipse_goodness 
    
    ellipse_goodness[0] = covered_pixels / ellipse_area
    axes_lengths = ellipse[1]  
    ellipse_goodness[2] = min(ellipse[1][1]/ellipse[1][0], ellipse[1][0]/ellipse[1][1])
    
    return ellipse_goodness

def process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, debug_mode_on, render_cv_window):
    global ray_lines, prev_model_center_avg, max_observed_distance

    final_rotated_rect = None
    image_array = [thresholded_image_relaxed, thresholded_image_medium, thresholded_image_strict] 
    name_array = ["relaxed", "medium", "strict"] 
    final_image = image_array[0] 
    final_contours = [] 
    ellipse_reduced_contours = [] 
    goodness = 0 
    
    # Dynamic Elliptical Kernel (Restored)
    k_size = int(gray_frame.shape[1] * 0.025)
    k_size = k_size if k_size % 2 != 0 else k_size + 1 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    
    gray_copy1 = gray_frame.copy()
    gray_copy2 = gray_frame.copy()
    gray_copy3 = gray_frame.copy()
    gray_copies = [gray_copy1, gray_copy2, gray_copy3]
    
    for i in range(1,4):
        # Morphological Closing (Restored)
        dilated_image = cv2.morphologyEx(image_array[i-1], cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        reduced_contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)

        if len(reduced_contours) > 0 and len(reduced_contours[0]) > 5:
            current_goodness = check_ellipse_goodness(dilated_image, reduced_contours[0], debug_mode_on)
            ellipse = cv2.fitEllipse(reduced_contours[0])
            
            if debug_mode_on: 
                cv2.imshow(name_array[i-1] + " threshold", gray_copies[i-1])
                
            total_pixels = check_contour_pixels(reduced_contours[0], dilated_image.shape, debug_mode_on)                 
            cv2.ellipse(gray_copies[i-1], ellipse, (255, 0, 0), 2)  
            
            final_goodness = current_goodness[0]*total_pixels[0]*total_pixels[0]*total_pixels[1]

            if final_goodness > 0 and final_goodness > goodness: 
                goodness = final_goodness
                ellipse_reduced_contours = total_pixels[2]
                best_image = image_array[i-1]
                final_contours = reduced_contours
                final_image = dilated_image

    test_frame = frame.copy()
    final_contours = [optimize_contours_by_angle(final_contours, gray_frame)]
    
    center_x, center_y = None, None

    # Fixed syntax error that crashed the previous script
    if final_contours and not isinstance(final_contours[0], list) and len(final_contours[0]) > 5:
        ellipse = cv2.fitEllipse(final_contours[0])
        final_rotated_rect = ellipse
        center_x, center_y = map(int, ellipse[0]) 

        ray_lines.append(final_rotated_rect)
        if len(ray_lines) > max_rays:
            num_to_remove = len(ray_lines) - max_rays
            ray_lines = ray_lines[num_to_remove:]  

    global sphere_center_locked_2d, locked_model_center_avg, prev_model_center_avg
    model_center_average = (320,240)
    model_center = compute_average_intersection(frame, ray_lines, 5, 1500, 5)

    if not sphere_center_locked_2d:
        if model_center is not None:
            model_center_average = update_and_average_point(model_centers, model_center, 200)
        else:
            model_center_average = prev_model_center_avg

        if model_center_average is not None and model_center_average[0] != 0:
            prev_model_center_avg = model_center_average
            locked_model_center_avg = model_center_average
    else:
        model_center_average = locked_model_center_avg

    if center_x is None or center_y is None or model_center_average is None or model_center_average[0] is None:
        return final_rotated_rect  

    if len(model_centers) >= 100 and center_x is not None:
        distance = math.sqrt((center_x - model_center_average[0]) ** 2 + (center_y - model_center_average[1]) ** 2)
        if distance > max_observed_distance:
            max_observed_distance = distance
            
    max_observed_distance = 202

    cv2.circle(frame, model_center_average, int(max_observed_distance), (255, 50, 50), 2)  
    cv2.circle(frame, model_center_average, 8, (255, 255, 0), -1) 

    if final_rotated_rect is not None:
        cv2.line(frame, model_center_average, (center_x, center_y), (255, 150, 50), 2)  
        cv2.ellipse(frame, final_rotated_rect, (20, 255, 255), 2) 
        
        dx = center_x - model_center_average[0]
        dy = center_y - model_center_average[1]
        extended_x = int(model_center_average[0] + 2 * dx)
        extended_y = int(model_center_average[1] + 2 * dy)
        cv2.line(frame, (center_x, center_y), (extended_x, extended_y), (200, 255, 0), 3) 

    if render_cv_window:
        cv2.imshow("Best Thresholded Image Contours on Frame", frame)

    if GL_SPHERE_AVAILABLE:
        gl_image = gl_sphere.update_sphere_rotation(center_x, center_y, model_center_average[0], model_center_average[1])

    center, direction = compute_gaze_vector(center_x, center_y, model_center_average[0], model_center_average[1])

    if center is not None and direction is not None:
        origin_text = f"Origin: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})"
        dir_text    = f"Direction: ({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f})"

        text_origin = (12, frame.shape[0] - 38)  
        text_dir    = (12, frame.shape[0] - 13)  
        text_origin2 = (10, frame.shape[0] - 40)  
        text_dir2    = (10, frame.shape[0] - 15)  

        cv2.putText(frame, origin_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, dir_text, text_dir, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, origin_text, text_origin2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, dir_text, text_dir2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Frame with Ellipse and Rays", frame)

    if GL_SPHERE_AVAILABLE and 'gl_image' in locals() and gl_image is not None:
        blended = cv2.addWeighted(frame, 0.6, gl_image, 0.4, 0)
        cv2.imshow("Eye Tracker + Sphere", blended)

    return final_rotated_rect

def update_and_average_point(point_list, new_point, N):
    point_list.append(new_point)  
    if len(point_list) > N:
        point_list.pop(0)  
    if not point_list:
        return None  
    avg_x = int(np.mean([p[0] for p in point_list]))
    avg_y = int(np.mean([p[1] for p in point_list]))
    return (avg_x, avg_y)

def draw_orthogonal_ray(image, ellipse, length=100, color=(0, 255, 0), thickness=1):
    (cx, cy), (major_axis, minor_axis), angle = ellipse
    angle_rad = np.deg2rad(angle)
    normal_dx = (minor_axis / 2) * np.cos(angle_rad)  
    normal_dy = (minor_axis / 2) * np.sin(angle_rad)
    pt1 = (int(cx - length * normal_dx / (minor_axis / 2)), int(cy - length * normal_dy / (minor_axis / 2)))
    pt2 = (int(cx + length * normal_dx / (minor_axis / 2)), int(cy + length * normal_dy / (minor_axis / 2)))
    cv2.line(image, pt1, pt2, color, thickness)
    return image 

stored_intersections = []  

def compute_average_intersection(frame, ray_lines, N, M, spacing):
    global stored_intersections
    if len(ray_lines) < 2 or N < 2:
        return (0, 0)  

    height, width = frame.shape[:2]
    selected_lines = random.sample(ray_lines, min(N, len(ray_lines)))
    intersections = []

    for i in range(len(selected_lines) - 1):
        line1 = selected_lines[i]
        line2 = selected_lines[i + 1]
        angle1 = line1[2]  
        angle2 = line2[2]  

        if abs(angle1 - angle2) >= 2:  
            intersection = find_line_intersection(line1, line2)
            if intersection and (0 <= intersection[0] < width) and (0 <= intersection[1] < height):
                intersections.append(intersection)
                stored_intersections.append(intersection)  

    if len(stored_intersections) > M:
        stored_intersections = prune_intersections(stored_intersections, M)

    if not intersections:
        return None  

    avg_x = np.mean([pt[0] for pt in stored_intersections])
    avg_y = np.mean([pt[1] for pt in stored_intersections])
    return (int(avg_x), int(avg_y))

def prune_intersections(intersections, maximum_intersections):
    if len(intersections) <= maximum_intersections:
        return intersections  
    pruned_intersections = intersections[-maximum_intersections:]
    return pruned_intersections

def rotation_from_a_to_b(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)

    if np.linalg.norm(v) < 1e-6:
        if c > 0:
            return np.eye(3, dtype=np.float32)
        else:
            axis = np.array([1.0, 0.0, 0.0])
            if abs(a[0]) > 0.9:
                axis = np.array([0.0, 1.0, 0.0])
            v = np.cross(a, axis)
            v = v / np.linalg.norm(v)
            s = np.linalg.norm(v)
    else:
        s = np.linalg.norm(v)
        v = v / s

    vx, vy, vz = v
    K = np.array([
        [0,    -vz,  vy],
        [vz,    0,  -vx],
        [-vy,  vx,   0 ]
    ], dtype=np.float32)

    R = np.eye(3, dtype=np.float32) + K * s + (K @ K) * ((1 - c) / (s ** 2))
    return R

def update_gaze_circle_from_current_gaze():
    global circle_x, circle_y, last_gaze_dir, calibrated, screen_buffer
    if not calibrated or last_gaze_dir is None:
        return
    if poly_coeffs_x is None or poly_coeffs_y is None:
        return

    g = last_gaze_dir
    if abs(g[2]) < 1e-3:
        return

    gx = g[0] / g[2]
    gy = g[1] / g[2]
    feat = _build_poly_features(gx, gy)

    u = feat @ poly_coeffs_x
    v = feat @ poly_coeffs_y

    screen_buffer.append((u, v))
    if len(screen_buffer) > BUFFER_SIZE:
        screen_buffer.pop(0)

    avg_u = np.mean([p[0] for p in screen_buffer])
    avg_v = np.mean([p[1] for p in screen_buffer])

    circle_x = int(np.clip(avg_u, 0, EXT_WIDTH - 1))
    circle_y = int(np.clip(avg_v, 0, EXT_HEIGHT - 1))

def find_line_intersection(ellipse1, ellipse2):
    (cx1, cy1), (_, minor_axis1), angle1 = ellipse1
    (cx2, cy2), (_, minor_axis2), angle2 = ellipse2
    angle1_rad = np.deg2rad(angle1)
    angle2_rad = np.deg2rad(angle2)

    dx1, dy1 = (minor_axis1 / 2) * np.cos(angle1_rad), (minor_axis1 / 2) * np.sin(angle1_rad)
    dx2, dy2 = (minor_axis2 / 2) * np.cos(angle2_rad), (minor_axis2 / 2) * np.sin(angle2_rad)

    A = np.array([[dx1, -dx2], [dy1, -dy2]])
    B = np.array([cx2 - cx1, cy2 - cy1])

    if np.linalg.det(A) == 0:
        return None  

    t1, t2 = np.linalg.solve(A, B)
    intersection_x = cx1 + t1 * dx1
    intersection_y = cy1 + t1 * dy1
    return (int(intersection_x), int(intersection_y))

def compute_gaze_vector(x, y, center_x, center_y, screen_width=640, screen_height=480):
    viewport_width = screen_width
    viewport_height = screen_height
    fov_y_deg = 45.0
    aspect_ratio = viewport_width / viewport_height
    far_clip = 100.0

    camera_position = np.array([0.0, 0.0, 3.0])
    fov_y_rad = np.radians(fov_y_deg)
    half_height_far = np.tan(fov_y_rad / 2) * far_clip
    half_width_far = half_height_far * aspect_ratio

    ndc_x = (2.0 * x) / viewport_width - 1.0
    ndc_y = 1.0 - (2.0 * y) / viewport_height

    far_x = ndc_x * half_width_far
    far_y = ndc_y * half_height_far
    far_z = camera_position[2] - far_clip
    far_point = np.array([far_x, far_y, far_z])

    ray_origin = camera_position
    ray_direction = far_point - camera_position
    ray_direction /= np.linalg.norm(ray_direction)
    ray_direction = -ray_direction

    inner_radius = 1.0 / 1.05
    sphere_offset_x = (center_x / screen_width) * 2.0 - 1.0
    sphere_offset_y = 1.0 - (center_y / screen_height) * 2.0

    # # TODO: For GL SPHERE
    # fov_y_rad = np.radians(45.0)
    # distance = 3.0
    # half_height = np.tan(fov_y_rad / 2.0) * distance
    # aspect_ratio = screen_width / max(1, screen_height)
    # half_width = half_height * aspect_ratio

    sphere_center = np.array([sphere_offset_x * 1.5, sphere_offset_y * 1.5, 0.0])

    origin = ray_origin
    direction = -ray_direction
    L = origin - sphere_center

    a = np.dot(direction, direction)
    b = 2 * np.dot(direction, L)
    c = np.dot(L, L) - inner_radius**2

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        t = -np.dot(direction, L) / np.dot(direction, direction)
        intersection_point = origin + t * direction
        intersection_local = intersection_point - sphere_center
        target_direction = intersection_local / np.linalg.norm(intersection_local)
    else:
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        t = None
        if t1 > 0 and t2 > 0:
            t = min(t1, t2)
        elif t1 > 0:
            t = t1
        elif t2 > 0:
            t = t2
        if t is None:
            return None, None

    intersection_point = origin + t * direction
    intersection_local = intersection_point - sphere_center
    target_direction = intersection_local / np.linalg.norm(intersection_local)

    circle_local_center = np.array([0.0, 0.0, inner_radius])
    circle_local_center /= np.linalg.norm(circle_local_center)

    rotation_axis = np.cross(circle_local_center, target_direction)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    if rotation_axis_norm < 1e-6:
        return sphere_center, circle_local_center

    rotation_axis /= rotation_axis_norm
    dot = np.dot(circle_local_center, target_direction)
    dot = np.clip(dot, -1.0, 1.0)
    angle_rad = np.arccos(dot)

    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    t_ = 1 - c
    x_, y_, z_ = rotation_axis

    rotation_matrix = np.array([
        [t_*x_*x_ + c, t_*x_*y_ - s*z_, t_*x_*z_ + s*y_],
        [t_*x_*y_ + s*z_, t_*y_*y_ + c, t_*y_*z_ - s*x_],
        [t_*x_*z_ - s*y_, t_*y_*z_ + s*x_, t_*z_*z_ + c]
    ])

    gaze_local = np.array([0.0, 0.0, inner_radius])
    gaze_rotated = rotation_matrix @ gaze_local
    gaze_rotated /= np.linalg.norm(gaze_rotated)

    global last_sphere_center, last_gaze_dir, calibrated_sphere_center
    last_sphere_center = sphere_center.copy()
    last_gaze_dir = gaze_rotated.copy()

    if calibrated_sphere_center is not None:
        sphere_center_out = calibrated_sphere_center
    else:
        sphere_center_out = sphere_center

    return sphere_center_out, gaze_rotated

def on_mouse_frame_with_rays(event, x, y, flags, param):
    global sphere_center_locked_2d, locked_model_center_avg, prev_model_center_avg
    global calibrated_sphere_center, calibrated, last_sphere_center

    if event == cv2.EVENT_LBUTTONDOWN:
        locked_model_center_avg = (x, y)
        prev_model_center_avg = locked_model_center_avg
        sphere_center_locked_2d = True
        if last_sphere_center is not None:
            calibrated_sphere_center = last_sphere_center.copy()
            calibrated = True

# def calibrate_gaze_to_external():
#     global calibrated, R_gaze_to_cam, calibrated_sphere_center
#     global sphere_center_locked_2d, locked_model_center_avg, prev_model_center_avg
#     if last_gaze_dir is None or last_sphere_center is None:
#         return

#     forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
#     R_gaze_to_cam = rotation_from_a_to_b(last_gaze_dir, forward)
#     calibrated_sphere_center = last_sphere_center.copy()
#     sphere_center_locked_2d = True
#     locked_model_center_avg = prev_model_center_avg
#     calibrated = True

def _build_poly_features(gx, gy):
    """Build 2nd-degree polynomial feature row: [1, gx, gy, gx^2, gy^2, gx*gy]"""
    return np.array([1.0, gx, gy, gx*gx, gy*gy, gx*gy])

def compute_polynomial_calibration():
    """Fit 2nd-degree polynomial from gaze vectors to screen coords (least-squares)."""
    global poly_coeffs_x, poly_coeffs_y
    n = len(calib_vectors_eye)
    if n < 6:
        print(f"Need at least 6 calibration points, have {n}.")
        return False

    A = np.zeros((n, 6))
    bx = np.zeros(n)
    by = np.zeros(n)
    for i, (v, pt) in enumerate(zip(calib_vectors_eye, calib_points_screen)):
        if abs(v[2]) < 1e-6:
            continue
        gx = v[0] / v[2]
        gy = v[1] / v[2]
        A[i] = _build_poly_features(gx, gy)
        bx[i] = pt[0]
        by[i] = pt[1]

    cx, res_x, _, _ = np.linalg.lstsq(A, bx, rcond=None)
    cy, res_y, _, _ = np.linalg.lstsq(A, by, rcond=None)
    poly_coeffs_x = cx
    poly_coeffs_y = cy

    # Leave-one-out cross-validation
    errors = []
    for i in range(n):
        A_loo = np.delete(A, i, axis=0)
        bx_loo = np.delete(bx, i)
        by_loo = np.delete(by, i)
        cx_loo, _, _, _ = np.linalg.lstsq(A_loo, bx_loo, rcond=None)
        cy_loo, _, _, _ = np.linalg.lstsq(A_loo, by_loo, rcond=None)
        pred_x = A[i] @ cx_loo
        pred_y = A[i] @ cy_loo
        err = math.sqrt((pred_x - bx[i])**2 + (pred_y - by[i])**2)
        errors.append(err)

    avg_err = np.mean(errors)
    max_err = np.max(errors)
    print(f"Polynomial calibration fitted ({n} points).")
    print(f"  LOO error: avg={avg_err:.1f}px, max={max_err:.1f}px")
    if avg_err > 40:
        print("  WARNING: High error — consider recalibrating.")

    _save_calibration()
    return True

def _calibration_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_data.npz")

def _history_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_history.npz")

def _save_calibration():
    """Save polynomial coefficients and raw calibration data."""
    global poly_coeffs_x, poly_coeffs_y
    path = _calibration_path()
    vectors = np.array(calib_vectors_eye)
    points = np.array(calib_points_screen)
    np.savez(path,
             poly_coeffs_x=poly_coeffs_x,
             poly_coeffs_y=poly_coeffs_y,
             vectors=vectors,
             points=points,
             timestamp=time.time())
    print(f"  Calibration saved to {path}")

    # Append to history for future ML training
    hist_path = _history_path()
    if os.path.exists(hist_path):
        old = np.load(hist_path, allow_pickle=True)
        old_vectors = list(old['all_vectors'])
        old_points = list(old['all_points'])
    else:
        old_vectors = []
        old_points = []
    old_vectors.append(vectors)
    old_points.append(points)
    np.savez(hist_path,
             all_vectors=np.array(old_vectors, dtype=object),
             all_points=np.array(old_points, dtype=object))
    total_pts = sum(len(v) for v in old_vectors)
    print(f"  History: {len(old_vectors)} sessions, {total_pts} total points.")

def load_calibration():
    """Load saved polynomial calibration. Returns True on success."""
    global poly_coeffs_x, poly_coeffs_y, calibrated
    path = _calibration_path()
    if not os.path.exists(path):
        print("No saved calibration found.")
        return False
    data = np.load(path, allow_pickle=True)
    poly_coeffs_x = data['poly_coeffs_x']
    poly_coeffs_y = data['poly_coeffs_y']
    ts = float(data['timestamp'])
    age_hrs = (time.time() - ts) / 3600
    calibrated = True
    print(f"Calibration loaded (age: {age_hrs:.1f}h).")
    if age_hrs > 24:
        print("  WARNING: Calibration is >24h old. Consider recalibrating.")
    return True

def start_calibration():
    global calib_state, calib_points_screen, calib_vectors_eye, calib_collecting, calib_collect_frames
    global calib_total_points
    calib_state = 1
    calib_vectors_eye = []
    calib_collecting = False
    calib_collect_frames = []

    # Generate 3x4 grid of calibration targets with margins
    margin_x = 40
    margin_y = 40
    cols, rows = 4, 3
    calib_points_screen = []
    for r in range(rows):
        for col in range(cols):
            x = int(margin_x + col * (EXT_WIDTH - 2 * margin_x) / (cols - 1))
            y = int(margin_y + r * (EXT_HEIGHT - 2 * margin_y) / (rows - 1))
            calib_points_screen.append((x, y))
    calib_total_points = len(calib_points_screen)
    print(f"Calibration started ({calib_total_points} points). Look at the RED dot and press 'c'.")

def begin_capture():
    """Start multi-sample collection for the current calibration point."""
    global calib_collecting, calib_collect_frames
    calib_collecting = True
    calib_collect_frames = []

def tick_capture():
    """Called each frame during collection. Returns True when done collecting."""
    global calib_collecting, calib_collect_frames, calib_state, calibrated
    if not calib_collecting:
        return False
    if last_gaze_dir is None:
        return False

    calib_collect_frames.append(last_gaze_dir.copy())
    if len(calib_collect_frames) < CALIB_SAMPLES:
        return False

    # Collection complete — compute median and check quality
    samples = np.array(calib_collect_frames)
    std_dev = np.std(samples, axis=0)
    max_std = np.max(std_dev)

    if max_std > CALIB_STD_THRESH:
        print(f"  High variance (std={max_std:.4f}). Retrying — hold still and press 'c'.")
        calib_collecting = False
        calib_collect_frames = []
        return False

    median_vec = np.median(samples, axis=0)
    median_vec = median_vec / np.linalg.norm(median_vec)
    calib_vectors_eye.append(median_vec)
    calib_collecting = False
    calib_collect_frames = []

    if calib_state >= calib_total_points:
        if compute_polynomial_calibration():
            calibrated = True
            print("Calibration Complete!")
        calib_state = 0
    else:
        calib_state += 1
        print(f"  Captured {len(calib_vectors_eye)}/{calib_total_points}. Look at next dot, press 'c'.")
    return True

def process_frame(frame):
    frame = crop_to_aspect_ratio(frame)
    darkest_point = get_darkest_area(frame)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if HIGH_FPS_MODE:
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        
    darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
    
    thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, 5)
    thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, 250)
    thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, 15)
    thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, 250)
    thresholded_image_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, 25)
    thresholded_image_relaxed = mask_outside_square(thresholded_image_relaxed, darkest_point, 250)
    
    final_rotated_rect = process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, False, False)
    return final_rotated_rect

# Process video from the selected eye camera + external camera preview
def process_camera():
    global selected_camera, circle_x, circle_y, calibrated

    try:
        cam_index = int(selected_camera.get())
    except ValueError:
        print("No valid camera selected.")
        return

    eye_cap = cv2.VideoCapture(cam_index)
    if not eye_cap.isOpened():
        print(f"Error: Could not open eye camera at index {cam_index}.")
        return

    if HIGH_FPS_MODE:
        eye_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        eye_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        eye_cap.set(cv2.CAP_PROP_FPS, 120)

    eye_cap.set(cv2.CAP_PROP_EXPOSURE, -5)

    external_index = 1 if cam_index == 0 else 0
    external_cap = cv2.VideoCapture(external_index)

    if external_cap.isOpened():
        external_cap.set(cv2.CAP_PROP_FRAME_WIDTH, EXT_WIDTH)
        external_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, EXT_HEIGHT)
    else:
        external_cap = None

    circle_x, circle_y = EXT_CX, EXT_CY
    calibrated = False

    # cv2.namedWindow("Original Eye Frame")
    # cv2.moveWindow("Original Eye Frame", 50, 50)
    cv2.namedWindow("Frame with Ellipse and Rays")
    cv2.moveWindow("Frame with Ellipse and Rays", 50, 600)
    cv2.setMouseCallback("Frame with Ellipse and Rays", on_mouse_frame_with_rays)
    
    if external_cap is not None:
        cv2.namedWindow("External Camera (Gaze)")
        cv2.moveWindow("External Camera (Gaze)", 720, 50)

    print("Controls: 'c' = calibrate, 'l' = load calibration, 'q' = quit, space = pause")

    while True:
        ret_eye, eye_frame = eye_cap.read()
        if not ret_eye:
            break

        # eye_frame = cv2.flip(eye_frame, 0)
        cv2.imshow("Original Eye Frame", eye_frame)
        process_frame(eye_frame)

        # Tick multi-sample collection if active
        if calib_collecting:
            tick_capture()

        if external_cap is not None:
            ret_ext, ext_frame = external_cap.read()
            if ret_ext:
                ext_frame_resized = cv2.resize(ext_frame, (EXT_WIDTH, EXT_HEIGHT))

                if calib_state > 0:
                    target = calib_points_screen[calib_state - 1]
                    cv2.circle(ext_frame_resized, target, 15, (0, 0, 255), -1)
                    # Draw small dots for all targets
                    for i, pt in enumerate(calib_points_screen):
                        if i < len(calib_vectors_eye):
                            cv2.circle(ext_frame_resized, pt, 5, (0, 200, 0), -1)
                        elif i != calib_state - 1:
                            cv2.circle(ext_frame_resized, pt, 5, (100, 100, 100), -1)

                    status_text = f"Point {calib_state}/{calib_total_points}"
                    if calib_collecting:
                        progress = len(calib_collect_frames)
                        status_text += f" - collecting [{progress}/{CALIB_SAMPLES}]"
                        cv2.circle(ext_frame_resized, target, 20, (0, 165, 255), 3)
                    else:
                        status_text += " - press 'c'"
                    cv2.putText(ext_frame_resized, status_text,
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                elif calibrated:
                    update_gaze_circle_from_current_gaze()
                    cv2.circle(ext_frame_resized, (circle_x, circle_y), 8, (0, 255, 0), -1)

                cv2.imshow("External Camera (Gaze)", ext_frame_resized)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)
        elif key == ord('l'):
            load_calibration()
        elif key == ord('c'):
            if calib_state == 0:
                start_calibration()
            elif not calib_collecting:
                begin_capture()

    eye_cap.release()
    if external_cap is not None:
        external_cap.release()
    cv2.destroyAllWindows()

def process_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
    if not video_path:
        return  
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        process_frame(frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
    
def selection_gui():
    global selected_camera
    cameras = detect_cameras()

    root = tk.Tk()
    root.title("Select Input Source")
    
    root.eval('tk::PlaceWindow . center')
    root.attributes('-topmost', True)
    root.update()
    root.attributes('-topmost', False)

    tk.Label(root, text="Eye Tracker 3D", font=("Arial", 12, "bold")).pack(pady=10)
    tk.Label(root, text="Select Camera:").pack(pady=5)

    selected_camera = tk.StringVar()
    selected_camera.set(str(cameras[0]) if cameras else "No cameras found")

    camera_dropdown = ttk.Combobox(root, textvariable=selected_camera, values=[str(cam) for cam in cameras])
    camera_dropdown.pack(pady=5)

    tk.Button(root, text="Start Camera", command=lambda: [root.destroy(), process_camera()]).pack(pady=5)
    tk.Button(root, text="Browse Video", command=lambda: [root.destroy(), process_video()]).pack(pady=5)

    if GL_SPHERE_AVAILABLE:
        try:
            app = gl_sphere.start_gl_window() 
        except Exception:
            pass

    root.mainloop()

if __name__ == "__main__":
    selection_gui()