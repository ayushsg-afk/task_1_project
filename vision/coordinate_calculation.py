
import pyrealsense2 as rs
import numpy as np
import cv2
import os

# ============================================================
# 1. LOAD CAMERA-TO-BOARD CALIBRATION
# ============================================================

CALIBRATION_FILE = "calibration/data/camera_to_board_v2.npz"

if not os.path.exists(CALIBRATION_FILE):
    raise FileNotFoundError(
        f"Calibration file not found: {CALIBRATION_FILE}"
    )

data = np.load(CALIBRATION_FILE)

T_camera_board = data["T_camera_board"]
camera_matrix = data["camera_matrix"]
distortion = data["distortion"]

# Inverse transformation: Camera -> Board
T_board_camera = np.linalg.inv(T_camera_board)

print("\nCamera-to-board calibration loaded.")
print("T_camera_board:")
print(T_camera_board)


# ============================================================
# 2. GLOBAL VARIABLES
# ============================================================

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color, 640, 480, rs.format.bgr8, 30
)

config.enable_stream(
    rs.stream.depth, 640, 480, rs.format.z16, 30
)

align = rs.align(rs.stream.color)

depth_frame = None
intrinsics = None

mouse_x = 0
mouse_y = 0
mouse_moved = False


# ============================================================
# 3. CAMERA COORDINATES -> BOARD COORDINATES
# IMPORTANT: Both input and transformation use METRES.
# ============================================================

def camera_to_board(point_camera_m):

    point_camera_h = np.array([
        point_camera_m[0],
        point_camera_m[1],
        point_camera_m[2],
        1.0
    ])

    point_board_h = T_board_camera @ point_camera_h

    return point_board_h[:3]


# ============================================================
# 4. MEASURE 3D POINT FROM PIXEL
# ============================================================

def measure_point(x, y):

    global depth_frame, intrinsics

    if depth_frame is None or intrinsics is None:
        print("Depth frame is not available.")
        return

    height = depth_frame.get_height()
    width = depth_frame.get_width()

    if not (0 <= x < width and 0 <= y < height):
        print("Selected pixel is outside the image.")
        return

    # Collect depth values from 5x5 neighbourhood
    depth_values = []

    for dx in range(-2, 3):
        for dy in range(-2, 3):

            px = x + dx
            py = y + dy

            if 0 <= px < width and 0 <= py < height:

                depth = depth_frame.get_distance(px, py)

                if depth > 0:
                    depth_values.append(depth)

    if len(depth_values) < 5:
        print("Insufficient valid depth pixels.")
        return

    # Median depth in metres
    z_m = float(np.median(depth_values))

    # Deproject pixel into camera coordinates (METRES)
    point_camera_m = np.array(
        rs.rs2_deproject_pixel_to_point(
            intrinsics,
            [x, y],
            z_m
        ),
        dtype=float
    )

    # Transform into board coordinates (METRES)
    point_board_m = camera_to_board(point_camera_m)

    # Convert to millimetres ONLY for display
    point_camera_mm = point_camera_m * 1000.0
    point_board_mm = point_board_m * 1000.0

    # Display results
    print("\n--------------------------------")
    print(f"PIXEL: {x} {y}")
    print(f"Valid depth pixels: {len(depth_values)}")
    print(f"Median depth: {z_m * 1000.0:.1f} mm")

    print("\nCamera coordinates:")
    print(f"X = {point_camera_mm[0]:.2f} mm")
    print(f"Y = {point_camera_mm[1]:.2f} mm")
    print(f"Z = {point_camera_mm[2]:.2f} mm")

    print("\nBoard coordinates:")
    print(f"X = {point_board_mm[0]:.2f} mm")
    print(f"Y = {point_board_mm[1]:.2f} mm")
    print(f"Z = {point_board_mm[2]:.2f} mm")

    print("--------------------------------")


# ============================================================
# 5. MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global mouse_x, mouse_y, mouse_moved

    if event == cv2.EVENT_MOUSEMOVE:

        mouse_x = x
        mouse_y = y
        mouse_moved = True

    elif event == cv2.EVENT_LBUTTONDOWN:

        print("\nSelected point:")
        measure_point(x, y)


# ============================================================
# 6. START CAMERA
# ============================================================

camera_started = False

try:

    pipeline.start(config)
    camera_started = True

    print("\nRealSense camera started.")
    print("Move mouse over RGB image to see pixel coordinates.")
    print("Left-click to measure 3D coordinates.")
    print("Press Q to exit.")

    cv2.namedWindow("RGB Image")
    cv2.namedWindow("Depth Image")

    cv2.setMouseCallback(
        "RGB Image",
        mouse_callback
    )

    # ========================================================
    # 7. MAIN CAMERA LOOP
    # ========================================================

    while True:

        frames = pipeline.wait_for_frames()

        # Align depth to RGB
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        current_depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not current_depth_frame:
            continue

        color_image = np.asanyarray(
            color_frame.get_data()
        )

        depth_frame = current_depth_frame

        # Intrinsics of aligned depth image
        intrinsics = (
            depth_frame.profile
            .as_video_stream_profile()
            .get_intrinsics()
        )

        # Copy RGB image for drawing
        display_image = color_image.copy()

        # Draw mouse crosshair and pixel coordinates
        if mouse_moved:

            cv2.drawMarker(
                display_image,
                (mouse_x, mouse_y),
                (0, 255, 0),
                cv2.MARKER_CROSS,
                15,
                2
            )

            cv2.putText(
                display_image,
                f"Pixel: ({mouse_x}, {mouse_y})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # Create depth visualization
        depth_image = np.asanyarray(
            depth_frame.get_data()
        )

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(
                depth_image,
                alpha=0.03
            ),
            cv2.COLORMAP_JET
        )

        # Display images
        cv2.imshow("RGB Image", display_image)
        cv2.imshow("Depth Image", depth_colormap)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

finally:

    if camera_started:
        pipeline.stop()

    cv2.destroyAllWindows()

    print("\nCamera stopped.")
