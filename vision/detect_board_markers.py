
import cv2
import numpy as np
import pyrealsense2 as rs

# =====================================================
# 1. CAMERA-TO-BOARD CALIBRATION
# =====================================================

# Existing saved transformation matrix.
# Based on the calibration values previously obtained.

T_camera_board = np.array([
    [-0.03968687, -0.99609763,  0.07883184, -0.03551927],
    [-0.98504565,  0.02576442, -0.17035628,  0.09886492],
    [ 0.16766043, -0.08441387, -0.98222415,  0.47676925],
    [ 0.0,         0.0,         0.0,         1.0]
])

# Inverse transformation:
# Camera coordinates -> Board coordinates

T_board_camera = np.linalg.inv(T_camera_board)


# =====================================================
# 2. ARUCO CONFIGURATION
# =====================================================

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(
    dictionary,
    parameters
)


# =====================================================
# 3. REALSENSE CAMERA CONFIGURATION
# =====================================================

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30
)

config.enable_stream(
    rs.stream.depth,
    640,
    480,
    rs.format.z16,
    30
)

profile = pipeline.start(config)

# Align depth to RGB image
align = rs.align(rs.stream.color)

print("\n===================================")
print(" REALSENSE BOARD MARKER DETECTION")
print("===================================")
print("Camera started successfully.")
print("Detecting four board markers...")
print("Press Q to quit.")


# =====================================================
# 4. PIXEL TO 3D CAMERA COORDINATES
# =====================================================

def get_camera_coordinates(
    depth_frame,
    intrinsics,
    px,
    py
):

    depth_values = []

    # Collect nearby valid depth measurements
    for dy in range(-3, 4):

        for dx in range(-3, 4):

            x = px + dx
            y = py + dy

            if x < 0 or y < 0:
                continue

            if x >= intrinsics.width or y >= intrinsics.height:
                continue

            depth = depth_frame.get_distance(x, y)

            if depth > 0:
                depth_values.append(depth)

    if len(depth_values) == 0:
        return None

    # Median depth reduces noise
    depth = float(np.median(depth_values))

    point_camera = rs.rs2_deproject_pixel_to_point(
        intrinsics,
        [float(px), float(py)],
        depth
    )

    return np.array(point_camera)


# =====================================================
# 5. CAMERA COORDINATES TO BOARD COORDINATES
# =====================================================

def camera_to_board(point_camera):

    point_camera_h = np.array([
        point_camera[0],
        point_camera[1],
        point_camera[2],
        1.0
    ])

    point_board = (
        T_board_camera @ point_camera_h
    )

    return point_board[:3]


# =====================================================
# 6. MAIN CAMERA LOOP
# =====================================================

try:

    while True:

        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        image = np.asanyarray(
            color_frame.get_data()
        )

        # Get actual RGB camera intrinsics
        intrinsics = (
            color_frame.profile
            .as_video_stream_profile()
            .intrinsics
        )

        # Detect ArUco markers
        corners, ids, rejected = detector.detectMarkers(
            image
        )

        if ids is not None:

            cv2.aruco.drawDetectedMarkers(
                image,
                corners,
                ids
            )

            print("\n========== BOARD MARKERS ==========")

            detected_count = 0

            for i, marker_id in enumerate(ids.flatten()):

                # Marker centre in pixel coordinates
                marker_corners = corners[i][0]

                center = np.mean(
                    marker_corners,
                    axis=0
                )

                px = int(round(center[0]))
                py = int(round(center[1]))

                # Get 3D camera coordinates
                point_camera = get_camera_coordinates(
                    depth_frame,
                    intrinsics,
                    px,
                    py
                )

                if point_camera is None:

                    print(
                        "\nMarker ID:",
                        int(marker_id),
                        "- Invalid depth"
                    )

                    continue

                # Convert to board coordinates
                point_board = camera_to_board(
                    point_camera
                )

                # Convert metres to millimetres
                camera_mm = point_camera * 1000
                board_mm = point_board * 1000

                detected_count += 1

                print("\nMarker ID:", int(marker_id))

                print(
                    "Pixel centre:",
                    px,
                    py
                )

                print(
                    "Camera XYZ (mm):",
                    np.round(camera_mm, 2)
                )

                print(
                    "Board XYZ (mm):",
                    np.round(board_mm, 2)
                )

                # Draw marker centre
                cv2.circle(
                    image,
                    (px, py),
                    5,
                    (0, 0, 255),
                    -1
                )

                # Display ID
                cv2.putText(
                    image,
                    "ID " + str(marker_id),
                    (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                # Display board coordinates
                label = (
                    "X:" + str(round(board_mm[0], 1)) +
                    " Y:" + str(round(board_mm[1], 1))
                )

                cv2.putText(
                    image,
                    label,
                    (px + 8, py + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 0),
                    1
                )

            print("\nTotal valid markers:", detected_count)

        else:

            cv2.putText(
                image,
                "No ArUco markers detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "Four Board Marker Detection",
            image
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# =====================================================
# 7. CLEANUP
# =====================================================

finally:

    pipeline.stop()

    cv2.destroyAllWindows()

    print("\nRealSense camera stopped.")
