
import cv2
import numpy as np
import pyrealsense2 as rs

# -----------------------------------------
# Board configuration
# -----------------------------------------
MARKER_IDS = [0, 1, 2, 3]
MARKER_SIZE = 0.037  # metres

CALIBRATION_FILE = "calibration/data/camera_to_board_v2.npz"

# -----------------------------------------
# Load camera-to-board calibration
# -----------------------------------------
data = np.load(CALIBRATION_FILE)

T_camera_board = data["T_camera_board"]
camera_matrix = data["camera_matrix"]
distortion = data["distortion"]

R_camera_board = T_camera_board[:3, :3]
t_camera_board = T_camera_board[:3, 3].reshape(3, 1)

rvec_board, _ = cv2.Rodrigues(R_camera_board)

print("\nCamera-to-board transformation:")
print(T_camera_board)

print("\nBoard origin relative to camera:")
print(f"X = {t_camera_board[0, 0] * 1000:.2f} mm")
print(f"Y = {t_camera_board[1, 0] * 1000:.2f} mm")
print(f"Z = {t_camera_board[2, 0] * 1000:.2f} mm")

# -----------------------------------------
# ArUco detector
# -----------------------------------------
dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(
    dictionary, parameters
)

# -----------------------------------------
# Start RealSense camera
# -----------------------------------------
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30
)

try:
    pipeline.start(config)

    print("\nFour-marker board detection started.")
    print("Expected marker IDs: 0, 1, 2, 3")
    print("Red = +X, Green = +Y, Blue = +Z")
    print("Press ESC to exit.")

    while True:

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())
        output = image.copy()

        corners, ids, rejected = detector.detectMarkers(image)

        if ids is not None:

            detected_ids = ids.flatten().tolist()

            cv2.aruco.drawDetectedMarkers(
                output, corners, ids
            )

            # Display detected IDs
            for i, marker_id in enumerate(detected_ids):

                center = np.mean(
                    corners[i].reshape(4, 2),
                    axis=0
                ).astype(int)

                cv2.putText(
                    output,
                    f"ID: {marker_id}",
                    tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            # Display board axes only when all four
            # calibration markers are visible.
            if all(mid in detected_ids for mid in MARKER_IDS):

                cv2.drawFrameAxes(
                    output,
                    camera_matrix,
                    distortion,
                    rvec_board,
                    t_camera_board,
                    0.05,
                    3
                )

                cv2.putText(
                    output,
                    "BOARD DETECTED: IDs 0, 1, 2, 3",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    output,
                    "Waiting for all four board markers",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        else:

            cv2.putText(
                output,
                "No ArUco markers detected",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        cv2.imshow("Four-Marker Board Detection", output)

        if cv2.waitKey(1) & 0xFF == 27:
            break

except Exception as e:
    print("Error:", e)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Camera stopped.")
