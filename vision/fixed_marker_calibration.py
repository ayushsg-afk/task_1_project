
import cv2
import numpy as np
import pyrealsense2 as rs
import os
import time

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MARKER_SIZE = 0.037       # 37 mm in metres
MARKER_ID = 4
NUM_SAMPLES = 30

OUTPUT_DIR = "calibration/data"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR, "camera_to_marker.npz"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# ARUCO SETUP
# --------------------------------------------------

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(
    dictionary, parameters
)

# Object points in marker coordinate frame.
# Required order for SOLVEPNP_IPPE_SQUARE.

half = MARKER_SIZE / 2.0

object_points = np.array([
    [-half,  half, 0],
    [ half,  half, 0],
    [ half, -half, 0],
    [-half, -half, 0]
], dtype=np.float32)

# --------------------------------------------------
# REALSENSE SETUP
# --------------------------------------------------

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color,
    640, 480,
    rs.format.bgr8,
    30
)

samples_t = []
samples_R = []

try:

    profile = pipeline.start(config)

    color_profile = (
        profile.get_stream(rs.stream.color)
        .as_video_stream_profile()
    )

    intr = color_profile.get_intrinsics()

    camera_matrix = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion = np.array(
        intr.coeffs[:5], dtype=np.float64
    )

    print("\nCamera matrix:")
    print(camera_matrix)

    print("\nMarker calibration started.")
    print("Keep marker fixed, flat and fully visible.")
    print("Collecting 30 measurements...")

    while len(samples_t) < NUM_SAMPLES:

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        image = np.asanyarray(
            color_frame.get_data()
        )

        corners, ids, rejected = detector.detectMarkers(
            image
        )

        display = image.copy()

        if ids is not None:

            for i, marker_id in enumerate(ids.flatten()):

                if marker_id != MARKER_ID:
                    continue

                image_points = corners[i].reshape(
                    4, 2
                ).astype(np.float32)

                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    camera_matrix,
                    distortion,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if not success:
                    continue

                R, _ = cv2.Rodrigues(rvec)

                samples_t.append(tvec.flatten())
                samples_R.append(R)

                cv2.aruco.drawDetectedMarkers(
                    display, [corners[i]]
                )

                cv2.drawFrameAxes(
                    display,
                    camera_matrix,
                    distortion,
                    rvec,
                    tvec,
                    0.03
                )

                print(
                    "Sample:",
                    len(samples_t),
                    "/",
                    NUM_SAMPLES,
                    "Position (mm):",
                    np.round(tvec.flatten() * 1000, 2)
                )

                time.sleep(0.05)

        cv2.imshow(
            "Fixed Marker Calibration",
            display
        )

        if cv2.waitKey(1) & 0xFF == 27:
            print("Calibration interrupted.")
            break

    if len(samples_t) == NUM_SAMPLES:

        # Median translation reduces outlier influence.
        mean_t = np.median(
            np.array(samples_t), axis=0
        )

        # Average rotations using SVD.
        R_sum = np.sum(
            np.array(samples_R), axis=0
        )

        U, S, Vt = np.linalg.svd(R_sum)
        mean_R = U @ Vt

        if np.linalg.det(mean_R) < 0:
            U[:, -1] *= -1
            mean_R = U @ Vt

        # Camera_T_Marker:
        # Converts marker-frame points into camera-frame points.

        T_camera_marker = np.eye(4)

        T_camera_marker[:3, :3] = mean_R
        T_camera_marker[:3, 3] = mean_t

        print("\nFinal marker position relative to camera:")
        print("X:", mean_t[0] * 1000, "mm")
        print("Y:", mean_t[1] * 1000, "mm")
        print("Z:", mean_t[2] * 1000, "mm")

        print("\nCamera_T_Marker:")
        print(T_camera_marker)

        np.savez(
            OUTPUT_FILE,
            T_camera_marker=T_camera_marker,
            camera_matrix=camera_matrix,
            distortion=distortion,
            marker_size=MARKER_SIZE,
            marker_id=MARKER_ID
        )

        print("\nCalibration saved successfully:")
        print(OUTPUT_FILE)

    else:
        print(
            "\nNot enough measurements collected:",
            len(samples_t)
        )

finally:

    pipeline.stop()
    cv2.destroyAllWindows()

    print("\nCamera stopped.")
