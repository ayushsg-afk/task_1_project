
import cv2
import numpy as np
import pyrealsense2 as rs
import os
import json
from datetime import datetime


# ============================================================
# 1. SETTINGS
# ============================================================

CALIBRATION_FILE = "calibration/data/camera_to_board_verified.npz"

WIDTH = 640
HEIGHT = 480
FPS = 30

PEG_TOP_Z_MM = 100.0

MIN_CONTOUR_AREA = 120
MIN_DEPTH_SAMPLES = 10

MIN_HEIGHT_MM = 70.0
MAX_HEIGHT_MM = 110.0

OUTPUT_FOLDER = "data"
OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "peg_detections.json"
)

COLOURS = {
    "Orange": (
        np.array([2, 80, 70]),
        np.array([18, 255, 255])
    ),
    "Green": (
        np.array([36, 70, 50]),
        np.array([80, 255, 255])
    ),
    "Yellow": (
        np.array([18, 80, 70]),
        np.array([35, 255, 255])
    ),
    "Blue": (
        np.array([96, 220, 100]),
        np.array([110, 255, 255])
    )
}

SHAPES = {
    "Orange": "Hexagon",
    "Green": "Circle",
    "Yellow": "Triangle",
    "Blue": "Square"
}


# ============================================================
# 2. LOAD CALIBRATION
# ============================================================

def load_calibration():

    if not os.path.exists(CALIBRATION_FILE):
        raise FileNotFoundError(CALIBRATION_FILE)

    data = np.load(CALIBRATION_FILE)

    T_camera_board = data["T_camera_board"].copy()

    if T_camera_board.shape != (4, 4):
        raise ValueError("Invalid calibration matrix.")

    # Inverse transformation:
    # Camera coordinates -> Board coordinates
    T_board_camera = np.linalg.inv(T_camera_board)

    # Convert inverse translation from metres to mm.
    T_board_camera[:3, 3] *= 1000.0

    K = data["camera_matrix"]
    distortion = data["distortion"]

    print("\nCalibration loaded successfully.")

    return T_board_camera, K, distortion


# ============================================================
# 3. START REALSENSE CAMERA
# ============================================================

def start_camera():

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        WIDTH,
        HEIGHT,
        rs.format.bgr8,
        FPS
    )

    config.enable_stream(
        rs.stream.depth,
        WIDTH,
        HEIGHT,
        rs.format.z16,
        FPS
    )

    pipeline.start(config)

    align = rs.align(rs.stream.color)

    print("\nRealSense camera started.")

    return pipeline, align


# ============================================================
# 4. PIXEL AND DEPTH TO BOARD COORDINATES
# ============================================================

def pixel_to_board(
    px,
    py,
    depth_frame,
    depth_intrinsics,
    T_board_camera
):

    depth_m = depth_frame.get_distance(
        int(px),
        int(py)
    )

    if depth_m <= 0:
        return None

    # Deproject pixel into camera coordinates.
    point_camera = rs.rs2_deproject_pixel_to_point(
        depth_intrinsics,
        [float(px), float(py)],
        depth_m
    )

    # Convert metres to millimetres.
    point_camera = np.array(
        point_camera,
        dtype=np.float64
    ) * 1000.0

    point_camera_h = np.append(
        point_camera,
        1.0
    )

    # Transform into board coordinates.
    point_board = T_board_camera @ point_camera_h

    return point_board[:3]


# ============================================================
# 5. PEG DETECTION
# ============================================================

def detect_pegs(
    image,
    depth_frame,
    depth_intrinsics,
    T_board_camera
):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    output = image.copy()

    detections = []

    for colour, (lower, upper) in COLOURS.items():

        mask = cv2.inRange(
            hsv,
            lower,
            upper
        )

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < MIN_CONTOUR_AREA:
                continue

            M = cv2.moments(contour)

            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # ------------------------------------------------
            # Collect depth samples in a 9x9 region.
            # ------------------------------------------------

            board_points = []
            depth_values = []

            for dy in range(-4, 5):

                for dx in range(-4, 5):

                    px = cx + dx
                    py = cy + dy

                    if not (
                        0 <= px < WIDTH and
                        0 <= py < HEIGHT
                    ):
                        continue

                    # Only use pixels inside the contour.
                    if cv2.pointPolygonTest(
                        contour,
                        (float(px), float(py)),
                        False
                    ) < 0:
                        continue

                    point = pixel_to_board(
                        px,
                        py,
                        depth_frame,
                        depth_intrinsics,
                        T_board_camera
                    )

                    if point is None:
                        continue

                    board_points.append(point)

                    depth_values.append(
                        depth_frame.get_distance(px, py) * 1000.0
                    )

            if len(board_points) < MIN_DEPTH_SAMPLES:
                continue

            # ------------------------------------------------
            # Calculate median board coordinates.
            # ------------------------------------------------

            board_points = np.array(
                board_points,
                dtype=np.float64
            )

            point_board = np.median(
                board_points,
                axis=0
            )

            x_mm = float(point_board[0])
            y_mm = float(point_board[1])
            raw_height_mm = float(point_board[2])

            # ------------------------------------------------
            # Reject detections outside expected height range.
            # ------------------------------------------------

            if (
                raw_height_mm < MIN_HEIGHT_MM or
                raw_height_mm > MAX_HEIGHT_MM
            ):
                continue

            height_mm = raw_height_mm

            median_depth = float(
                np.median(depth_values)
            )

            # ------------------------------------------------
            # Draw detected peg.
            # ------------------------------------------------

            x, y, w, h = cv2.boundingRect(contour)

            cv2.drawContours(
                output,
                [contour],
                -1,
                (0, 255, 0),
                2
            )

            cv2.circle(
                output,
                (cx, cy),
                5,
                (0, 0, 255),
                -1
            )

            label = (
                f"{colour}: "
                f"X={x_mm:.1f} "
                f"Y={y_mm:.1f} "
                f"Z={height_mm:.1f}"
            )

            cv2.putText(
                output,
                label,
                (x, max(y - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2
            )

            detections.append({

                "colour": colour,
                "shape": SHAPES[colour],
                "pixel": (cx, cy),
                "depth_mm": median_depth,
                "x_mm": x_mm,
                "y_mm": y_mm,
                "height_mm": height_mm,
                "depth_samples": len(board_points),
                "area": area

            })

    return output, detections


# ============================================================
# 6. SAVE DETECTED PEG COORDINATES TO JSON
# ============================================================

def save_detections(detections):

    if not detections:
        print("\nNo pegs detected. Nothing saved.")
        return

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    data = {

        "timestamp": datetime.now().isoformat(),

        "coordinate_frame": "board",

        "units": "mm",

        "board_axes": {
            "positive_x": "right",
            "positive_y": "up",
            "positive_z": "toward_camera"
        },

        "expected_peg_top_height_mm": PEG_TOP_Z_MM,

        "pegs": []

    }

    for peg in detections:

        peg_data = {

            "colour": peg["colour"],

            "shape": peg["shape"],

            "pixel": {
                "u": int(peg["pixel"][0]),
                "v": int(peg["pixel"][1])
            },

            "board_position_mm": {
                "x": round(peg["x_mm"], 2),
                "y": round(peg["y_mm"], 2),
                "z": round(peg["height_mm"], 2)
            },

            "depth_mm": round(
                peg["depth_mm"],
                2
            ),

            "depth_samples": peg["depth_samples"]

        }

        data["pegs"].append(peg_data)

    with open(
        OUTPUT_FILE,
        "w"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )

    print("\n================================")
    print("PEG COORDINATES SAVED")
    print("================================")

    print("File:", OUTPUT_FILE)

    print("Number of pegs:", len(detections))

    for peg in detections:

        print(
            f"{peg['colour']}: "
            f"X={peg['x_mm']:.2f} mm, "
            f"Y={peg['y_mm']:.2f} mm, "
            f"Z={peg['height_mm']:.2f} mm"
        )


# ============================================================
# 7. MAIN PROGRAM
# ============================================================

def main():

    T_board_camera, K, distortion = load_calibration()

    pipeline, align = start_camera()

    latest_detections = []

    try:

        print("\n================================")
        print("       PEG DETECTION V2")
        print("================================")

        print("S = Save peg coordinates")
        print("Q = Quit")

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

            # Aligned depth uses color-image coordinates.
            depth_intrinsics = (
                color_frame.profile
                .as_video_stream_profile()
                .get_intrinsics()
            )

            output, latest_detections = detect_pegs(
                image,
                depth_frame,
                depth_intrinsics,
                T_board_camera
            )

            cv2.imshow(
                "Peg Detection V2",
                output
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):

                save_detections(latest_detections)

            elif key == ord('q'):

                break

    finally:

        pipeline.stop()

        cv2.destroyAllWindows()


# ============================================================
# 8. EXECUTION
# ============================================================

if __name__ == "__main__":

    main()

