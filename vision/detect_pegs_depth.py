
import pyrealsense2 as rs
import numpy as np
import cv2
import os

# ============================================================
# 1. SETTINGS
# ============================================================

CALIBRATION_FILE = "calibration/data/camera_to_board_verified.npz"

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FPS = 30

K = np.array([
    [603.22033691, 0, 321.35440063],
    [0, 602.73803711, 255.30134583],
    [0, 0, 1]
], dtype=np.float64)

PEG_TOP_Z_MM = 100.0

X_MIN = -63.0
X_MAX = 237.0
Y_MIN = -295.0
Y_MAX = 5.0

MIN_AREA = 150

EROSION_SIZE = 3
EROSION_ITERATIONS = 1

MIN_HEIGHT_MM = 95.0
MAX_HEIGHT_MM = 105.0
MIN_TOP_PIXELS = 100

COLOUR_RANGES = {
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

EXPECTED_SHAPES = {
    "Orange": "Hexagon",
    "Green": "Circle",
    "Yellow": "Triangle",
    "Blue": "Square"
}


# ============================================================
# 2. LOAD CALIBRATION
# ============================================================

def load_calibration(filename):

    if not os.path.exists(filename):
        raise FileNotFoundError(
            "Calibration file not found: " + filename
        )

    data = np.load(filename)

    print("\nCalibration keys:", data.files)

    if "T_camera_board" not in data.files:
        raise ValueError("T_camera_board not found.")

    T_camera_board = data["T_camera_board"].copy()

    if T_camera_board.shape != (4, 4):
        raise ValueError("Calibration matrix must be 4x4.")

    T_board_camera = np.linalg.inv(T_camera_board)

    # Convert translation from metres to millimetres.
    T_board_camera[:3, 3] *= 1000.0

    print("\nT_camera_board:")
    print(T_camera_board)

    print("\nT_board_camera (translation in mm):")
    print(T_board_camera)

    return T_camera_board, T_board_camera


# ============================================================
# 3. DEPTH FUNCTIONS
# ============================================================

def get_median_depth(depth_frame, cx, cy):

    x = int(round(cx))
    y = int(round(cy))

    depths = []

    for dy in range(-3, 4):
        for dx in range(-3, 4):

            px = x + dx
            py = y + dy

            if not (0 <= px < IMAGE_WIDTH and
                    0 <= py < IMAGE_HEIGHT):
                continue

            d = depth_frame.get_distance(px, py)

            if d > 0:
                depths.append(d)

    if len(depths) < 5:
        return None

    return float(np.median(depths))


def get_raw_center_depth(depth_frame, cx, cy):

    x = int(round(cx))
    y = int(round(cy))

    if not (0 <= x < IMAGE_WIDTH and
            0 <= y < IMAGE_HEIGHT):
        return None

    depth_m = depth_frame.get_distance(x, y)

    if depth_m <= 0:
        return None

    return depth_m * 1000.0


# ============================================================
# 4. PIXEL TO BOARD COORDINATES
# ============================================================

def calculate_xy_from_pixel(cx, cy, T_board_camera):

    pixel = np.array(
        [cx, cy, 1.0],
        dtype=np.float64
    )

    ray_camera = np.linalg.inv(K) @ pixel
    ray_camera /= np.linalg.norm(ray_camera)

    R = T_board_camera[:3, :3]
    t = T_board_camera[:3, 3]

    ray_board = R @ ray_camera

    if abs(ray_board[2]) < 1e-9:
        return None

    scale = (
        PEG_TOP_Z_MM - t[2]
    ) / ray_board[2]

    if scale <= 0:
        return None

    point_board = t + scale * ray_board

    return point_board


# ============================================================
# 5. TOP FACE DETECTION
# ============================================================

def get_top_face_center(
    contour,
    depth_frame,
    depth_intrinsics,
    T_board_camera
):

    contour_mask = np.zeros(
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        dtype=np.uint8
    )

    cv2.drawContours(
        contour_mask,
        [contour],
        -1,
        255,
        -1
    )

    kernel = np.ones(
        (EROSION_SIZE, EROSION_SIZE),
        np.uint8
    )

    eroded = cv2.erode(
        contour_mask,
        kernel,
        iterations=EROSION_ITERATIONS
    )

    ys, xs = np.where(eroded > 0)

    total_pixels = len(xs)

    valid_depth_pixels = 0
    top_pixels = 0

    all_heights = []
    top_heights = []

    top_x_pixels = []
    top_y_pixels = []

    for index in range(0, total_pixels, 2):

        px = int(xs[index])
        py = int(ys[index])

        depth_m = depth_frame.get_distance(px, py)

        if depth_m <= 0:
            continue

        valid_depth_pixels += 1

        point_camera = rs.rs2_deproject_pixel_to_point(
            depth_intrinsics,
            [float(px), float(py)],
            depth_m
        )

        point_camera = (
            np.array(point_camera, dtype=np.float64)
            * 1000.0
        )

        point_camera_h = np.append(
            point_camera,
            1.0
        )

        point_board = T_board_camera @ point_camera_h

        height = float(point_board[2])

        all_heights.append(height)

        if MIN_HEIGHT_MM <= height <= MAX_HEIGHT_MM:

            top_pixels += 1
            top_heights.append(height)

            top_x_pixels.append(px)
            top_y_pixels.append(py)

    acceptance_ratio = 0.0

    if valid_depth_pixels > 0:

        acceptance_ratio = (
            top_pixels / valid_depth_pixels
        ) * 100.0

    diagnostics = {
        "total_pixels": total_pixels,
        "valid_depth_pixels": valid_depth_pixels,
        "top_pixels": top_pixels,
        "acceptance_ratio": acceptance_ratio,
        "median_height": None,
        "all_min": None,
        "all_max": None,
        "all_median": None
    }

    if len(all_heights) > 0:

        diagnostics["all_min"] = float(np.min(all_heights))
        diagnostics["all_max"] = float(np.max(all_heights))
        diagnostics["all_median"] = float(np.median(all_heights))

    if len(top_heights) > 0:

        median_height = float(np.median(top_heights))
        diagnostics["median_height"] = median_height

    else:
        median_height = None

    if top_pixels < MIN_TOP_PIXELS:
        return None, diagnostics

    cx = float(np.median(top_x_pixels))
    cy = float(np.median(top_y_pixels))

    return (
        cx,
        cy,
        median_height,
        top_pixels
    ), diagnostics


# ============================================================
# 6. SHAPE DETECTION
# ============================================================

def detect_shape(contour):

    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)

    if perimeter <= 0:
        return "Unknown", 0.0, 0

    circularity = (
        4.0 * np.pi * area
    ) / (perimeter * perimeter)

    approximation = cv2.approxPolyDP(
        contour,
        0.03 * perimeter,
        True
    )

    vertices = len(approximation)

    if circularity > 0.80:
        shape = "Circle"
    elif vertices == 3:
        shape = "Triangle"
    elif vertices == 4:
        shape = "Square"
    elif vertices in [5, 6, 7]:
        shape = "Hexagon"
    else:
        shape = "Unknown"

    return shape, circularity, vertices


# ============================================================
# 7. WORKSPACE CHECK
# ============================================================

def inside_workspace(x, y):

    return (
        X_MIN <= x <= X_MAX
        and Y_MIN <= y <= Y_MAX
    )


# ============================================================
# 8. FRAME PROCESSING
# ============================================================

def process_frame(
    color_image,
    depth_frame,
    depth_intrinsics,
    T_board_camera
):

    hsv = cv2.cvtColor(
        color_image,
        cv2.COLOR_BGR2HSV
    )

    output = color_image.copy()

    detection_summary = []

    for colour, limits in COLOUR_RANGES.items():

        lower, upper = limits

        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)

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

        valid_candidates = []

        for contour in contours:

            area = cv2.contourArea(contour)

            # ------------------------------------------------
            # AREA FILTER DIAGNOSTIC
            # ------------------------------------------------

            if area < MIN_AREA:

                print(
                    f"\n{colour}: REJECTED by area filter. "
                    f"Area = {area:.1f}, minimum = {MIN_AREA}"
                )

                continue

            # ------------------------------------------------
            # CONTOUR CENTRE AND RAW DEPTH
            # ------------------------------------------------

            M = cv2.moments(contour)

            if M["m00"] != 0:

                contour_cx = M["m10"] / M["m00"]
                contour_cy = M["m01"] / M["m00"]

                raw_center_depth = get_raw_center_depth(
                    depth_frame,
                    contour_cx,
                    contour_cy
                )

                raw_median_depth = get_median_depth(
                    depth_frame,
                    contour_cx,
                    contour_cy
                )

            else:

                contour_cx = None
                contour_cy = None
                raw_center_depth = None
                raw_median_depth = None

            print(f"\n========== {colour} CANDIDATE ==========")
            print("Contour area:", f"{area:.1f}")

            print(
                "Contour centre pixel:",
                contour_cx,
                contour_cy
            )

            print(
                "Raw centre depth:",
                "Invalid" if raw_center_depth is None
                else f"{raw_center_depth:.1f} mm"
            )

            print(
                "Raw 7x7 median depth:",
                "Invalid" if raw_median_depth is None
                else f"{raw_median_depth * 1000.0:.1f} mm"
            )

            # ------------------------------------------------
            # TOP FACE DETECTION
            # ------------------------------------------------

            top_face, diagnostics = get_top_face_center(
                contour,
                depth_frame,
                depth_intrinsics,
                T_board_camera
            )

            print(
                "Total eroded pixels:",
                diagnostics["total_pixels"]
            )

            print(
                "Valid depth pixels:",
                diagnostics["valid_depth_pixels"]
            )

            print(
                "Top-face pixels:",
                diagnostics["top_pixels"]
            )

            print(
                "Top-face acceptance:",
                f"{diagnostics['acceptance_ratio']:.1f}%"
            )

            print(
                "All valid heights (mm):",
                "min =", diagnostics["all_min"],
                "max =", diagnostics["all_max"],
                "median =", diagnostics["all_median"]
            )

            print(
                f"{colour}: Height-filter result: "
                f"{diagnostics['top_pixels']} accepted pixels, "
                f"median height = {diagnostics['all_median']} mm"
            )

            if top_face is None:

                print(
                    f"{colour}: REJECTED by height/top-face filter. "
                    f"Accepted pixels = {diagnostics['top_pixels']}, "
                    f"required = {MIN_TOP_PIXELS}"
                )

                continue

            cx, cy, measured_height, pixel_count = top_face

            depth_m = get_median_depth(
                depth_frame,
                cx,
                cy
            )

            if depth_m is None:

                print(
                    f"{colour}: REJECTED because centre depth "
                    "samples are insufficient."
                )

                continue

            depth_mm = depth_m * 1000.0

            board_point = calculate_xy_from_pixel(
                cx,
                cy,
                T_board_camera
            )

            if board_point is None:

                print(
                    f"{colour}: REJECTED because ray-plane "
                    "intersection is invalid."
                )

                continue

            x_mm = float(board_point[0])
            y_mm = float(board_point[1])
            z_mm = float(board_point[2])

            # ------------------------------------------------
            # WORKSPACE DIAGNOSTIC
            # ------------------------------------------------

            print(
                f"{colour}: Calculated board coordinates: "
                f"X={x_mm:.1f}, Y={y_mm:.1f}, Z={z_mm:.1f} mm"
            )

            if not inside_workspace(x_mm, y_mm):

                print(
                    f"{colour}: REJECTED by workspace filter. "
                    f"X range = [{X_MIN}, {X_MAX}], "
                    f"Y range = [{Y_MIN}, {Y_MAX}]"
                )

                continue

            shape, circularity, vertices = detect_shape(contour)

            valid_candidates.append({

                "colour": colour,
                "shape": EXPECTED_SHAPES[colour],
                "area": area,
                "circularity": circularity,
                "vertices": vertices,
                "cx": cx,
                "cy": cy,
                "depth": depth_mm,
                "height": measured_height,
                "x": x_mm,
                "y": y_mm,
                "z": z_mm,
                "pixel_count": pixel_count,
                "contour": contour

            })

            print(f"{colour}: ACCEPTED.")

        # ----------------------------------------------------
        # SELECT BEST CANDIDATE
        # ----------------------------------------------------

        if len(valid_candidates) > 0:

            best = max(
                valid_candidates,
                key=lambda item: item["area"]
            )

            detection_summary.append(best)

            cv2.drawContours(
                output,
                [best["contour"]],
                -1,
                (0, 255, 0),
                2
            )

            center_pixel = (
                int(round(best["cx"])),
                int(round(best["cy"]))
            )

            cv2.circle(
                output,
                center_pixel,
                5,
                (0, 0, 255),
                -1
            )

            label = (
                f"{colour}: "
                f"X={best['x']:.1f}, "
                f"Y={best['y']:.1f}"
            )

            text_position = (
                center_pixel[0] - 80,
                center_pixel[1] - 15
            )

            cv2.putText(
                output,
                label,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1
            )

        cv2.imshow(colour + " Mask", mask)

    return output, detection_summary


# ============================================================
# 9. MAIN CAMERA FUNCTION
# ============================================================

def main():

    T_camera_board, T_board_camera = load_calibration(
        CALIBRATION_FILE
    )

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        rs.format.bgr8,
        FPS
    )

    config.enable_stream(
        rs.stream.depth,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        rs.format.z16,
        FPS
    )

    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)

    color_profile = (
        profile.get_stream(rs.stream.color)
        .as_video_stream_profile()
    )

    depth_intrinsics = color_profile.get_intrinsics()

    print("\nRealSense camera started.")
    print("Contour-centre diagnostics enabled.")
    print("Area, height and workspace rejection diagnostics enabled.")
    print("Top-face detection enabled.")
    print("\nPress Q to quit.")

    try:

        while True:

            frames = pipeline.wait_for_frames()

            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(
                color_frame.get_data()
            )

            output, detections = process_frame(
                color_image,
                depth_frame,
                depth_intrinsics,
                T_board_camera
            )

            cv2.imshow(
                "Peg Detection - Diagnostic Mode",
                output
            )

            if len(detections) > 0:

                print(
                    "\n========== DETECTION SUMMARY =========="
                )

                for item in detections:

                    print("Colour:", item["colour"])
                    print("Detected shape:", item["shape"])

                    print(
                        "Contour area:",
                        f"{item['area']:.1f}"
                    )

                    print(
                        "Centre pixel:",
                        item["cx"],
                        item["cy"]
                    )

                    print(
                        "Depth:",
                        f"{item['depth']:.1f} mm"
                    )

                    print(
                        "Top-face pixel count:",
                        item["pixel_count"]
                    )

                    print(
                        "Depth-derived height:",
                        f"{item['height']:.1f} mm"
                    )

                    print(
                        "Board coordinates (mm):",
                        f"{item['x']:.1f}",
                        f"{item['y']:.1f}",
                        f"{item['z']:.1f}"
                    )

                    print("---------------------------------------")

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

    finally:

        pipeline.stop()
        cv2.destroyAllWindows()

        print("\nCamera stopped.")


if __name__ == "__main__":
    main()
