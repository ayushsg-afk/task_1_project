import cv2
import numpy as np
import pyrealsense2 as rs

# Load existing calibration
data = np.load("calibration/data/camera_to_board_v2.npz")
T_camera_board = data["T_camera_board"]

R = T_camera_board[:3, :3]
t = T_camera_board[:3, 3]
board_normal = R[:, 2]

# Start RealSense
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color, 640, 480, rs.format.bgr8, 30
)
config.enable_stream(
    rs.stream.depth, 640, 480, rs.format.z16, 30
)

profile = pipeline.start(config)
align = rs.align(rs.stream.color)

points_camera = []
points_board = []

selected_pixel = None
processing_click = False


def mouse_callback(event, x, y, flags, param):
    global selected_pixel, processing_click

    if event == cv2.EVENT_LBUTTONDOWN and not processing_click:
        selected_pixel = (x, y)
        processing_click = True


def get_median_depth(depth_frame, x, y):
    values = []

    for dy in range(-2, 3):
        for dx in range(-2, 3):
            px = x + dx
            py = y + dy

            if 0 <= px < 640 and 0 <= py < 480:
                z = depth_frame.get_distance(px, py)

                if z > 0:
                    values.append(z)

    if len(values) < 5:
        return None

    return float(np.median(values))


def camera_to_board(point):
    p = np.append(point, 1.0)
    return (np.linalg.inv(T_camera_board) @ p)[:3]


print("\nBOARD PLANE VALIDATION")
print("----------------------")
print("Click 6 to 9 different WHITE areas on the sheet.")
print("Avoid marker prints, marker borders and sheet edges.")
print("Each click collects 15 depth frames.")
print("Press Q when finished.")
print("Press C to clear all collected points.\n")

cv2.namedWindow("Board Plane Validation")
cv2.setMouseCallback("Board Plane Validation", mouse_callback)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        image = np.asanyarray(color_frame.get_data())

        if processing_click and selected_pixel is not None:
            x, y = selected_pixel
            depth_samples = []

            for _ in range(15):
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                depth_frame = aligned.get_depth_frame()

                z = get_median_depth(depth_frame, x, y)

                if z is not None:
                    depth_samples.append(z)

            if len(depth_samples) >= 10:
                z = float(np.median(depth_samples))

                intr = (
                    depth_frame.profile
                    .as_video_stream_profile()
                    .get_intrinsics()
                )

                point = np.array(
                    rs.rs2_deproject_pixel_to_point(
                        intr, [x, y], z
                    ),
                    dtype=float
                )

                board_point = camera_to_board(point)

                points_camera.append(point)
                points_board.append(board_point)

                print("\nPoint", len(points_camera))
                print("Pixel:", x, y)
                print("Median depth:", round(z * 1000, 2), "mm")
                print("Camera XYZ (mm):", np.round(point * 1000, 2))
                print("Board XYZ (mm):", np.round(board_point * 1000, 2))

            else:
                print("\nNot enough valid depth readings. Try another point.")

            selected_pixel = None
            processing_click = False

        cv2.putText(
            image,
            f"Points collected: {len(points_camera)}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            "Click white paper areas | Q: finish | C: clear",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )

        cv2.imshow("Board Plane Validation", image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('c'):
            points_camera.clear()
            points_board.clear()
            print("\nAll points cleared.")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()


# Fit a plane to depth-camera points
if len(points_camera) < 6:
    print("\nAt least 6 valid points are required.")
else:
    P = np.array(points_camera)

    centroid = np.mean(P, axis=0)
    centered = P - centroid

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    depth_normal = Vt[-1]

    # Make both normals point in the same direction
    if np.dot(depth_normal, board_normal) < 0:
        depth_normal = -depth_normal

    dot_value = np.clip(
        np.dot(depth_normal, board_normal), -1.0, 1.0
    )

    angle = np.degrees(np.arccos(dot_value))

    # Distance of depth points from fitted depth plane
    residuals = centered @ depth_normal
    plane_rms = np.sqrt(np.mean(residuals ** 2))

    # Board-plane height of the depth measurements
    board_z = np.array(points_board)[:, 2]

    mean_offset = np.mean(board_z)
    z_range = np.max(board_z) - np.min(board_z)

    print("\n========== PLANE VALIDATION RESULTS ==========")

    print("\nNumber of points:", len(P))

    print("\nDepth-fitted plane normal:")
    print(np.round(depth_normal, 6))

    print("\nArUco board plane normal:")
    print(np.round(board_normal, 6))

    print("\nAngle between planes (degrees):")
    print(round(angle, 4))

    print("\nDepth plane fitting RMS (mm):")
    print(round(plane_rms * 1000, 3))

    print("\nMean board Z offset (mm):")
    print(round(mean_offset * 1000, 3))

    print("\nBoard Z range (mm):")
    print(round(z_range * 1000, 3))

    print("\n==============================================")
