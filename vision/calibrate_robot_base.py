
import cv2
import numpy as np
import pyrealsense2 as rs
import math

# ==========================================
# CAMERA-TO-BOARD CALIBRATION
# ==========================================

T_camera_board = np.array([
    [-0.03968687, -0.99609763,  0.07883184, -0.03551927],
    [-0.98504565,  0.02576442, -0.17035628,  0.09886492],
    [ 0.16766043, -0.08441387, -0.98222415,  0.47676925],
    [ 0.0,         0.0,         0.0,         1.0]
])

T_board_camera = np.linalg.inv(T_camera_board)

# ==========================================
# REALSENSE CONFIGURATION
# ==========================================

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color, 640, 480, rs.format.bgr8, 30
)

config.enable_stream(
    rs.stream.depth, 640, 480, rs.format.z16, 30
)

pipeline.start(config)

align = rs.align(rs.stream.color)

# ==========================================
# GLOBAL VARIABLES
# ==========================================

selected_points = []
point_names = [
    "BASE ROTATION AXIS CENTRE",
    "ROBOT FORWARD DIRECTION"
]

# ==========================================
# PIXEL TO BOARD COORDINATES
# ==========================================

def get_board_coordinates(depth_frame, color_frame, px, py):

    intr = (
        color_frame.profile
        .as_video_stream_profile()
        .intrinsics
    )

    depth_values = []

    for dy in range(-3, 4):
        for dx in range(-3, 4):

            x = px + dx
            y = py + dy

            if x < 0 or y < 0:
                continue

            if x >= intr.width or y >= intr.height:
                continue

            d = depth_frame.get_distance(x, y)

            if d > 0:
                depth_values.append(d)

    if len(depth_values) < 5:
        return None

    depth = float(np.median(depth_values))

    point_camera = rs.rs2_deproject_pixel_to_point(
        intr,
        [float(px), float(py)],
        depth
    )

    point_camera = np.array(point_camera)

    point_camera_h = np.append(point_camera, 1.0)

    point_board = T_board_camera @ point_camera_h

    return point_board[:3] * 1000


# ==========================================
# MOUSE CLICK CALLBACK
# ==========================================

def mouse_callback(event, x, y, flags, param):

    global selected_points

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(selected_points) >= 2:
            print("Both points already selected.")
            return

        selected_points.append((x, y))

        print(
            "\nSelected",
            point_names[len(selected_points) - 1],
            "at pixel:",
            x, y
        )


# ==========================================
# MAIN PROGRAM
# ==========================================

cv2.namedWindow("Robot Base Calibration")

cv2.setMouseCallback(
    "Robot Base Calibration",
    mouse_callback
)

print("\n====================================")
print(" ROBOT BASE POSITION CALIBRATION")
print("====================================")

print("\n1. Click the centre of the base rotation axis.")
print("2. Click a point in the robot's forward direction.")
print("3. Press R to reset selections.")
print("4. Press C to calculate.")
print("5. Press Q to quit.")

try:

    while True:

        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        image = np.asanyarray(color_frame.get_data())

        # Draw selected points
        for i, point in enumerate(selected_points):

            cv2.circle(
                image,
                point,
                7,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                image,
                str(i + 1),
                (point[0] + 10, point[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "Robot Base Calibration",
            image
        )

        key = cv2.waitKey(1) & 0xFF

        # Reset selections
        if key == ord('r'):

            selected_points.clear()

            print("\nSelections reset.")

        # Calculate coordinates
        elif key == ord('c'):

            if len(selected_points) != 2:

                print("\nPlease select exactly two points.")
                continue

            board_points = []

            for point in selected_points:

                result = get_board_coordinates(
                    depth_frame,
                    color_frame,
                    point[0],
                    point[1]
                )

                if result is None:
                    print("Invalid depth. Please select again.")
                    board_points = []
                    break

                board_points.append(result)

            if len(board_points) != 2:
                continue

            base = board_points[0]
            forward = board_points[1]

            dx = forward[0] - base[0]
            dy = forward[1] - base[1]

            heading = math.degrees(
                math.atan2(dy, dx)
            )

            print("\n====================================")
            print(" ROBOT BASE CALIBRATION RESULTS")
            print("====================================")

            print("\nBase position in board coordinates:")

            print("X =", round(base[0], 2), "mm")
            print("Y =", round(base[1], 2), "mm")
            print("Z =", round(base[2], 2), "mm")

            print("\nForward point:")
            print("X =", round(forward[0], 2), "mm")
            print("Y =", round(forward[1], 2), "mm")
            print("Z =", round(forward[2], 2), "mm")

            print("\nEstimated heading:")
            print(round(heading, 2), "degrees")

            np.savez(
                "calibration/robot_base_estimate.npz",
                base_position_mm=base,
                forward_position_mm=forward,
                heading_deg=heading
            )

            print(
                "\nSaved to calibration/robot_base_estimate.npz"
            )

        elif key == ord('q'):
            break

finally:

    pipeline.stop()
    cv2.destroyAllWindows()

    print("\nCamera stopped.")
