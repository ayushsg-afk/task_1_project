
import cv2
import numpy as np
import pyrealsense2 as rs

# ==========================================
# 1. LOAD CAMERA CALIBRATION
# ==========================================

data = np.load("calibration/data/camera_to_board_v2.npz")

T_camera_board = data["T_camera_board"]
K = data["camera_matrix"]
dist = data["distortion"]

# ==========================================
# 2. DEFINE WORKSPACE COORDINATES
# ==========================================

# Workspace size
L = 0.300  # 300 mm in metres

# Bottom-left corner relative to ArUco origin
X0 = -0.063
Y0 = -0.295

# Workspace corners in board coordinates
# Board convention: +X right, +Y upward, +Z towards camera

workspace_board = np.array([
    [X0,     Y0,     0.0, 1.0],  # Bottom-left
    [X0 + L, Y0,     0.0, 1.0],  # Bottom-right
    [X0 + L, Y0 + L, 0.0, 1.0],  # Top-right
    [X0,     Y0 + L, 0.0, 1.0]   # Top-left
])

# ==========================================
# 3. TRANSFORM BOARD POINTS TO CAMERA FRAME
# ==========================================

# solvePnP gives board-to-camera transformation
workspace_camera = (
    T_camera_board @ workspace_board.T
).T[:, :3]

# ==========================================
# 4. PROJECT 3D POINTS INTO IMAGE
# ==========================================

image_points, _ = cv2.projectPoints(
    workspace_camera,
    np.zeros((3, 1)),
    np.zeros((3, 1)),
    K,
    dist
)

image_points = np.round(
    image_points.reshape(-1, 2)
).astype(np.int32)

print("\nWorkspace image coordinates:")

labels = ["BL", "BR", "TR", "TL"]

for label, point in zip(labels, image_points):
    print(label, point)

print("\nWorkspace coordinates in board frame (mm):")
for label, point in zip(labels, workspace_board):
    print(label, point[:2] * 1000)

# ==========================================
# 5. START REALSENSE CAMERA
# ==========================================

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30
)

pipeline.start(config)

# ==========================================
# 6. DISPLAY WORKSPACE BOUNDARY
# ==========================================

try:

    while True:

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        # Draw workspace boundary
        cv2.polylines(
            frame,
            [image_points],
            True,
            (0, 255, 0),
            3
        )

        # Draw and label corners
        for point, label in zip(image_points, labels):

            cv2.circle(
                frame,
                tuple(point),
                6,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                label,
                tuple(point + [5, -8]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        cv2.putText(
            frame,
            "300 x 300 mm Workspace",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Workspace Boundary Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    pipeline.stop()
    cv2.destroyAllWindows()
