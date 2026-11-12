import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color, 640, 480,
    rs.format.bgr8, 30
)

config.enable_stream(
    rs.stream.depth, 640, 480,
    rs.format.z16, 30
)

pipeline.start(config)

align = rs.align(rs.stream.color)

print("Camera started.")
print("Press S to capture robot base.")
print("Press Q to quit.")

try:
    while True:

        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        color = frames.get_color_frame()
        depth = frames.get_depth_frame()

        if not color or not depth:
            continue

        image = np.asanyarray(color.get_data())
        depth_image = np.asanyarray(depth.get_data())

        cv2.imshow("Robot Base Capture", image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):

            cv2.imwrite(
                "vision/robot_base_view.png",
                image
            )

            np.save(
                "vision/robot_base_depth.npy",
                depth_image
            )

            print("Images saved successfully.")
            break

        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
