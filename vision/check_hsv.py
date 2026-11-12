import cv2
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

print("Press Q to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        cv2.imshow("Camera", image)
        cv2.imshow("HSV Green", cv2.inRange(
            hsv, np.array([36, 70, 50]), np.array([80, 255, 255])
        ))
        cv2.imshow("HSV Yellow", cv2.inRange(
            hsv, np.array([18, 80, 70]), np.array([35, 255, 255])
        ))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
