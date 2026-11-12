
import cv2
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color, 640, 480, rs.format.bgr8, 30
)

pipeline.start(config)

points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

cv2.namedWindow("Blue Colour Sampling")
cv2.setMouseCallback("Blue Colour Sampling", mouse_callback)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        display = frame.copy()

        cv2.putText(
            display,
            "Click BLUE PEG, then EMPTY BLUE WORKSPACE",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

        for i, (x, y) in enumerate(points):
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)

            patch = hsv[
                max(0, y-7):min(480, y+8),
                max(0, x-7):min(640, x+8)
            ]

            median = np.median(
                patch.reshape(-1, 3), axis=0
            )

            print(
                ["BLUE PEG", "WORKSPACE"][i]
                if i < 2 else "Additional sample",
                "HSV:",
                np.round(median, 1)
            )

        cv2.imshow("Blue Colour Sampling", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
