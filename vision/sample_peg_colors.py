
import cv2
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color, 640, 480, rs.format.bgr8, 30
)

pipeline.start(config)

samples = []

def mouse_callback(event, x, y, flags, param):
    global samples

    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        patch = hsv[
            max(0, y-7):min(480, y+8),
            max(0, x-7):min(640, x+8)
        ]

        median = np.median(
            patch.reshape(-1, 3), axis=0
        )

        samples.append(median)

        print(
            f"Sample {len(samples)} | "
            f"Pixel: ({x}, {y}) | "
            f"HSV: {median.astype(int)}"
        )

cv2.namedWindow("Peg Colour Sampling")
cv2.setMouseCallback(
    "Peg Colour Sampling", mouse_callback
)

print("\nPEG COLOUR SAMPLING")
print("-------------------")
print("Click the centre of each peg's coloured surface.")
print("Order: Orange, Green, Yellow, Blue")
print("Press Q to quit.\n")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        display = frame.copy()

        cv2.putText(
            display,
            "Click peg centres | Q = Quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow("Peg Colour Sampling", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

print("\nCollected HSV samples:")
for i, sample in enumerate(samples):
    print(f"Sample {i+1}: {sample.astype(int)}")
