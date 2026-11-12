
import cv2
import numpy as np
import pyrealsense2 as rs


# Start RealSense color stream
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color, 640, 480, rs.format.bgr8, 30
)

# HSV color ranges

COLOR_RANGES = {

    "Orange": [
        (np.array([5, 100, 100]), np.array([19, 255, 255]))
    ],

    "Green": [
        (np.array([35, 70, 50]), np.array([85, 255, 255]))
    ],

    "Yellow": [
        (np.array([20, 100, 100]), np.array([35, 255, 255]))
    ],

    "Blue": [
        (np.array([90, 70, 50]), np.array([130, 255, 255]))
    ]
}










def classify_shape(contour):

    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return "Unknown"

    approx = cv2.approxPolyDP(
        contour, 0.012 * perimeter, True
    )

    vertices = len(approx)

    if vertices == 3:
        return "Triangle"

    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)

        if 0.65 <= aspect_ratio <= 1.35:
            return "Square"
        else:
            return "Unknown"

    elif vertices == 6:
        return "Hexagon"

    elif vertices >= 8:
        return "Circle"

    return "Unknown"








def detect_pegs(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    kernel = np.ones((5, 5), np.uint8)

    for color_name, ranges in COLOR_RANGES.items():

        # Combine the masks for this color
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for lower, upper in ranges:
            current_mask = cv2.inRange(
                hsv, lower, upper
            )
            mask = cv2.bitwise_or(mask, current_mask)

        # Remove small noise
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(contour)

            # Ignore small regions
            if area < 500:
                continue

            shape = classify_shape(contour)

            x, y, w, h = cv2.boundingRect(contour)

            # Draw contour and bounding box
            cv2.drawContours(
                frame, [contour], -1, (0, 255, 0), 2
            )

            cv2.rectangle(
                frame, (x, y), (x + w, y + h),
                (255, 0, 0), 2
            )

            label = color_name + " " + shape

            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    return frame


try:
    pipeline.start(config)

    print("Peg detection started!")
    print("Show your colored pegs to the camera.")
    print("Press ESC to exit.")

    while True:

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        result = detect_pegs(frame)

        cv2.imshow("Peg Detection", result)

        if cv2.waitKey(1) & 0xFF == 27:
            break

except Exception as e:
    print("Camera error:", e)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Camera stopped.")
