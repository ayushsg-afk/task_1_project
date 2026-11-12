
import cv2
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

colors = {
    "Orange": ([2, 80, 70], [18, 255, 255]),
    "Green": ([30, 70, 50], [80, 255, 255]),
    "Yellow": ([18, 80, 70], [35, 255, 255]),
    "Blue": ([85, 70, 50], [135, 255, 255])
}

kernel = np.ones((3, 3), np.uint8)

def classify_shape(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)

    if perimeter == 0 or area == 0:
        return "Unknown"

    circularity = 4 * np.pi * area / (perimeter * perimeter)

    # Circle detection
    if circularity > 0.82:
        return "Circle"

    # Polygon detection
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    vertices = len(approx)

    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        return "Square"
    elif vertices in [5, 6, 7]:
        return "Hexagon"

    return "Unknown"


try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        output = frame.copy()

        for name, (low, high) in colors.items():

            mask = cv2.inRange(
                hsv,
                np.array(low, dtype=np.uint8),
                np.array(high, dtype=np.uint8)
            )

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:

                if cv2.contourArea(contour) < 100:
                    continue

                shape = classify_shape(contour)

                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

                label = name + " " + shape

                cv2.putText(
                    output, label, (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2
                )

        cv2.imshow("Peg Detection", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
