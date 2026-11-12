
import cv2
import numpy as np
import pyrealsense2 as rs

if not hasattr(cv2, "aruco"):
    raise RuntimeError(
        "ArUco is unavailable. Install opencv-contrib-python."
    )

# Dictionaries to test
dictionary_names = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_6X6_250",
    "DICT_6X6_1000",
    "DICT_7X7_250"
]

dictionaries = {}

for name in dictionary_names:
    dictionary_id = getattr(cv2.aruco, name)
    dictionaries[name] = cv2.aruco.getPredefinedDictionary(
        dictionary_id
    )

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color, 640, 480, rs.format.bgr8, 30
)

seen = set()

try:
    pipeline.start(config)

    print("ArUco detection started.")
    print("Place your marker flat on the table.")
    print("Press ESC to exit.")

    while True:

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())
        output = image.copy()

        for name, dictionary in dictionaries.items():

            if hasattr(cv2.aruco, "ArucoDetector"):
                detector = cv2.aruco.ArucoDetector(dictionary)
                corners, ids, rejected = detector.detectMarkers(image)
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    image, dictionary
                )

            if ids is None:
                continue

            for i, marker_id in enumerate(ids.flatten()):

                key = (name, int(marker_id))

                if key not in seen:
                    print(
                        "Detected dictionary:",
                        name,
                        "| Marker ID:",
                        int(marker_id)
                    )
                    seen.add(key)

            cv2.aruco.drawDetectedMarkers(
                output, corners, ids
            )

            cv2.putText(
                output,
                name,
                (10, 25 + 25 * dictionary_names.index(name)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1
            )

        cv2.imshow("ArUco Detection", output)

        if cv2.waitKey(1) & 0xFF == 27:
            break

except Exception as e:
    print("Error:", e)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Camera stopped.")
