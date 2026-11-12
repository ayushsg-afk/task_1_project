
import pyrealsense2 as rs
import numpy as np
import cv2

# Start camera
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color, 640, 480, rs.format.bgr8, 30
)
config.enable_stream(
    rs.stream.depth, 640, 480, rs.format.z16, 30
)

# Align depth image with color image
align = rs.align(rs.stream.color)

depth_frame = None
selected_pixel = None


# Mouse callback: measure depth where you click
def mouse_callback(event, x, y, flags, param):
    global selected_pixel

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_pixel = (x, y)

        if depth_frame:
            distance = depth_frame.get_distance(x, y)

            if distance > 0:
                print("--------------------------------")
                print("Pixel X:", x)
                print("Pixel Y:", y)
                print("Depth:", round(distance, 4), "meters")
                print("--------------------------------")
            else:
                print("Invalid depth. Try another pixel.")


try:
    print("Starting RealSense camera...")
    pipeline.start(config)

    cv2.namedWindow("RGB Image")
    cv2.setMouseCallback("RGB Image", mouse_callback)

    colorizer = rs.colorizer()

    print("Camera started!")
    print("Click on an object in the RGB image to measure its depth.")
    print("Press ESC to exit.")

    while True:

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = np.asanyarray(
            colorizer.colorize(depth_frame).get_data()
        )

        # Display selected pixel
        if selected_pixel is not None:
            x, y = selected_pixel

            cv2.circle(color_image, (x, y), 5, (0, 0, 255), -1)

        cv2.imshow("RGB Image", color_image)
        cv2.imshow("Depth Image", depth_colormap)

        if cv2.waitKey(1) & 0xFF == 27:
            break

except Exception as e:
    print("Error:", e)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Camera stopped.")
