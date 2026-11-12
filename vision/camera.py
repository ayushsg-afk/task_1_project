
import pyrealsense2 as rs
import numpy as np
import cv2

def main():

    # Create RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Configure RGB and depth streams
    config.enable_stream(
        rs.stream.color, 640, 480, rs.format.bgr8, 30
    )

    config.enable_stream(
        rs.stream.depth, 640, 480, rs.format.z16, 30
    )

    # Align depth frames with RGB frames
    align = rs.align(rs.stream.color)

    try:
        print("Starting Intel RealSense D435i...")

        pipeline.start(config)

        print("Camera started successfully!")
        print("Press ESC to exit.")

        # Depth colorizer
        colorizer = rs.colorizer()

        while True:

            # Wait for frames
            frames = pipeline.wait_for_frames()

            # Align depth to color
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Convert frames to NumPy arrays
            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = np.asanyarray(
                colorizer.colorize(depth_frame).get_data()
            )

            # Display images
            cv2.imshow("RGB Camera", color_image)
            cv2.imshow("Depth Camera", depth_colormap)

            # Exit using ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except Exception as e:
        print("Camera error:", e)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera stopped.")


if __name__ == "__main__":
    main()
