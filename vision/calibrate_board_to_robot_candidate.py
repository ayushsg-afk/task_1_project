#!/usr/bin/env python3

import os
import sys
import math
import cv2
import numpy as np
import pyrealsense2 as rs

from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kinematics"))

from forward_kinematics import forward_kinematics
from robot_parameters import JOINT_ORIGINS

DEVICE = "/dev/ttyUSB0"
BAUDRATE = 1000000
PROTOCOL = 2.0

HOME = {11: 2, 12: 1982, 13: 2377, 14: 2811}
MOTOR_IDS = [11, 12, 13, 14]

# Same camera-to-board calibration as the existing vision script.
T_camera_board = np.array([
    [-0.03968687, -0.99609763,  0.07883184, -0.03551927],
    [-0.98504565,  0.02576442, -0.17035628,  0.09886492],
    [ 0.16766043, -0.08441387, -0.98222415,  0.47676925],
    [ 0.0,         0.0,         0.0,         1.0]
])
T_board_camera = np.linalg.inv(T_camera_board)

ADDR_MODEL = 0
ADDR_DRIVE = 10
ADDR_MODE = 11
ADDR_HOMING = 20
ADDR_TORQUE = 64
ADDR_ERROR = 70
ADDR_POSITION = 132

selected = []
names = [
    "J1 ROTATION AXIS CENTRE",
    "LEFT GRIPPER FINGERTIP",
    "RIGHT GRIPPER FINGERTIP"
]


def read_register(packet, port, motor_id, address, size):
    if size == 1:
        value, result, error = packet.read1ByteTxRx(
            port, motor_id, address
        )
    elif size == 2:
        value, result, error = packet.read2ByteTxRx(
            port, motor_id, address
        )
    else:
        value, result, error = packet.read4ByteTxRx(
            port, motor_id, address
        )

    if result != COMM_SUCCESS:
        raise RuntimeError(packet.getTxRxResult(result))
    if error != 0:
        raise RuntimeError(packet.getRxPacketError(error))

    return value


def signed32(value):
    return value - 2**32 if value >= 2**31 else value


def read_joint_angles():
    port = PortHandler(DEVICE)
    packet = PacketHandler(PROTOCOL)

    if not port.openPort():
        raise RuntimeError("Cannot open U2D2")

    try:
        if not port.setBaudRate(BAUDRATE):
            raise RuntimeError("Cannot set baud rate")

        positions = []

        for motor_id in MOTOR_IDS:
            model = read_register(packet, port, motor_id, ADDR_MODEL, 2)
            drive = read_register(packet, port, motor_id, ADDR_DRIVE, 1)
            mode = read_register(packet, port, motor_id, ADDR_MODE, 1)
            torque = read_register(packet, port, motor_id, ADDR_TORQUE, 1)
            offset = signed32(
                read_register(packet, port, motor_id, ADDR_HOMING, 4)
            )
            error = read_register(packet, port, motor_id, ADDR_ERROR, 1)
            pos = read_register(packet, port, motor_id, ADDR_POSITION, 4)

            if model != 1020 or mode != 3 or torque != 0:
                raise RuntimeError(
                    f"ID {motor_id}: unexpected model/mode/torque"
                )
            if drive & 1 or offset != 0 or error != 0:
                raise RuntimeError(
                    f"ID {motor_id}: direction, offset or hardware error "
                    "does not match the assumed mapping"
                )
            if not 0 <= pos <= 4095:
                raise RuntimeError(f"ID {motor_id}: invalid position")

            positions.append(pos)

            print(
                f"ID {motor_id}: position={pos}, "
                f"torque={torque}, drive={drive}, mode={mode}"
            )

    finally:
        port.closePort()

    q = (np.array(positions, dtype=float) - 2048.0) * (
        2.0 * math.pi / 4096.0
    )

    return q


def board_point(depth_frame, color_frame, px, py):
    intr = color_frame.profile.as_video_stream_profile().intrinsics

    samples = []
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            x, y = px + dx, py + dy
            if 0 <= x < intr.width and 0 <= y < intr.height:
                d = depth_frame.get_distance(x, y)
                if d > 0:
                    samples.append(d)

    if not samples:
        raise RuntimeError(f"No valid depth at pixel ({px},{py})")

    depth = float(np.median(samples))
    camera_point = rs.rs2_deproject_pixel_to_point(
        intr, [float(px), float(py)], depth
    )
    homogeneous = np.append(np.array(camera_point), 1.0)
    return (T_board_camera @ homogeneous)[:3] * 1000.0


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(selected) < 3:
        selected.append((x, y))
        print(f"Selected {names[len(selected)-1]}: ({x}, {y})")


def main():
    print("\nREAD-ONLY BOARD-TO-ROBOT CALIBRATION")
    print("Torque must remain OFF. No motor writes are used.")

    q = read_joint_angles()
    print("\nNominal joint angles (degrees):", np.round(np.degrees(q), 3))

    T_local = forward_kinematics(q)
    tcp_local = T_local[:3, 3] * 1000.0

    # Joint 1 axis location relative to the robot base frame.
    axis_local = JOINT_ORIGINS[0] * 1000.0

    predicted_vector = tcp_local - axis_local
    predicted_length = np.linalg.norm(predicted_vector[:2])

    print("Predicted TCP in robot frame (mm):", np.round(tcp_local, 2))
    print("Predicted J1-axis-to-TCP vector (mm):",
          np.round(predicted_vector, 2))
    print("Predicted horizontal separation (mm):",
          round(predicted_length, 2))

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)
    align = rs.align(rs.stream.color)

    cv2.namedWindow("Board to Robot Candidate")
    cv2.setMouseCallback("Board to Robot Candidate", mouse_callback)

    print("\nClick in this order:")
    for i, name in enumerate(names, 1):
        print(f"{i}. {name}")
    print("Press C to calculate, R to reset, Q to quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color = aligned.get_color_frame()
            depth = aligned.get_depth_frame()

            if not color or not depth:
                continue

            image = np.asanyarray(color.get_data())

            for i, point in enumerate(selected):
                cv2.circle(image, point, 6, (0, 0, 255), -1)
                cv2.putText(
                    image, str(i + 1),
                    (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

            cv2.imshow("Board to Robot Candidate", image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                selected.clear()
                print("Selections reset.")

            elif key == ord('c'):
                if len(selected) != 3:
                    print("Select all three points first.")
                    continue

                measured = [
                    board_point(depth, color, x, y)
                    for x, y in selected
                ]

                axis_board = measured[0]
                tcp_board = 0.5 * (measured[1] + measured[2])

                measured_vector = tcp_board - axis_board
                measured_length = np.linalg.norm(measured_vector[:2])
                length_error = abs(measured_length - predicted_length)

                print("\nMeasured J1 axis (board mm):",
                      np.round(axis_board, 2))
                print("Measured gripper midpoint (board mm):",
                      np.round(tcp_board, 2))
                print("Measured horizontal separation (mm):",
                      round(measured_length, 2))
                print("Difference from FK prediction (mm):",
                      round(length_error, 2))

                if length_error > 15.0:
                    print("\nREJECTED: measured and predicted lengths differ")
                    print("by more than 15 mm. Check points and calibration.")
                    continue

                angle_board = math.atan2(
                    measured_vector[1], measured_vector[0]
                )
                angle_robot = math.atan2(
                    predicted_vector[1], predicted_vector[0]
                )
                yaw = angle_board - angle_robot
                yaw = math.atan2(math.sin(yaw), math.cos(yaw))

                c, s = math.cos(yaw), math.sin(yaw)
                R = np.array([
                    [c, -s, 0.0],
                    [s,  c, 0.0],
                    [0.0, 0.0, 1.0]
                ])

                # XY translation uses the measured J1 axis.
                base_xy = axis_board[:2] - R[:2, :2] @ axis_local[:2]

                # Z translation uses the measured midpoint of both
                # fingertip surfaces and the URDF TCP height.
                base_z = tcp_board[2] - tcp_local[2]

                T_board_robot = np.eye(4)
                T_board_robot[:3, :3] = R
                T_board_robot[:3, 3] = [
                    base_xy[0], base_xy[1], base_z
                ]

                print("\nCANDIDATE ROBOT-BASE POSE IN BOARD FRAME")
                print("Position (mm):",
                      np.round(T_board_robot[:3, 3], 2))
                print("Yaw (degrees):", round(math.degrees(yaw), 2))
                print("T_board_robot:")
                print(np.round(T_board_robot, 5))

                out = os.path.join(
                    ROOT, "calibration",
                    "board_to_robot_candidate_20261112.npz"
                )
                np.savez(
                    out,
                    T_board_robot_candidate=T_board_robot,
                    joint_angles_rad=q,
                    axis_board_mm=axis_board,
                    left_tip_board_mm=measured[1],
                    right_tip_board_mm=measured[2],
                    tcp_board_mm=tcp_board,
                    predicted_horizontal_separation_mm=predicted_length,
                    measured_horizontal_separation_mm=measured_length,
                    length_error_mm=length_error,
                    execution_allowed=False
                )

                print("\nSaved candidate only:", out)
                print("Physical execution remains DISABLED.")
                break

            elif key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
