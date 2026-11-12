#!/usr/bin/env python3

import sys
import time
import select
import termios
import tty

from dynamixel_sdk import PortHandler, PacketHandler

DEVICE = "/dev/ttyUSB0"
BAUDRATE = 1000000
PROTOCOL = 2.0

ARM_MOTOR_IDS = [11, 12, 13, 14]
GRIPPER_ID = 15

ADDR_DRIVE_MODE = 10
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_HARDWARE_ERROR = 70
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

GRIPPER_OPEN_POS = 2600
GRIPPER_CLOSE_POS = 1500
BASE_ROTATION_SPAN = 2048

HOME_POSITIONS = {
    11: 0,
    12: 2170,
    13: 2120,
    14: 2840,
    15: 2640,
}

JOINT_LIMITS = {
    11: (-2048, 2048),
    12: (800, 3300),
    13: (800, 3300),
    14: (1000, 3300),
    15: (1300, 2800),
}

DEFAULT_STEP_COUNTS = 15
MIN_STEP_COUNTS = 2
MAX_STEP_COUNTS = 150

TELEOP_ACCELERATION_MS = 100
TELEOP_VELOCITY_MS = 400
HOME_VELOCITY_MS = 5000
HOME_ACCELERATION_MS = 1500

port = PortHandler(DEVICE)
packet = PacketHandler(PROTOCOL)
original_terminal = termios.tcgetattr(sys.stdin)
active_motors = []
motor_modes = {}


def read_position(motor_id):
    value, result, error = packet.read4ByteTxRx(port, motor_id, ADDR_PRESENT_POSITION)
    if result != 0 or error != 0:
        err_msg = packet.getRxPacketError(error) if error != 0 else packet.getTxRxResult(result)
        raise RuntimeError(f"Read pos failed on ID {motor_id}: {err_msg}")
    if value >= 2**31:
        value -= 2**32
    return value


def read_register(motor_id, address):
    value, result, error = packet.read1ByteTxRx(port, motor_id, address)
    if result != 0 or error != 0:
        err_msg = packet.getRxPacketError(error) if error != 0 else packet.getTxRxResult(result)
        raise RuntimeError(f"Read reg failed on ID {motor_id}, addr {address}: {err_msg}")
    return value


def write_register(motor_id, address, value):
    result, error = packet.write1ByteTxRx(port, motor_id, address, value)
    if result != 0 or error != 0:
        err_msg = packet.getRxPacketError(error) if error != 0 else packet.getTxRxResult(result)
        raise RuntimeError(f"Write reg failed on ID {motor_id}, addr {address}: {err_msg}")


def safe_goal_position(motor_id, raw_target):
    mode = motor_modes.get(motor_id, 3)
    if mode == 3:
        return max(0, min(4095, int(raw_target) % 4096))
    return int(raw_target)


def write_4byte(motor_id, address, value):
    val_to_write = safe_goal_position(motor_id, value) if address == ADDR_GOAL_POSITION else int(value)
    val_32 = val_to_write & 0xFFFFFFFF
    result, error = packet.write4ByteTxRx(port, motor_id, address, val_32)
    if result != 0 or error != 0:
        err_msg = packet.getRxPacketError(error) if error != 0 else packet.getTxRxResult(result)
        raise RuntimeError(
            f"Write failed on ID {motor_id}, Addr {address}, Val {val_to_write} (code={error}: {err_msg})"
        )


def set_motion_profile(accel_ms, vel_ms):
    for motor_id in active_motors:
        write_4byte(motor_id, ADDR_PROFILE_ACCELERATION, accel_ms)
        write_4byte(motor_id, ADDR_PROFILE_VELOCITY, vel_ms)


def set_torque(enable: bool, goal_positions: dict):
    if enable:
        for m_id in active_motors:
            write_register(m_id, ADDR_TORQUE_ENABLE, 1)
            curr = read_position(m_id)
            goal_positions[m_id] = curr
            write_4byte(m_id, ADDR_GOAL_POSITION, curr)
    else:
        for m_id in active_motors:
            try:
                write_register(m_id, ADDR_TORQUE_ENABLE, 0)
            except Exception:
                pass


def update_base_limits():
    h11 = HOME_POSITIONS[11]
    JOINT_LIMITS[11] = (h11 - BASE_ROTATION_SPAN, h11 + BASE_ROTATION_SPAN)


def restore_terminal():
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_terminal)


def get_key():
    ready, _, _ = select.select([sys.stdin], [], [], 0.02)
    if ready:
        return sys.stdin.read(1)
    return None


try:
    if not port.openPort():
        raise RuntimeError("Cannot open port on " + DEVICE)

    if not port.setBaudRate(BAUDRATE):
        raise RuntimeError("Cannot set baud rate")

    print("\n=======================================================")
    print("      OPENMANIPULATOR-X TELEOPERATION & CONTROL")
    print("=======================================================")

    active_motors = []
    for m_id in ARM_MOTOR_IDS:
        _, res, _ = packet.ping(port, m_id)
        if res == 0:
            active_motors.append(m_id)
        else:
            raise RuntimeError(f"Arm joint ID {m_id} not detected on bus!")

    _, res, _ = packet.ping(port, GRIPPER_ID)
    gripper_available = (res == 0)
    if gripper_available:
        active_motors.append(GRIPPER_ID)
        print(">> Gripper (ID 15) DETECTED.")
    else:
        print(">> Gripper (ID 15) NOT detected (arm-only mode).")

    goal_positions = {}
    step_counts = DEFAULT_STEP_COUNTS
    torque_enabled = False

    for m_id in active_motors:
        torque = read_register(m_id, ADDR_TORQUE_ENABLE)
        if torque != 0:
            write_register(m_id, ADDR_TORQUE_ENABLE, 0)

    for m_id in active_motors:
        err = read_register(m_id, ADDR_HARDWARE_ERROR)
        if err != 0:
            raise RuntimeError(f"Hardware error on motor {m_id} (Code: {err})")

        curr_mode = read_register(m_id, ADDR_OPERATING_MODE)
        if m_id == GRIPPER_ID:
            motor_modes[m_id] = curr_mode
        else:
            if curr_mode != 4:
                try:
                    write_register(m_id, ADDR_OPERATING_MODE, 4)
                    motor_modes[m_id] = 4
                except Exception:
                    motor_modes[m_id] = curr_mode
            else:
                motor_modes[m_id] = 4

        drive = read_register(m_id, ADDR_DRIVE_MODE)
        if drive != 4:
            write_register(m_id, ADDR_DRIVE_MODE, 4)

        curr = read_position(m_id)
        goal_positions[m_id] = curr

    set_motion_profile(TELEOP_ACCELERATION_MS, TELEOP_VELOCITY_MS)

    print("\nCurrent Motor Positions:")
    for m_id in active_motors:
        label = "Gripper" if m_id == 15 else f"Joint {m_id}"
        print(f"  {label:10s} (ID {m_id:2d}): {goal_positions[m_id]:5d}")

    print("\n-------------------------------------------------------")
    print("SET HOME POSITION FOR BASE (ID 11):")
    print(f"  [1] Use CURRENT position ({goal_positions[11]} counts) as Home")
    print("  [2] Use 2048 (Center) as Home")
    choice = input("Select [1 or 2, default 1]: ").strip()

    if choice == "2":
        HOME_POSITIONS[11] = 2048
        print(">> Home set to 2048.")
    else:
        for m_id in active_motors:
            HOME_POSITIONS[m_id] = goal_positions[m_id]
        print(f">> Current position ({goal_positions[11]}) saved as HOME.")

    update_base_limits()

    print(f"\nID 11 Range configured with respect to Home ({HOME_POSITIONS[11]}):")
    print(f"  Clockwise (CW) limit        : {JOINT_LIMITS[11][0]:5d} counts (-180°)")
    print(f"  Anticlockwise (CCW) limit    : {JOINT_LIMITS[11][1]:5d} counts (+180°)")

    print("""
KEYBOARD CONTROLS:
--------------------------------------------------------
 [Q] / [A] : Joint 11 (Base)     Anticlockwise (CCW) / Clockwise (CW)
 [W] / [S] : Joint 12 (Shoulder) Up / Down
 [E] / [D] : Joint 13 (Elbow)    Up / Down
 [R] / [F] : Joint 14 (Wrist)    Up / Down

 [O]       : Gripper OPEN completely
 [C]       : Gripper CLOSE completely
 [G] / [V] : Gripper jog OPEN / CLOSE

 [T]       : Enable / Disable Torque
 [H]       : Move SLOWLY to HOME position
 [P]       : Save current pose as new HOME
 [+] / [-] : Increase / Decrease step size
 [1/2/3]   : Step presets (Fine=2, Med=15, Fast=50)
 [SPACE]   : Emergency Stop / Freeze motion
 [ESC]     : Exit & disable torque
--------------------------------------------------------
""")

    initial_cmd = input("Type 'ENABLE' to turn on torque now (or Enter to start torque-off): ").strip()
    if initial_cmd.upper() == "ENABLE":
        set_torque(True, goal_positions)
        torque_enabled = True
        print(">> Torque is ON.")
    else:
        print(">> Torque is OFF. Press 'T' anytime to enable.")

    key_map = {
        "q": (11, 1),   "a": (11, -1),
        "w": (12, 1),   "s": (12, -1),
        "e": (13, 1),   "d": (13, -1),
        "r": (14, 1),   "f": (14, -1),
        "g": (15, 1),   "v": (15, -1),
    }

    tty.setcbreak(sys.stdin.fileno())

    while True:
        raw_key = get_key()
        if raw_key is None:
            continue

        key = raw_key.lower()

        if raw_key == "\x1b":
            print("\n[ESC] Exiting...")
            break

        elif key == "t":
            torque_enabled = not torque_enabled
            set_torque(torque_enabled, goal_positions)
            state_str = "ENABLED" if torque_enabled else "DISABLED"
            print(f"\n>> Torque {state_str}.")
            continue

        elif key == "p":
            print("\n>> Saved current pose as new HOME position:")
            for m_id in active_motors:
                HOME_POSITIONS[m_id] = read_position(m_id)
                print(f"   ID {m_id:2d}: {HOME_POSITIONS[m_id]:5d}")
            update_base_limits()
            print(f">> Joint 11 limits updated around {HOME_POSITIONS[11]}: {JOINT_LIMITS[11]}")
            continue

        elif key == "h":
            if not torque_enabled:
                print("\n>> Cannot move to HOME: Torque is OFF. Press 'T' first.")
                continue

            print(f"\n>> Moving SLOWLY to HOME over {HOME_VELOCITY_MS / 1000.0:.1f}s... (SPACE to stop)")
            set_motion_profile(HOME_ACCELERATION_MS, HOME_VELOCITY_MS)

            for m_id in active_motors:
                target = HOME_POSITIONS[m_id]
                goal_positions[m_id] = target
                write_4byte(m_id, ADDR_GOAL_POSITION, target)

            start_time = time.time()
            total_wait = (HOME_VELOCITY_MS / 1000.0) + 0.5
            aborted = False

            while (time.time() - start_time) < total_wait:
                chk = get_key()
                if chk == " ":
                    print("\n>> [ABORT] Stop requested! Freezing position.")
                    for m_id in active_motors:
                        curr = read_position(m_id)
                        goal_positions[m_id] = curr
                        write_4byte(m_id, ADDR_GOAL_POSITION, curr)
                    aborted = True
                    break
                elif chk == "\x1b":
                    print("\n>> [EMERGENCY STOP] Disabling torque!")
                    set_torque(False, goal_positions)
                    torque_enabled = False
                    aborted = True
                    break
                time.sleep(0.05)

            set_motion_profile(TELEOP_ACCELERATION_MS, TELEOP_VELOCITY_MS)
            if not aborted:
                print(">> Reached HOME position smoothly.")
            continue

        elif key in ["o", "c"]:
            if not gripper_available:
                print("\n>> Gripper ID 15 not connected.", end="", flush=True)
                continue
            if not torque_enabled:
                print("\n>> Torque is OFF. Press 'T' first.", end="", flush=True)
                continue

            target = GRIPPER_OPEN_POS if key == "o" else GRIPPER_CLOSE_POS
            goal_positions[GRIPPER_ID] = target
            write_4byte(GRIPPER_ID, ADDR_GOAL_POSITION, target)
            action = "OPENED" if key == "o" else "CLOSED"
            print(f"\r>> Gripper {action} ({target})                               ", end="", flush=True)
            continue

        elif raw_key in ["+", "="]:
            step_counts = min(MAX_STEP_COUNTS, step_counts + 5)
            print(f"\n>> Step size: {step_counts} counts")
            continue

        elif raw_key in ["-", "_"]:
            step_counts = max(MIN_STEP_COUNTS, step_counts - 5)
            print(f"\n>> Step size: {step_counts} counts")
            continue

        elif raw_key == "1":
            step_counts = 2
            print(f"\n>> Step size [FINE]: {step_counts} counts")
            continue

        elif raw_key == "2":
            step_counts = 15
            print(f"\n>> Step size [MEDIUM]: {step_counts} counts")
            continue

        elif raw_key == "3":
            step_counts = 50
            print(f"\n>> Step size [COARSE]: {step_counts} counts")
            continue

        elif raw_key == " ":
            if torque_enabled:
                for m_id in active_motors:
                    goal_positions[m_id] = read_position(m_id)
                    write_4byte(m_id, ADDR_GOAL_POSITION, goal_positions[m_id])
                print("\n>> Position held.")
            continue

        elif key in key_map:
            motor_id, direction = key_map[key]

            if motor_id not in active_motors:
                continue

            if not torque_enabled:
                print("\n>> Torque is OFF. Press 'T' to enable.", end="", flush=True)
                continue

            min_lim, max_lim = JOINT_LIMITS[motor_id]
            curr_goal = goal_positions[motor_id]

            proposed_goal = curr_goal + direction * step_counts
            clamped_goal = max(min_lim, min(max_lim, proposed_goal))

            if clamped_goal == curr_goal:
                bound_label = "CCW (MAX)" if direction > 0 else "CW (MIN)"
                print(f"\r>> ID {motor_id} at {bound_label} limit ({min_lim}..{max_lim})    ", end="", flush=True)
                continue

            goal_positions[motor_id] = clamped_goal
            write_4byte(motor_id, ADDR_GOAL_POSITION, clamped_goal)

            current = read_position(motor_id)
            if motor_id == 11:
                offset = current - HOME_POSITIONS[11]
                deg = offset * (360.0 / 4096.0)
                if abs(offset) <= 2:
                    dir_str = "CENTER"
                    deg_str = "  0.0°"
                else:
                    dir_str = "CCW" if offset > 0 else "CW "
                    deg_str = f"{abs(deg):5.1f}°"

                print(
                    f"\rBase ID 11 | Pos: {current:5d} | Rel Home: {dir_str} {deg_str} ({offset:+5d}) | Step: {step_counts:2d} ",
                    end="",
                    flush=True,
                )
            else:
                print(
                    f"\rID {motor_id:2d} | Present: {current:5d} | Goal: {clamped_goal:5d} | Step: {step_counts:2d}       ",
                    end="",
                    flush=True,
                )

except KeyboardInterrupt:
    print("\nKeyboard interrupt detected.")

except Exception as error:
    print("\nERROR:", error)

finally:
    restore_terminal()
    if port.is_open:
        set_torque(False, {})
        port.closePort()
    print("\nTorque turned OFF. Serial port closed cleanly.")
