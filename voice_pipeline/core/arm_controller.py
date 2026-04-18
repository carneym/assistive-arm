"""
SO-ARM 101 Controller for the assistive arm voice pipeline.

Controls Feetech STS3215 smart serial servos via USB.
Provides preprogrammed movements like wave, raise arm, and go home.

The SO-ARM 101 has 6 joints:
  - ID 1: Base rotation
  - ID 2: Shoulder
  - ID 3: Elbow
  - ID 4: Wrist pitch
  - ID 5: Wrist roll
  - ID 6: Gripper
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# STS3215 Protocol Constants
STS_HEADER = 0xFF
STS_INST_PING = 0x01
STS_INST_READ = 0x02
STS_INST_WRITE = 0x03
STS_INST_SYNC_WRITE = 0x83
STS_GOAL_POSITION_L = 42
STS_PRESENT_POSITION_L = 56
STS_TORQUE_ENABLE = 40


class Joint(str, Enum):
    """SO-ARM 101 joint names."""
    BASE = "base"
    SHOULDER = "shoulder"
    ELBOW = "elbow"
    WRIST_PITCH = "wrist_pitch"
    WRIST_ROLL = "wrist_roll"
    GRIPPER = "gripper"


@dataclass
class JointConfig:
    """Configuration for a single joint."""
    servo_id: int
    min_pos: int = 0
    max_pos: int = 4095
    home_pos: int = 2048


@dataclass
class Pose:
    """A named pose with joint positions."""
    name: str
    positions: Dict[Joint, int]
    speed: int = 500


# Default joint configurations for SO-ARM 101
# Limits expanded to allow full range of motion
DEFAULT_JOINT_CONFIG: Dict[Joint, JointConfig] = {
    Joint.BASE: JointConfig(servo_id=1, home_pos=2048),
    Joint.SHOULDER: JointConfig(servo_id=2, home_pos=742, min_pos=200, max_pos=3800),
    Joint.ELBOW: JointConfig(servo_id=3, home_pos=3208, min_pos=200, max_pos=3800),
    Joint.WRIST_PITCH: JointConfig(servo_id=4, home_pos=2715),
    Joint.WRIST_ROLL: JointConfig(servo_id=5, home_pos=2051),
    Joint.GRIPPER: JointConfig(servo_id=6, home_pos=1701, min_pos=1000, max_pos=3000),
}


# =============================================================================
# PREPROGRAMMED POSES - Calibrated to your arm's home position
# =============================================================================
# To recalibrate: position arm, run --read-positions, update values below

POSES = {
    "home": Pose(
        name="home",
        positions={
            Joint.BASE: 2021,
            Joint.SHOULDER: 742,
            Joint.ELBOW: 3208,
            Joint.WRIST_PITCH: 2715,
            Joint.WRIST_ROLL: 2051,
            Joint.GRIPPER: 1701,
        },
        speed=500,
    ),
    "raised": Pose(
        name="raised",
        positions={
            Joint.BASE: 2021,
            Joint.SHOULDER: 300,      # Arm raised high
            Joint.ELBOW: 2400,        # Elbow pointing up
            Joint.WRIST_PITCH: 2715,
            Joint.WRIST_ROLL: 2051,
            Joint.GRIPPER: 1701,
        },
        speed=400,
    ),
    "wave_ready": Pose(
        name="wave_ready",
        positions={
            Joint.BASE: 2021,
            Joint.SHOULDER: 400,      # Shoulder raised
            Joint.ELBOW: 2600,        # Arm extended out
            Joint.WRIST_PITCH: 2715,
            Joint.WRIST_ROLL: 2051,
            Joint.GRIPPER: 1850,
        },
        speed=400,
    ),
}


# =============================================================================
# Diagnostic Functions
# =============================================================================

def scan_serial_ports() -> List[Dict[str, str]]:
    """Scan for available serial ports."""
    try:
        import serial.tools.list_ports
        ports = []
        for port in serial.tools.list_ports.comports():
            ports.append({
                "port": port.device,
                "description": port.description,
                "hwid": port.hwid,
            })
        return ports
    except ImportError:
        logger.error("pyserial not installed")
        return []


def find_arm_port(baudrate: int = 1000000) -> Optional[str]:
    """Auto-detect the SO-ARM 101 by scanning ports and pinging servos."""
    ports = scan_serial_ports()
    if not ports:
        return None

    candidates = [p for p in ports if any(x in p["port"].lower() for x in
                  ["usb", "serial", "ttyusb", "ttyacm", "cu.usbserial", "cu.wchusbserial"])]
    if not candidates:
        candidates = ports

    for port_info in candidates:
        port = port_info["port"]
        try:
            import serial
            ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.1)
            ping_packet = bytes([0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB])
            ser.write(ping_packet)
            ser.flush()
            time.sleep(0.01)
            response = ser.read(10)
            ser.close()
            if len(response) >= 6 and response[0] == 0xFF and response[1] == 0xFF:
                return port
        except Exception:
            continue
    return None


@dataclass
class DiagnosticResult:
    """Result of arm diagnostic test."""
    port_found: bool
    port: Optional[str]
    servos_responding: Dict[str, bool]
    all_ok: bool
    message: str


def run_arm_diagnostic(port: Optional[str] = None, baudrate: int = 1000000) -> DiagnosticResult:
    """Run a diagnostic test on the arm."""
    servos = {}

    if port is None:
        port = find_arm_port(baudrate)
        if port is None:
            return DiagnosticResult(
                port_found=False, port=None, servos_responding={},
                all_ok=False, message="No arm found. Check USB connection."
            )

    try:
        import serial
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.1)
    except Exception as e:
        return DiagnosticResult(
            port_found=False, port=port, servos_responding={},
            all_ok=False, message=f"Failed to open port {port}: {e}"
        )

    joint_names = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]
    for servo_id, joint_name in enumerate(joint_names, start=1):
        checksum = (~(servo_id + 0x02 + 0x01)) & 0xFF
        ping_packet = bytes([0xFF, 0xFF, servo_id, 0x02, 0x01, checksum])
        try:
            ser.reset_input_buffer()
            ser.write(ping_packet)
            ser.flush()
            time.sleep(0.005)
            response = ser.read(10)
            servos[joint_name] = len(response) >= 6 and response[0] == 0xFF
        except Exception:
            servos[joint_name] = False
    ser.close()

    responding_count = sum(1 for v in servos.values() if v)
    all_ok = responding_count == 6
    if all_ok:
        message = f"All 6 servos responding on {port}"
    elif responding_count > 0:
        failed = [k for k, v in servos.items() if not v]
        message = f"{responding_count}/6 servos responding. Missing: {', '.join(failed)}"
    else:
        message = f"No servos responding on {port}. Check power."

    return DiagnosticResult(port_found=True, port=port, servos_responding=servos,
                           all_ok=all_ok, message=message)


def print_diagnostic_report(result: DiagnosticResult):
    """Print a formatted diagnostic report."""
    print("\n" + "=" * 50)
    print("  SO-ARM 101 Diagnostic Report")
    print("=" * 50)

    if not result.port_found:
        print(f"\n  Status: FAILED")
        print(f"  {result.message}")
    else:
        status = "OK" if result.all_ok else "PARTIAL"
        print(f"\n  Port: {result.port}")
        print(f"  Status: {status}\n")
        for joint, responding in result.servos_responding.items():
            icon = "[OK]" if responding else "[--]"
            print(f"    {icon} {joint}")
        if not result.all_ok:
            print(f"\n  {result.message}")
    print("\n" + "=" * 50 + "\n")


def read_all_positions(port: str, baudrate: int = 1000000) -> Optional[Dict[str, int]]:
    """Read current positions of all servos."""
    try:
        import serial
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.1)
    except Exception:
        return None

    joint_names = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]
    positions = {}

    for servo_id, joint_name in enumerate(joint_names, start=1):
        params = [STS_PRESENT_POSITION_L, 2]
        packet_len = len(params) + 2
        checksum = (~(servo_id + packet_len + STS_INST_READ + sum(params))) & 0xFF
        packet = bytes([0xFF, 0xFF, servo_id, packet_len, STS_INST_READ] + params + [checksum])

        try:
            ser.reset_input_buffer()
            ser.write(packet)
            ser.flush()
            time.sleep(0.005)
            response = ser.read(10)
            if len(response) >= 8 and response[0] == 0xFF:
                positions[joint_name] = response[5] | (response[6] << 8)
            else:
                positions[joint_name] = None
        except Exception:
            positions[joint_name] = None

    ser.close()
    return positions


def print_positions(port: Optional[str] = None, baudrate: int = 1000000):
    """Print current positions of all servos."""
    if port is None:
        port = find_arm_port(baudrate)
        if port is None:
            print("ERROR: No arm found.")
            return

    positions = read_all_positions(port, baudrate)

    print("\n" + "=" * 50)
    print("  Current Servo Positions")
    print("=" * 50)
    print(f"\n  Port: {port}\n")

    if positions is None:
        print("  ERROR: Could not read positions.")
    else:
        print("  Copy these values to POSES['home'] in arm_controller.py:\n")
        for joint, pos in positions.items():
            if pos is not None:
                print(f"    Joint.{joint.upper()}: {pos},")
            else:
                print(f"    Joint.{joint.upper()}: ERROR,")

    print("\n" + "=" * 50 + "\n")


# =============================================================================
# Servo Controller
# =============================================================================

class FeetechServoController:
    """Low-level controller for Feetech STS3215 servos."""

    def __init__(self, port: str, baudrate: int = 1000000):
        self.port = port
        self.baudrate = baudrate
        self._serial = None

    def connect(self) -> bool:
        try:
            import serial
            self._serial = serial.Serial(
                port=self.port, baudrate=self.baudrate,
                timeout=0.1, write_timeout=0.1,
            )
            logger.info(f"Connected to servo bus on {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    @property
    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def _checksum(self, data: List[int]) -> int:
        return (~sum(data)) & 0xFF

    def _send_packet(self, servo_id: int, instruction: int, params: List[int] = None) -> bool:
        if not self.is_connected:
            return False
        params = params or []
        length = len(params) + 2
        packet = [STS_HEADER, STS_HEADER, servo_id, length, instruction] + params
        packet.append(self._checksum(packet[2:]))
        try:
            self._serial.write(bytes(packet))
            self._serial.flush()
            return True
        except Exception as e:
            logger.error(f"Send failed: {e}")
            return False

    def ping(self, servo_id: int) -> bool:
        self._send_packet(servo_id, STS_INST_PING)
        try:
            response = self._serial.read(6)
            return len(response) >= 6
        except Exception:
            return False

    def set_torque(self, servo_id: int, enable: bool) -> bool:
        return self._send_packet(servo_id, STS_INST_WRITE,
                                [STS_TORQUE_ENABLE, 1 if enable else 0])

    def set_position(self, servo_id: int, position: int, speed: int = 500) -> bool:
        position = max(0, min(4095, position))
        speed = max(0, min(4095, speed))
        pos_l, pos_h = position & 0xFF, (position >> 8) & 0xFF
        speed_l, speed_h = speed & 0xFF, (speed >> 8) & 0xFF
        return self._send_packet(servo_id, STS_INST_WRITE,
                                [STS_GOAL_POSITION_L, pos_l, pos_h, 0, 0, speed_l, speed_h])

    def sync_write_positions(self, positions: Dict[int, Tuple[int, int]]) -> bool:
        if not positions:
            return True
        params = [STS_GOAL_POSITION_L, 6]
        for servo_id, (position, speed) in positions.items():
            position = max(0, min(4095, position))
            speed = max(0, min(4095, speed))
            pos_l, pos_h = position & 0xFF, (position >> 8) & 0xFF
            speed_l, speed_h = speed & 0xFF, (speed >> 8) & 0xFF
            params.extend([servo_id, pos_l, pos_h, 0, 0, speed_l, speed_h])
        return self._send_packet(0xFE, STS_INST_SYNC_WRITE, params)


# =============================================================================
# High-Level Arm Controller
# =============================================================================

class SOArm101:
    """High-level controller for the SO-ARM 101 assistive arm."""

    def __init__(self, port: Optional[str] = None, baudrate: int = 1000000):
        self.joint_config = DEFAULT_JOINT_CONFIG
        self._servo = FeetechServoController(port or "/dev/ttyUSB0", baudrate)
        self._connected = False
        self._enabled = False

    def connect(self) -> bool:
        if not self._servo.connect():
            logger.warning("Arm not connected - simulation mode")
            return False

        all_ok = True
        for joint, config in self.joint_config.items():
            if not self._servo.ping(config.servo_id):
                logger.warning(f"Servo {config.servo_id} ({joint.value}) not responding")
                all_ok = False

        self._connected = all_ok
        if all_ok:
            logger.info("All servos responding")
        return all_ok

    def disconnect(self):
        if self._enabled:
            self.disable()
        self._servo.disconnect()
        self._connected = False

    def enable(self) -> bool:
        if not self._connected:
            return False
        for joint, config in self.joint_config.items():
            self._servo.set_torque(config.servo_id, True)
        self._enabled = True
        logger.info("Arm enabled")
        return True

    def disable(self):
        for joint, config in self.joint_config.items():
            self._servo.set_torque(config.servo_id, False)
        self._enabled = False
        logger.info("Arm disabled")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _clamp(self, joint: Joint, position: int) -> int:
        config = self.joint_config[joint]
        return max(config.min_pos, min(config.max_pos, position))

    def set_joint(self, joint: Joint, position: int, speed: int = 500) -> bool:
        if not self._connected:
            return True
        config = self.joint_config[joint]
        return self._servo.set_position(config.servo_id, self._clamp(joint, position), speed)

    def go_to_pose(self, pose: Pose, blocking: bool = True) -> bool:
        logger.info(f"Moving to pose: {pose.name}")
        if not self._connected:
            if blocking:
                time.sleep(0.5)
            return True

        positions = {}
        for joint, position in pose.positions.items():
            config = self.joint_config[joint]
            positions[config.servo_id] = (self._clamp(joint, position), pose.speed)

        self._servo.sync_write_positions(positions)
        if blocking:
            time.sleep(1.0)
        return True

    def go_home(self) -> bool:
        return self.go_to_pose(POSES["home"])

    def raise_arm(self) -> bool:
        return self.go_to_pose(POSES["raised"])

    def lower_arm(self) -> bool:
        return self.go_home()

    def wave(self) -> bool:
        logger.info("Performing wave")
        self.go_to_pose(POSES["wave_ready"])
        time.sleep(0.3)

        # Get wrist roll home position from POSES
        wrist_home = POSES["home"].positions[Joint.WRIST_ROLL]
        for _ in range(3):
            self.set_joint(Joint.WRIST_ROLL, wrist_home - 400, speed=800)
            time.sleep(0.25)
            self.set_joint(Joint.WRIST_ROLL, wrist_home + 400, speed=800)
            time.sleep(0.25)

        time.sleep(0.2)
        self.go_home()
        return True

    def stop(self):
        logger.warning("EMERGENCY STOP")
        self.disable()


# =============================================================================
# Voice Pipeline Integration
# =============================================================================

@dataclass
class ExecutionResult:
    success: bool
    message: str = ""
    error: Optional[str] = None


class ArmController:
    """Wrapper that integrates SOArm101 with the voice pipeline."""

    def __init__(self, port: Optional[str] = None, enabled: bool = True, run_diagnostic: bool = True):
        self.enabled = enabled
        self._arm: Optional[SOArm101] = None
        self._port = port
        self._diagnostic_result: Optional[DiagnosticResult] = None
        self._is_simulation = False

        if enabled:
            self._initialize(run_diagnostic)

    def _initialize(self, run_diagnostic: bool = True):
        if run_diagnostic:
            self._diagnostic_result = run_arm_diagnostic(self._port)
            print_diagnostic_report(self._diagnostic_result)
            if self._diagnostic_result.port_found and self._diagnostic_result.port:
                self._port = self._diagnostic_result.port
            if not self._diagnostic_result.all_ok:
                logger.warning("Arm diagnostic failed - simulation mode")
                self._is_simulation = True
                return

        self._arm = SOArm101(port=self._port)
        if self._arm.connect():
            self._arm.enable()
            logger.info("Arm controller ready")
            self._is_simulation = False
        else:
            logger.warning("Arm controller in simulation mode")
            self._is_simulation = True

    @property
    def is_simulation(self) -> bool:
        return self._is_simulation

    @property
    def diagnostic_result(self) -> Optional[DiagnosticResult]:
        return self._diagnostic_result

    def execute(self, intent) -> ExecutionResult:
        if not self.enabled or self._arm is None:
            return ExecutionResult(success=True, message="Arm disabled - simulating")

        action = intent.action.value
        try:
            if action == "wave":
                self._arm.wave()
                return ExecutionResult(success=True, message="Wave complete")
            elif action == "raise_arm":
                self._arm.raise_arm()
                return ExecutionResult(success=True, message="Arm raised")
            elif action == "lower_arm":
                self._arm.lower_arm()
                return ExecutionResult(success=True, message="Arm lowered")
            elif action == "go_home":
                self._arm.go_home()
                return ExecutionResult(success=True, message="Returned home")
            elif action == "stop":
                self._arm.stop()
                return ExecutionResult(success=True, message="Stopped")
            else:
                return ExecutionResult(success=True, message=f"{action} simulated")
        except Exception as e:
            logger.error(f"Arm error: {e}")
            return ExecutionResult(success=False, error=str(e))

    def close(self):
        if self._arm is not None:
            self._arm.disconnect()
