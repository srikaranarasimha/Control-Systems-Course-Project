"""Recognize legal chess moves from a webcam and optionally notify an Arduino."""

from __future__ import annotations

import argparse
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

import chess
import cv2
import numpy as np
from numpy.typing import NDArray

from chess_piece_vision import (
    ChessPositionRecognizer,
    LegalMoveDetector,
    model_cache_directory,
)

Point = tuple[int, int]
CameraPrediction = tuple[
    NDArray[np.float32],
    dict[str, NDArray[np.float32]] | None,
]


class SerialReader(Protocol):
    def readline(self) -> bytes: ...


class ArduinoMoveLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.command: str | None = None
        self.command_sent_at: str | None = None
        self.active = False
        self.lock = threading.Lock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._write(
                f"=== Arduino log session started: "
                f"{datetime.now().astimezone().isoformat(timespec='milliseconds')} ==="
            )
            self._write(f"Log file: {self.path.resolve()}")
        except OSError as error:
            raise RuntimeError(f"Could not initialize Arduino log {self.path}: {error}") from error

    def command_sent(self, chess_command: str, motor_command: str) -> None:
        with self.lock:
            if self.active:
                self._write("Move replaced before the previous target was reached.")
                self._write("--- END MOVE LOG ---")
            self.command = f"{chess_command} -> {motor_command}"
            self.command_sent_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
            self.active = False
            self._write(
                f"[{self.command_sent_at}] CAMERA MOVE: {self.command}; "
                f"serial command={motor_command!r}"
            )

    def record(self, line: str) -> None:
        with self.lock:
            if not self.active:
                if self.command is not None and line.startswith("Moving to "):
                    self._write("--- MOVE LOG ---")
                    self._write(f"Command sent: {self.command}")
                    self._write(f"Command time: {self.command_sent_at}")
                    self._write(
                        f"Arduino move started: "
                        f"{datetime.now().astimezone().isoformat(timespec='milliseconds')}"
                    )
                    self._write(
                        "Velocity formula: raw_velocity = "
                        "(position_now - position_previous) * 1000 / elapsed_ms"
                    )
                    self._write(
                        "Filtered velocity formula: filtered_velocity = "
                        "previous_filtered_velocity + "
                        "0.25 * (raw_velocity - previous_filtered_velocity)"
                    )
                    self._write(
                        "Measured velocity is limited to +/-250 potentiometer counts/second."
                    )
                    self._write(
                        "PD formula: P = KP * error; "
                        "D = -KD * filtered_velocity; u = P + D."
                    )
                    self._write(
                        "Target-directed effort: u_toward = u * sign(error). "
                        "PWM cap: pwm_limit = MIN_PWM + (MAX_PWM - MIN_PWM) * "
                        "min(abs(error), DECELERATION_DISTANCE) / DECELERATION_DISTANCE."
                    )
                    self._write(
                        "Command: pwm = constrain(u_toward, MIN_PWM, pwm_limit); "
                        "direction follows target error."
                    )
                    self._write("Units: velocity is potentiometer counts/second; PWM is 0-255.")
                    self._write(
                        "Use Arduino t_ms values for sample timing; bracketed timestamps "
                        "are when the host received each line."
                    )
                    self._write("Arduino output:")
                    self.active = True

            self._write(
                f"[{datetime.now().astimezone().isoformat(timespec='milliseconds')}] "
                f"ARDUINO: {line}"
            )
            if (
                line.startswith("Target reached")
                or line.startswith("Target confirmed")
                or line.startswith("ERROR: move timed out")
                or line == "Stopped"
            ):
                self._write("--- END MOVE LOG ---")
                self.active = False
                self.command = None
                self.command_sent_at = None

    def close(self) -> None:
        with self.lock:
            if self.active:
                self._write("Program stopped before the Arduino reported target completion.")
                self._write("--- END MOVE LOG ---")
                self.active = False
            self._write(
                f"=== Arduino log session ended: "
                f"{datetime.now().astimezone().isoformat(timespec='milliseconds')} ==="
            )

    def _write(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{line}\n")


class ArduinoSerialMonitor:
    def __init__(
        self,
        connection: SerialReader,
        move_logger: ArduinoMoveLogger,
    ) -> None:
        self.connection = connection
        self.move_logger = move_logger
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._read_lines, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stopping.set()
        self.thread.join(timeout=1.0)

    def _read_lines(self) -> None:
        while not self.stopping.is_set():
            raw_line = self.connection.readline()
            if not raw_line:
                continue
            line = raw_line.decode("ascii", errors="replace").rstrip("\r\n")
            print(f"ARDUINO: {line}", flush=True)
            self.move_logger.record(line)


def order_board_corners(corners: list[Point]) -> list[Point]:
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError("Select exactly four board corners.")

    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    if not cv2.isContourConvex(ordered):
        raise ValueError("Select four distinct corners around the board.")
    return [(round(float(x)), round(float(y))) for x, y in ordered]


def board_transform_from_corners(
    corners: list[Point],
    size: int,
) -> NDArray[np.float64]:
    ordered_corners = order_board_corners(corners)
    points = np.asarray(ordered_corners, dtype=np.float32)
    destination = np.asarray(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(points, destination)
    if not np.isfinite(transform).all():
        raise ValueError("Could not calculate perspective correction from those corners.")
    return transform


def select_board_corners(
    capture: cv2.VideoCapture,
    first_frame: NDArray[np.uint8],
    size: int,
) -> tuple[list[Point], NDArray[np.uint8]] | None:
    corners: list[Point] = []
    frame = first_frame
    window = "Manual board setup"
    crop_window = "Perspective crop preview"
    crop_window_open = False
    scale = 1.0

    def on_mouse(event: int, x: int, y: int, *_args: object) -> None:
        del _args
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append((round(x / scale), round(y / scale)))

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.setMouseCallback(window, on_mouse)
        while True:
            scale = min(1.0, 1280 / frame.shape[1])
            display_size = (round(frame.shape[1] * scale), round(frame.shape[0] * scale))
            display = cv2.resize(frame, display_size)
            for index, (x, y) in enumerate(corners):
                point = (round(x * scale), round(y * scale))
                cv2.circle(display, point, 7, (0, 255, 0), -1)
                cv2.putText(
                    display,
                    str(index + 1),
                    (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            if len(corners) < 4:
                instructions = "Click the four board corners in any order"
            else:
                instructions = "Enter=use corners | r=reset | q=quit"
            cv2.putText(
                display,
                instructions,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            cv2.imshow(window, display)
            if len(corners) == 4:
                try:
                    transform = board_transform_from_corners(corners, size)
                    crop = warp_board(frame, transform, size)
                    cv2.imshow(crop_window, crop)
                    crop_window_open = True
                except ValueError:
                    if crop_window_open:
                        cv2.destroyWindow(crop_window)
                        crop_window_open = False
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                return None
            if key == ord("r"):
                corners.clear()
                if crop_window_open:
                    cv2.destroyWindow(crop_window)
                    crop_window_open = False
            elif key in (8, 127) and corners:
                corners.pop()
                if crop_window_open:
                    cv2.destroyWindow(crop_window)
                    crop_window_open = False
            elif key in (10, 13) and len(corners) == 4:
                try:
                    ordered_corners = order_board_corners(corners)
                    board_transform_from_corners(ordered_corners, size)
                    return ordered_corners, frame
                except ValueError:
                    corners.clear()
            ok, next_frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read a live frame during board setup.")
            frame = next_frame
    finally:
        cv2.destroyWindow(window)
        if crop_window_open:
            cv2.destroyWindow(crop_window)


def warp_board(
    frame: NDArray[np.uint8],
    transform: NDArray[np.float64],
    size: int,
) -> NDArray[np.uint8]:
    return cv2.warpPerspective(frame, transform, (size, size))


def open_serial_port(port: str, baud: int):
    import serial

    try:
        connection = serial.Serial(port, baudrate=baud, timeout=0.1, write_timeout=1)
    except serial.SerialException as error:
        if "resource busy" in str(error).lower() or "device busy" in str(error).lower():
            raise RuntimeError(
                f"Could not open Arduino serial port {port}: it is already in use. "
                "Close the Arduino IDE Serial Monitor/Plotter and any other program "
                "connected to this port, then try again."
            ) from error
        raise RuntimeError(f"Could not open Arduino serial port {port}: {error}") from error
    return connection


def motor_command_for_side(side: str) -> str:
    """Map the side that just moved to its configured Arduino target command."""
    commands = {"W": "L", "B": "R"}
    try:
        return commands[side]
    except KeyError as error:
        raise ValueError(f"Unknown chess side: {side!r}") from error


def predict_camera_frame(
    recognizer: ChessPositionRecognizer,
    frame: NDArray[np.uint8],
    corners: list[Point],
    camera_view: Literal["front", "left", "right"] | None,
) -> CameraPrediction:
    if camera_view is None:
        probabilities = recognizer.predict(frame, corners)
        alternative_views = {
            "left": recognizer.predict(frame, corners, camera_view="left"),
            "right": recognizer.predict(frame, corners, camera_view="right"),
        }
        return probabilities, alternative_views

    return recognizer.predict(frame, corners, camera_view=camera_view), None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize chess-piece positions and print/send legal move events."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0).")
    parser.add_argument("--port", help="Optional Arduino serial port, e.g. /dev/ttyACM0 or COM3.")
    parser.add_argument("--baud", type=int, default=9600, help="Serial baud rate (default: 9600).")
    parser.add_argument(
        "--motor-log",
        type=Path,
        default=Path(__file__).resolve().with_name("motor_move_logs.txt"),
        help="Append Arduino logs here (default: motor_move_logs.txt beside this script).",
    )
    parser.add_argument(
        "--side-to-move",
        choices=("W", "B"),
        default="W",
        help="Side expected to make the first move after startup (default: W).",
    )
    parser.add_argument(
        "--fen",
        default=chess.STARTING_FEN,
        help="Starting FEN for the position shown on the board.",
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=model_cache_directory(),
        help="Directory for the downloaded offline recognition models.",
    )
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    if args.baud <= 0:
        raise ValueError("Baud rate must be positive.")
    try:
        initial_board = chess.Board(args.fen)
    except ValueError as error:
        raise ValueError(f"Invalid starting FEN: {error}") from error
    if not initial_board.is_valid():
        raise ValueError("Starting FEN does not describe a valid chess position.")
    initial_board.turn = args.side_to_move == "W"
    initial_board.clear_stack()

    print("Loading offline chess-piece recognition models...")
    recognizer = ChessPositionRecognizer.load(args.model_cache)
    detector = LegalMoveDetector(initial_board)

    backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
    capture = cv2.VideoCapture(args.camera, backend)
    if not capture.isOpened() and backend != cv2.CAP_ANY:
        capture.release()
        print(
            "V4L2 could not open the camera; falling back to OpenCV's default backend.",
            file=sys.stderr,
        )
        capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}.")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    connection = None
    move_logger = ArduinoMoveLogger(args.motor_log) if args.port else None
    serial_monitor = None
    try:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Could not read a frame from the camera.")

        board_size = 800
        print("Select the playable board corners in the live preview.")
        selection = select_board_corners(capture, frame, board_size)
        if selection is None:
            return 0
        corners, frame = selection
        transform = board_transform_from_corners(corners, board_size)
        if args.port:
            connection = open_serial_port(args.port, args.baud)
            assert move_logger is not None
            serial_monitor = ArduinoSerialMonitor(connection, move_logger)
            serial_monitor.start()
            print(
                f"Arduino serial connected on {args.port} at {args.baud} baud; "
                "confirmed moves send L or R.",
                flush=True,
            )
            print(f"Arduino move logs will be saved to {args.motor_log.resolve()}")
        else:
            print(
                "Arduino control is disabled: start with --port <serial-port> "
                "to send camera-detected moves.",
                flush=True,
            )

        print(
            "Monitoring board using piece recognition and legal-move checks. "
            "Arduino status will print below when --port is set. Hold the "
            "configured starting position steady while camera view is "
            "identified. Press q to quit."
        )

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera-inference") as executor:
            pending_prediction: Future[CameraPrediction] | None = None
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Could not read a frame from the camera.")

                if pending_prediction is not None and pending_prediction.done():
                    probabilities, alternative_views = pending_prediction.result()
                    pending_prediction = None
                    move = detector.update(probabilities, alternative_views)
                    if move is not None:
                        side = "W" if not detector.board.turn else "B"
                        command = f"MOVE,{side}"
                        motor_command = motor_command_for_side(side)
                        print(
                            f"{command} -> {motor_command} "
                            f"({move.uci()}; {detector.status})",
                            flush=True,
                        )
                        if connection is not None:
                            assert move_logger is not None
                            move_logger.command_sent(command, motor_command)
                            connection.write(f"{motor_command}\n".encode("ascii"))
                            connection.flush()

                if pending_prediction is None:
                    pending_prediction = executor.submit(
                        predict_camera_frame,
                        recognizer,
                        frame.copy(),
                        corners,
                        detector.camera_view,
                    )

                board = warp_board(frame, transform, board_size)
                for line in range(1, 8):
                    cv2.line(
                        board,
                        (line * board_size // 8, 0),
                        (line * board_size // 8, board_size),
                        (120, 120, 120),
                        1,
                    )
                    cv2.line(
                        board,
                        (0, line * board_size // 8),
                        (board_size, line * board_size // 8),
                        (120, 120, 120),
                        1,
                    )
                for square in detector.candidate_squares:
                    cell = detector.view_cell(square)
                    if cell is None:
                        continue
                    row, col = cell
                    x0, x1 = col * board_size // 8, (col + 1) * board_size // 8
                    y0, y1 = row * board_size // 8, (row + 1) * board_size // 8
                    cv2.rectangle(board, (x0, y0), (x1, y1), (0, 0, 255), 3)
                status = (
                    f"Next: {'W' if not detector.board.turn else 'B'} | "
                    f"{detector.camera_view or 'camera?'} | "
                    f"{detector.status} | q: quit"
                )
                cv2.putText(
                    board,
                    status,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 200, 0),
                    2,
                )
                cv2.imshow("Chess move detector", board)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return 0
    finally:
        capture.release()
        if serial_monitor is not None:
            serial_monitor.close()
        if connection is not None:
            connection.close()
        if move_logger is not None:
            move_logger.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (RuntimeError, ValueError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
