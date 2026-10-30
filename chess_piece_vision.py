"""Offline chess-position recognition and legal move confirmation."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections import Counter, deque
from pathlib import Path
from typing import Literal, Protocol
from urllib.request import urlopen

import chess
import cv2
import numpy as np
from numpy.typing import NDArray

BoardImage = NDArray[np.uint8]
Probabilities = NDArray[np.float32]

MODEL_REVISION = "4690fd5418ff7c405a76d1c156077ecd865ac053"
MODEL_FILES = {
    "board_photo_occupancy.onnx": (
        6_094_010,
        "dcafa86708036f1c50d2471bcfef838a3973e63bc6e3eef08dd3adccf6e0067d",
    ),
    "board_photo_pieces.onnx": (
        6_135_013,
        "79f7833e4fd802eb10de4d2765bb243ec51c40bb5097989af9ff4db789f2a8af",
    ),
}
PIECE_CLASSES = "bknpqrBKNPQR"
MIN_MOVE_EVIDENCE = 2.5
MIN_MOVE_SCORE_GAP = 0.35
MIN_UNCHANGED_SQUARE_ACCURACY = 0.60
ORIENTATION_CONFIRMATIONS = 3


class TensorInfo(Protocol):
    name: str


class InferenceSession(Protocol):
    def get_inputs(self) -> list[TensorInfo]: ...

    def get_outputs(self) -> list[TensorInfo]: ...

    def run(
        self,
        output_names: list[str],
        input_feed: dict[str, NDArray[np.float32]],
    ) -> list[NDArray[np.float32]]: ...


def model_cache_directory() -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "chess_move_detector" / "models"


def _download_model(filename: str, cache_dir: Path) -> Path:
    expected_size, expected_hash = MODEL_FILES[filename]
    path = cache_dir / filename
    url = (
        "https://huggingface.co/cstr/chess-board-photo-onnx/resolve/"
        f"{MODEL_REVISION}/{filename}"
    )
    temporary_path: Path | None = None
    try:
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if path.stat().st_size == expected_size and digest == expected_hash:
                return path

        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            digest = hashlib.sha256()
            size = 0
            with urlopen(url, timeout=30) as response:
                while chunk := response.read(64 * 1024):
                    temporary.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)

        if size != expected_size or digest.hexdigest() != expected_hash:
            raise RuntimeError(
                f"Downloaded model {filename} did not match its published "
                "size and SHA-256 checksum."
            )
        temporary_path.replace(path)
    except OSError as error:
        raise RuntimeError(
            f"Could not download chess recognition model {filename}: {error}"
        ) from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def _softmax(logits: NDArray[np.float32]) -> Probabilities:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return (exponentials / exponentials.sum(axis=1, keepdims=True)).astype(
        np.float32
    )


def _to_tensor(crops: list[NDArray[np.uint8]]) -> NDArray[np.float32]:
    images = np.stack(crops).astype(np.float32) / 255.0
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    normalized = (images - mean) / std
    return np.ascontiguousarray(normalized.transpose(0, 3, 1, 2))


def _warp_with_margin(
    image: BoardImage,
    corners: list[tuple[int, int]],
    margin: int,
    output_size: int,
) -> BoardImage:
    source = np.asarray(corners, dtype=np.float32)
    destination = np.asarray(
        (
            (margin, margin),
            (margin + 400, margin),
            (margin + 400, margin + 400),
            (margin, margin + 400),
        ),
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, transform, (output_size, output_size))


def _occupancy_crops(warped: BoardImage) -> list[NDArray[np.uint8]]:
    return [
        warped[
            50 * row + 25 : 50 * row + 125,
            50 * col + 25 : 50 * col + 125,
        ]
        for row in range(8)
        for col in range(8)
    ]


def _occupancy_crops_for_view(
    warped: BoardImage,
    camera_view: Literal["front", "left", "right"],
) -> list[NDArray[np.uint8]]:
    if camera_view == "front":
        return _occupancy_crops(warped)

    camera_right = camera_view == "right"
    rotated_crops = _occupancy_crops(_rotate_for_side_view(warped, camera_right))
    crops = []
    for index in range(64):
        row, col = _side_crop_coordinates(index // 8, index % 8, camera_right)
        crops.append(rotated_crops[row * 8 + col])
    return crops


def _piece_crop(warped: BoardImage, row: int, col: int) -> NDArray[np.uint8]:
    height_increase = 1 + 2 * ((7 - row) / 7)
    left_increase = 0 if col >= 4 else 0.25 + 0.75 * ((3 - col) / 3)
    right_increase = 0 if col < 4 else 0.25 + 0.75 * ((col - 4) / 3)
    x1 = int(200 + 50 * (col - left_increase))
    x2 = int(200 + 50 * (col + 1 + right_increase))
    y1 = int(200 + 50 * (row - height_increase))
    y2 = int(200 + 50 * (row + 1))
    crop = warped[y1:y2, x1:x2]
    if col < 4:
        crop = cv2.flip(crop, 1)
    output = np.zeros((200, 100, 3), dtype=np.uint8)
    output[200 - crop.shape[0] :, : crop.shape[1]] = crop
    return output


def _side_piece_crops(
    warped: BoardImage,
    camera_right: bool,
) -> list[NDArray[np.uint8]]:
    rotated = _rotate_for_side_view(warped, camera_right)
    return [
        _piece_crop(
            rotated,
            *_side_crop_coordinates(index // 8, index % 8, camera_right),
        )
        for index in range(64)
    ]


def _rotate_for_side_view(warped: BoardImage, camera_right: bool) -> BoardImage:
    rotation = (
        cv2.ROTATE_90_CLOCKWISE
        if camera_right
        else cv2.ROTATE_90_COUNTERCLOCKWISE
    )
    return cv2.rotate(warped, rotation)


def _side_crop_coordinates(
    row: int,
    col: int,
    camera_right: bool,
) -> tuple[int, int]:
    return (col, 7 - row) if camera_right else (7 - col, row)


class ChessPositionRecognizer:
    def __init__(
        self,
        occupancy_session: InferenceSession,
        piece_session: InferenceSession,
    ) -> None:
        self._occupancy_session = occupancy_session
        self._piece_session = piece_session
        self._occupancy_input = occupancy_session.get_inputs()[0].name
        self._occupancy_output = occupancy_session.get_outputs()[0].name
        self._piece_input = piece_session.get_inputs()[0].name
        self._piece_output = piece_session.get_outputs()[0].name

    @classmethod
    def load(cls, cache_dir: Path | None = None) -> ChessPositionRecognizer:
        try:
            import onnxruntime
        except ImportError as error:
            raise RuntimeError(
                "ONNX Runtime is missing. Install the project dependencies with "
                "`python -m pip install -r requirements.txt`."
            ) from error

        directory = model_cache_directory() if cache_dir is None else cache_dir
        occupancy_path = _download_model("board_photo_occupancy.onnx", directory)
        pieces_path = _download_model("board_photo_pieces.onnx", directory)
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = min(4, os.cpu_count() or 1)
        return cls(
            onnxruntime.InferenceSession(str(occupancy_path), sess_options=options),
            onnxruntime.InferenceSession(str(pieces_path), sess_options=options),
        )

    def predict(
        self,
        frame: BoardImage,
        corners: list[tuple[int, int]],
        camera_view: Literal["front", "left", "right"] = "front",
    ) -> Probabilities:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        occupancy_board = _warp_with_margin(rgb, corners, margin=50, output_size=500)
        occupancy_logits = self._occupancy_session.run(
            [self._occupancy_output],
            {
                self._occupancy_input: _to_tensor(
                    _occupancy_crops_for_view(occupancy_board, camera_view)
                )
            },
        )[0]
        occupancy_probabilities = _softmax(occupancy_logits)

        piece_board = _warp_with_margin(rgb, corners, margin=200, output_size=800)
        if camera_view != "front":
            camera_right = camera_view == "right"
            piece_crops = _side_piece_crops(piece_board, camera_right)
        else:
            piece_crops = [
                _piece_crop(piece_board, index // 8, index % 8)
                for index in range(64)
            ]
        logits = self._piece_session.run(
            [self._piece_output],
            {self._piece_input: _to_tensor(piece_crops)},
        )[0]
        piece_probabilities = _softmax(logits)

        probabilities = np.zeros((64, 13), dtype=np.float32)
        probabilities[:, 0] = occupancy_probabilities[:, 0]
        probabilities[:, 1:] = (
            occupancy_probabilities[:, 1:2] * piece_probabilities
        )
        return probabilities


class LegalMoveDetector:
    def __init__(
        self,
        initial_board: chess.Board,
        confirmations: int = 3,
        history_size: int = 15,
    ) -> None:
        if not 1 <= confirmations <= history_size:
            raise ValueError("Confirmations must be between 1 and the history size.")
        self.board = initial_board.copy()
        self.confirmations = confirmations
        self._votes: deque[str | None] = deque(maxlen=history_size)
        self._orientation_votes: deque[tuple[int, str]] = deque(maxlen=5)
        self._position_history: deque[Probabilities] = deque(maxlen=3)
        self._move_diagnostic = "no legal-move evidence"
        self.candidate_squares: list[str] = []
        self.orientation: int | None = None
        self.camera_view: Literal["front", "left", "right"] | None = None
        self.status = "Detecting board orientation"

    def update(
        self,
        probabilities: Probabilities,
        alternative_views: dict[str, Probabilities] | None = None,
    ) -> chess.Move | None:
        if probabilities.shape != (64, 13):
            raise ValueError("Position probabilities must have shape (64, 13).")
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise ValueError("Position probabilities must be finite and non-negative.")
        views = {"front": probabilities}
        if alternative_views is not None:
            views.update(alternative_views)
        for name, view_probabilities in views.items():
            if name not in ("front", "left", "right"):
                raise ValueError(f"Unsupported camera view: {name}")
            if view_probabilities.shape != (64, 13):
                raise ValueError("Position probabilities must have shape (64, 13).")
            if (
                not np.isfinite(view_probabilities).all()
                or np.any(view_probabilities < 0)
            ):
                raise ValueError("Position probabilities must be finite and non-negative.")

        if self.orientation is None or self.camera_view is None:
            if alternative_views is None:
                orientation = infer_board_orientation(probabilities, self.board)
                if orientation is None:
                    self.status = (
                        "Board orientation is ambiguous; check the FEN and corners"
                    )
                    return None
                self.orientation = orientation
                self.camera_view = "front"
            else:
                view = infer_camera_view(views, self.board)
                if view is not None:
                    self._orientation_votes.append(view)

            if self.camera_view is None:
                counts = Counter(self._orientation_votes)
                if not counts:
                    self.status = (
                        "Board orientation is ambiguous; check the FEN and corners"
                    )
                    return None
                candidate_view, votes = counts.most_common(1)[0]
                if votes < ORIENTATION_CONFIRMATIONS:
                    self.status = (
                        f"Confirming camera orientation "
                        f"({votes}/{ORIENTATION_CONFIRMATIONS} frames)"
                    )
                    return None
                self.orientation, self.camera_view = candidate_view
                self._votes.clear()
                self._position_history.clear()
                self.status = (
                    f"Board orientation detected "
                    f"({self.orientation * 90} degrees, {self.camera_view} view)"
                )

        selected_view = views.get(self.camera_view, probabilities)
        chess_probabilities = to_chess_orientation(
            selected_view,
            self.orientation,
        )
        self._position_history.append(chess_probabilities)
        history_average = np.mean(
            np.stack(self._position_history),
            axis=0,
            dtype=np.float32,
        )
        smoothed_probabilities = (
            0.65 * chess_probabilities + 0.35 * history_average
        ).astype(np.float32)
        candidate = self._best_move(smoothed_probabilities)
        if candidate is None:
            self._votes.append(None)
            self.candidate_squares = []
            self.status = (
                f"Waiting for a confident legal move ({self._move_diagnostic})"
            )
            return None

        move, squares, evidence = candidate
        self.candidate_squares = [
            chess.square_name(square ^ 56) for square in squares
        ]
        self._votes.append(move.uci())
        count = Counter(self._votes)[move.uci()]
        if count < self.confirmations:
            self.status = (
                f"Candidate {move.uci()} (evidence {evidence:.2f}; "
                f"{count}/{self.confirmations} frames in the last "
                f"{self._votes.maxlen})"
            )
            return None

        self.board.push(move)
        self._votes.clear()
        self.candidate_squares = []
        self.status = f"Move confirmed: {move.uci()}"
        return move

    def view_cell(self, square: str) -> tuple[int, int] | None:
        if self.orientation is None:
            return None
        chess_square = chess.parse_square(square)
        view_index = grid_index_for(chess_square ^ 56, self.orientation)
        return divmod(view_index, 8)

    def _best_move(
        self, probabilities: Probabilities
    ) -> tuple[chess.Move, list[int], float] | None:
        expected_before = _board_labels(self.board)
        expected_before_array = np.asarray(expected_before)
        occupied_probability = np.clip(
            probabilities[:, 1:].sum(axis=1),
            1e-8,
            1.0 - 1e-8,
        )
        unchanged = np.ones(64, dtype=bool)
        candidates: list[tuple[float, chess.Move, list[int], float]] = []
        for move in self.board.legal_moves:
            next_board = self.board.copy()
            next_board.push(move)
            expected_after = _board_labels(next_board)
            changed = [
                index
                for index, (before, after) in enumerate(
                    zip(expected_before, expected_after)
                )
                if before != after
            ]
            if not changed:
                continue
            unchanged[:] = True
            unchanged[changed] = False
            stable_labels = probabilities[unchanged].argmax(axis=1)
            stable_accuracy = np.mean(
                stable_labels == expected_before_array[unchanged]
            )

            expected = np.asarray([expected_after[index] for index in changed])
            prior = np.asarray([expected_before[index] for index in changed])
            observed = probabilities[changed]
            confidence = observed[np.arange(len(changed)), expected]
            old_confidence = observed[np.arange(len(changed)), prior]
            class_evidence = float(
                (
                    np.log(np.maximum(confidence, 1e-8))
                    - np.log(np.maximum(old_confidence, 1e-8))
                ).sum()
            )
            expected_occupied = np.asarray(
                [expected_after[index] != 0 for index in changed]
            )
            previously_occupied = np.asarray(
                [expected_before[index] != 0 for index in changed]
            )
            observed_occupancy = occupied_probability[changed]
            occupancy_evidence = float(
                (
                    np.log(
                        np.where(
                            expected_occupied,
                            observed_occupancy,
                            1.0 - observed_occupancy,
                        )
                    )
                    - np.log(
                        np.where(
                            previously_occupied,
                            observed_occupancy,
                            1.0 - observed_occupancy,
                        )
                    )
                ).sum()
            )
            evidence = occupancy_evidence + 0.25 * (
                class_evidence - occupancy_evidence
            )
            candidates.append((evidence, move, changed, float(stable_accuracy)))

        if not candidates:
            self._move_diagnostic = "no legal-move candidates"
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        viable = [
            candidate
            for candidate in candidates
            if candidate[0] >= MIN_MOVE_EVIDENCE
            and candidate[3] >= MIN_UNCHANGED_SQUARE_ACCURACY
        ]
        if not viable:
            evidence, move, _, accuracy = candidates[0]
            self._move_diagnostic = (
                f"best {move.uci()}: evidence {evidence:.2f}/"
                f"{MIN_MOVE_EVIDENCE:.1f}, board match {accuracy:.0%}/"
                f"{MIN_UNCHANGED_SQUARE_ACCURACY:.0%}"
            )
            return None
        if len(viable) > 1 and viable[0][0] - viable[1][0] < MIN_MOVE_SCORE_GAP:
            self._move_diagnostic = (
                f"ambiguous between {viable[0][1].uci()} and "
                f"{viable[1][1].uci()}"
            )
            return None
        self._move_diagnostic = "evidence accepted"
        return viable[0][1], viable[0][2], viable[0][0]


def _board_labels(board: chess.Board) -> list[int]:
    labels = [0] * 64
    for square, piece in board.piece_map().items():
        symbol = piece.symbol()
        labels[square ^ 56] = PIECE_CLASSES.index(symbol) + 1
    return labels


def grid_index_for(square: int, rotation: int) -> int:
    row, col = divmod(square, 8)
    rotation %= 4
    if rotation == 0:
        return row * 8 + col
    if rotation == 2:
        return (7 - row) * 8 + (7 - col)
    if rotation == 1:
        return (7 - col) * 8 + row
    return col * 8 + (7 - row)


def to_chess_orientation(
    probabilities: Probabilities,
    rotation: int,
) -> Probabilities:
    chess_probabilities = np.empty_like(probabilities)
    for square in range(64):
        view_index = grid_index_for(square, rotation)
        chess_probabilities[square] = probabilities[view_index]
    return chess_probabilities


def infer_board_orientation(
    probabilities: Probabilities,
    board: chess.Board,
    minimum_score_gap: float = 2.0,
) -> int | None:
    expected = _board_labels(board)
    scores: list[tuple[float, int]] = []
    for rotation in range(4):
        score = _orientation_score(probabilities, expected, rotation)
        scores.append((score, rotation))

    scores.sort(reverse=True)
    if scores[0][0] - scores[1][0] < minimum_score_gap:
        return None
    return scores[0][1]


def infer_camera_view(
    views: dict[str, Probabilities],
    board: chess.Board,
    minimum_score_gap: float = 2.0,
) -> tuple[int, str] | None:
    expected = _board_labels(board)
    candidates: list[tuple[float, int, str]] = []
    for camera_view, rotations in (
        ("front", (0, 2)),
        ("right", (1,)),
        ("left", (3,)),
    ):
        probabilities = views.get(camera_view)
        if probabilities is None:
            continue
        for rotation in rotations:
            candidates.append(
                (
                    _orientation_score(probabilities, expected, rotation),
                    rotation,
                    camera_view,
                )
            )

    if len(candidates) < 2:
        return None
    candidates.sort(reverse=True)
    if candidates[0][0] - candidates[1][0] < minimum_score_gap:
        return None
    return candidates[0][1], candidates[0][2]


def _orientation_score(
    probabilities: Probabilities,
    expected_labels: list[int],
    rotation: int,
) -> float:
    return sum(
        float(
            np.log(
                max(
                    probabilities[grid_index_for(square, rotation), label],
                    1e-8,
                )
            )
        )
        for square, label in enumerate(expected_labels)
    )
