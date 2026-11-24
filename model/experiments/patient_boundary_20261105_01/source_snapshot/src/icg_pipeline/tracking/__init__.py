"""Dense optical-flow contour propagation with explicit failure states."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..data import contour_to_points


@dataclass
class TrackedContour:
    points: Optional[np.ndarray]
    status: str
    forward_backward_error: Optional[float]
    valid_point_fraction: float
    frame_idx: int


class OpticalFlowTracker:
    """Track contour samples with dense Farneback flow on laparoscopic frames.

    The forward/backward check and image bounds are conservative quality gates,
    not guarantees that a propagated contour remains anatomically correct.
    Outputs are complete closed contours: any missing sample suppresses the
    contour rather than connecting across an unsupported arc.
    """

    def __init__(
        self,
        num_points: int = 100,
        min_valid_fraction: float = 0.6,
        max_forward_backward_error: float = 2.0,
        max_displacement: float = 50.0,
        min_local_std: float = 2.0,
        min_valid_points: int = 8,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 21,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
    ):
        if num_points < 3:
            raise ValueError("num_points must be at least 3")
        if not 0 < min_valid_fraction <= 1:
            raise ValueError("min_valid_fraction must be in (0, 1]")
        if min_valid_points < 3:
            raise ValueError("min_valid_points must be at least 3")
        self.num_points = num_points
        self.min_valid_fraction = min_valid_fraction
        self.max_forward_backward_error = max_forward_backward_error
        self.max_displacement = max_displacement
        self.min_local_std = min_local_std
        self.min_valid_points = min_valid_points
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.prev_frame: Optional[np.ndarray] = None
        self.points: Optional[np.ndarray] = None
        self.frame_idx = 0

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.ndim != 2:
            raise ValueError(f"Expected grayscale or BGR frame, got shape {frame.shape}")
        return frame

    def initialize(self, frame: np.ndarray, contour: np.ndarray) -> None:
        gray = self._gray(frame)
        points = contour_to_points(contour, self.num_points)
        height, width = gray.shape
        if not np.isfinite(points).all() or (
            (points[:, 0] < 0).any() or (points[:, 0] >= width).any()
            or (points[:, 1] < 0).any() or (points[:, 1] >= height).any()
        ):
            raise ValueError("Initial contour points must be finite and inside the frame")
        self.prev_frame = gray.copy()
        self.points = points.astype(np.float32)
        self.frame_idx = 0

    def correct(self, frame: np.ndarray, contour: np.ndarray) -> None:
        """Correct/reinitialize using a fresh segmentation measurement."""
        self.initialize(frame, contour)

    def track(self, frame: np.ndarray) -> TrackedContour:
        self.frame_idx += 1
        gray = self._gray(frame)
        if self.prev_frame is None or self.points is None:
            return TrackedContour(None, "uninitialized", None, 0.0, self.frame_idx)
        if gray.shape != self.prev_frame.shape:
            self.prev_frame = gray.copy()
            self.points = None
            return TrackedContour(None, "failed_frame_size_changed", None, 0.0, self.frame_idx)

        forward = cv2.calcOpticalFlowFarneback(
            self.prev_frame, gray, None, self.pyr_scale, self.levels,
            self.winsize, self.iterations, self.poly_n, self.poly_sigma, 0,
        )
        backward = cv2.calcOpticalFlowFarneback(
            gray, self.prev_frame, None, self.pyr_scale, self.levels,
            self.winsize, self.iterations, self.poly_n, self.poly_sigma, 0,
        )
        old = self.points
        height, width = gray.shape
        source_x = np.rint(old[:, 0]).astype(np.int32)
        source_y = np.rint(old[:, 1]).astype(np.int32)
        source_inside = (
            (source_x >= 0) & (source_x < width)
            & (source_y >= 0) & (source_y < height)
        )
        displacement = np.zeros_like(old)
        displacement[source_inside] = forward[source_y[source_inside], source_x[source_inside]]
        new = old + displacement
        target_x = np.rint(new[:, 0]).astype(np.int32)
        target_y = np.rint(new[:, 1]).astype(np.int32)
        target_inside = (
            (target_x >= 0) & (target_x < width)
            & (target_y >= 0) & (target_y < height)
        )
        reverse = np.zeros_like(old)
        sample_reverse = source_inside & target_inside
        reverse[sample_reverse] = backward[target_y[sample_reverse], target_x[sample_reverse]]
        error = np.linalg.norm(displacement + reverse, axis=1)
        local_std = np.zeros(len(old), dtype=np.float32)
        for index, (x, y) in enumerate(zip(source_x, source_y)):
            if source_inside[index] and target_inside[index]:
                previous_patch = self.prev_frame[max(0, y - 5):y + 6, max(0, x - 5):x + 6]
                current_patch = gray[
                    max(0, target_y[index] - 5):target_y[index] + 6,
                    max(0, target_x[index] - 5):target_x[index] + 6,
                ]
                local_std[index] = min(float(previous_patch.std()), float(current_patch.std()))
        inside = (
            (new[:, 0] >= 0) & (new[:, 0] < width)
            & (new[:, 1] >= 0) & (new[:, 1] < height)
        )
        valid = (
            source_inside & target_inside
            & (error <= self.max_forward_backward_error)
            & (np.linalg.norm(displacement, axis=1) <= self.max_displacement)
            & (local_std >= self.min_local_std)
            & inside & np.isfinite(new).all(axis=1)
        )
        fraction = float(valid.mean())
        mean_error = float(error[valid].mean()) if valid.any() else None

        if fraction < self.min_valid_fraction or int(valid.sum()) < min(self.min_valid_points, self.num_points):
            self.prev_frame = gray.copy()
            self.points = None
            return TrackedContour(None, "failed_quality_check", mean_error, fraction, self.frame_idx)

        # Downstream drawing and temporal IoU close this point sequence. Removing
        # failed samples would bridge unsupported arcs and could cumulatively
        # shrink the contour while each frame still passed the fraction gate.
        # Until partial/open contours are supported, require every sample and
        # preserve their count and ordering. A trusted new initialization can
        # establish a complete contour again.
        if not valid.all():
            self.prev_frame = gray.copy()
            self.points = None
            return TrackedContour(None, "failed_incomplete_contour", mean_error, fraction, self.frame_idx)

        propagated = new.astype(np.float32)
        self.prev_frame = gray.copy()
        self.points = propagated
        return TrackedContour(propagated, "propagated", mean_error, fraction, self.frame_idx)


from .gated import GatedContourTracker
