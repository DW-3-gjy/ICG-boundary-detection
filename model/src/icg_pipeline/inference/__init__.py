"""Image segmentation and explicitly qualified video contour propagation."""

import json
import logging
import time
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from ..data import contour_to_points, extract_contours, postprocess_mask, preprocess_image, read_image, write_image
from ..models import create_model
from ..tracking import OpticalFlowTracker, GatedContourTracker
from ..utils import get_device, load_checkpoint, load_config

logger = logging.getLogger(__name__)


class ICGPredictor:
    """Predict annotated foreground regions; do not assume they are surgical margins."""

    def __init__(self, config: Dict[str, Any], model: Optional[torch.nn.Module] = None):
        config = copy.deepcopy(config)
        checkpoint = None
        if model is None:
            model_path = config.get('inference', {}).get('model_path')
            if not model_path or not Path(model_path).is_file():
                raise FileNotFoundError(f'Model checkpoint not found: {model_path}')
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            saved = checkpoint.get('config', {})
            if saved:
                for key, value in config.get('model', {}).items():
                    if key in saved['model'] and value != saved['model'][key]:
                        raise ValueError(f'Checkpoint model override is incompatible: {key}')
                config['model'] = copy.deepcopy(saved['model'])
                caller_data = config.setdefault('data', {})
                saved_data = saved['data']
                for key, default in (('normalization', 'legacy'), ('in_channels', saved['model'].get('in_channels', 1)),
                                     ('image_size', [256, 256])):
                    expected = saved_data.get(key, default)
                    if key in caller_data and caller_data[key] != expected:
                        raise ValueError(f'Checkpoint preprocessing override is incompatible: {key}')
                    caller_data[key] = copy.deepcopy(expected)
        self.config = config
        self.device = get_device(config["device"])
        self.model = (model if model is not None else create_model(config, load_pretrained=False)).to(self.device)
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.image_size = tuple(config["data"]["image_size"])
        self.channels = int(config["data"].get("in_channels", config["model"].get("in_channels", 1)))
        self.normalization = config['data'].get('normalization', 'legacy')
        self.min_contour_area = float(config.get("inference", {}).get("min_contour_area", 100))
        self.min_mean_probability = float(config.get("inference", {}).get("min_mean_probability", 0.55))
        self.tracking_enabled = bool(config.get("tracking", {}).get("enabled", True))
        self.output_dir = Path(config["inference"].get("output_dir", "predictions"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_overlay = bool(config["inference"].get("save_overlay", True))
        self.overlay_alpha = float(config["inference"].get("overlay_alpha", 0.5))
        self.tracker: Optional[OpticalFlowTracker] = None
        self.tracking_policy = config.get('tracking', {}).get('correction_policy', 'legacy')
        if self.tracking_policy not in ('legacy', 'gated'):
            raise ValueError(f'Unknown correction policy: {self.tracking_policy}')
        self.gated_tracker = None

    @torch.inference_mode()
    def predict_image(self, image: np.ndarray) -> Dict[str, Any]:
        original_size = image.shape[:2]
        tensor = preprocess_image(image, self.image_size, self.channels, self.normalization).to(self.device)
        logits = self.model(tensor)
        if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 2:
            raise ValueError(f"Expected one two-class output, got {tuple(logits.shape)}")
        probabilities = logits.softmax(dim=1)[0, 1]
        mask = postprocess_mask(logits, original_size)
        contours = extract_contours(mask, self.min_contour_area)
        probability_map = torch.nn.functional.interpolate(
            probabilities[None, None],
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()
        ranked = []
        native = probabilities.cpu().numpy()
        for contour in contours:
            if self.tracking_policy == 'gated':
                native_contour = contour.astype(np.float32).copy()
                native_contour[..., 0] *= native.shape[1]/original_size[1]
                native_contour[..., 1] *= native.shape[0]/original_size[0]
                interior = np.zeros(native.shape, np.uint8)
                cv2.drawContours(interior, [np.rint(native_contour).astype(np.int32)], -1, 1, -1)
                interior = (interior > 0) & (native >= .5)
                score = float(native[interior].mean()) if interior.any() else 0.
            else:
                score = float(probability_map[contour[:, 0, 1], contour[:, 0, 0]].mean())
            ranked.append((contour, score))
        accepted = next((contour for contour, score in ranked if score >= self.min_mean_probability), None)
        return {
            "mask": mask,
            "probability_map": probability_map,
            "contours": contours,
            "contour_points": [contour_to_points(c) for c in contours],
            "selected_contour": contour_to_points(accepted) if accepted is not None else None,
            "segmentation_probability": next(
                (score for contour, score in ranked if contour is accepted), None
            ),
            "original_size": original_size,
        }

    def create_overlay(self, image: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Use distinct colors and captions for segmentations, propagation, and failure."""
        overlay = image.copy()
        if overlay.ndim == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        points = result.get("tracked_contour")
        status = result.get("tracking_status", "segmentation_only")
        if points is not None:
            cv2.polylines(overlay, [np.round(points).astype(np.int32).reshape(-1, 1, 2)],
                          True, (255, 160, 0) if status == "propagated" else (0, 220, 0), 2)
        elif (result.get("selected_contour") is not None and not status.startswith("tracking_failed")
              and status != 'uninitialized_no_reliable_segmentation'):
            cv2.drawContours(overlay, [result["selected_contour"].reshape(-1, 1, 2).astype(np.int32)],
                             -1, (0, 220, 0), 2)
        if status.startswith("tracking_failed"):
            cv2.putText(overlay, "TRACKING FAILED - boundary not displayed", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif status == "uninitialized_no_reliable_segmentation":
            cv2.putText(overlay, "WAITING FOR RELIABLE INITIALIZATION", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        elif status == "propagated":
            cv2.putText(overlay, "PROPAGATED - not a new observation", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 160, 0), 2)
        elif result.get("selected_contour") is not None:
            cv2.putText(overlay, "SEGMENTATION-SUPPORTED", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
        if self.save_overlay:
            original = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return cv2.addWeighted(original, 1 - self.overlay_alpha, overlay, self.overlay_alpha, 0)
        return overlay

    def predict_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        track_contours: bool = True,
    ) -> Dict[str, Any]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and fps > 0 else 25.0
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError(f"Video reports invalid dimensions: {video_path}")
        writer = None
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                cap.release()
                raise OSError(f"Could not create output video: {destination}")

        self.tracker = None
        self.gated_tracker = None
        results: List[Dict[str, Any]] = []
        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                started = time.perf_counter()
                result = self.predict_image(frame)
                contour = result["selected_contour"]
                track_this = track_contours and self.tracking_enabled
                if track_this and self.tracking_policy == 'gated':
                    if self.gated_tracker is None:
                        tracking_config = self.config.get('tracking', {})
                        self.gated_tracker = GatedContourTracker(
                            working_size=tracking_config.get('working_size', [288, 512]),
                            min_probability=tracking_config.get('min_probability', .75),
                            min_temporal_iou=tracking_config.get('min_temporal_iou', .5),
                            num_points=tracking_config.get('num_points', 100),
                            **tracking_config.get('optical_flow', {}))
                    result.update(self.gated_tracker.update(frame, contour, result.get('segmentation_probability')))
                elif track_this:
                    if contour is not None:
                        if self.tracker is None:
                            tracking_config = self.config.get("tracking", {})
                            flow_config = tracking_config.get("optical_flow", {})
                            self.tracker = OpticalFlowTracker(
                                num_points=int(tracking_config.get("num_points", 100)),
                                min_valid_fraction=float(flow_config.get("min_valid_fraction", 0.6)),
                                max_forward_backward_error=float(
                                    flow_config.get("max_forward_backward_error", 2.0)
                                ),
                                max_displacement=float(flow_config.get("max_displacement", 50.0)),
                                min_local_std=float(flow_config.get("min_local_std", 2.0)),
                                min_valid_points=int(flow_config.get("min_valid_points", 8)),
                                pyr_scale=float(flow_config.get("pyr_scale", 0.5)),
                                levels=int(flow_config.get("levels", 3)),
                                winsize=int(flow_config.get("winsize", 21)),
                                iterations=int(flow_config.get("iterations", 3)),
                                poly_n=int(flow_config.get("poly_n", 5)),
                                poly_sigma=float(flow_config.get("poly_sigma", 1.2)),
                            )
                        self.tracker.correct(frame, contour)
                        result["tracked_contour"] = contour
                        result["tracking_status"] = "segmentation_supported"
                        result["valid_point_fraction"] = 1.0
                        result["forward_backward_error"] = None
                    elif self.tracker is not None:
                        tracked = self.tracker.track(frame)
                        result["tracked_contour"] = tracked.points
                        result["tracking_status"] = (
                            "propagated" if tracked.points is not None
                            else f"tracking_failed:{tracked.status}"
                        )
                        result["valid_point_fraction"] = tracked.valid_point_fraction
                        result["forward_backward_error"] = tracked.forward_backward_error
                    else:
                        result["tracked_contour"] = None
                        result["tracking_status"] = "uninitialized_no_reliable_segmentation"
                        result["valid_point_fraction"] = 0.0
                        result["forward_backward_error"] = None
                else:
                    result["tracked_contour"] = contour
                    result["tracking_status"] = "segmentation_supported" if contour is not None else "no_segmentation"
                    result["valid_point_fraction"] = None
                    result["forward_backward_error"] = None
                result.update({
                    "frame_idx": frame_idx,
                    "processing_time_ms": (time.perf_counter() - started) * 1000,
                })
                if writer is not None:
                    writer.write(self.create_overlay(frame, result))
                # Do not retain source-resolution masks/probabilities for an entire video.
                results.append({k: v for k, v in result.items() if k not in
                                {'mask', 'probability_map', 'contours', 'contour_points', 'selected_contour'}})
                frame_idx += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        mean_ms = float(np.mean([r["processing_time_ms"] for r in results])) if results else 0.0
        summary = {
            "frames": results,
            "frame_count": frame_idx,
            "mean_processing_time_ms": mean_ms,
            "mean_fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
            "output_video": str(output_path) if output_path else None,
        }
        if output_path is not None:
            metadata_path = Path(output_path).with_suffix(".json")
        else:
            metadata_path = self.output_dir / f"{Path(video_path).stem}_tracking.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            key: value for key, value in summary.items() if key != "frames"
        }
        serializable["frames"] = [
            {key: value for key, value in row.items()
             if key not in {"mask", "probability_map", "contours", "contour_points", "selected_contour"}}
            | {"tracked_contour": row["tracked_contour"].tolist() if row.get("tracked_contour") is not None else None}
            for row in results
        ]
        metadata_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        summary["metadata_path"] = str(metadata_path)
        return summary


def run_inference(
    config_path: str,
    input_path: str,
    output_path: Optional[str] = None,
    config_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = config_override if config_override is not None else load_config(config_path)
    predictor = ICGPredictor(config)
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".m4v"}:
        output = output_path or str(predictor.output_dir / f"{path.stem}_overlay.mp4")
        return predictor.predict_video(str(path), output)
    image = read_image(path)
    result = predictor.predict_image(image)
    result["tracking_status"] = "segmentation_supported" if result["selected_contour"] is not None else "no_segmentation"
    mask_path = predictor.output_dir / f"{path.stem}_mask.png"
    write_image(mask_path, result["mask"])
    if predictor.save_overlay or output_path:
        overlay_path = Path(output_path) if output_path else predictor.output_dir / f"{path.stem}_overlay.png"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        write_image(overlay_path, predictor.create_overlay(image, result))
        result["overlay_path"] = str(overlay_path)
    result["mask_path"] = str(mask_path)
    return result
