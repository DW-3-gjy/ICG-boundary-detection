"""Evaluation metrics for ICG segmentation and tracking."""

import numpy as np
import torch
from typing import Any, Dict, List, Tuple, Optional
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree
import cv2
import logging

logger = logging.getLogger(__name__)


def compute_dice(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Dice coefficient for binary masks."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    intersection = np.logical_and(pred, target).sum()
    union = pred.sum() + target.sum()
    
    return (2.0 * intersection + smooth) / (union + smooth)


def compute_iou(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Intersection over Union (IoU) for binary masks."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    
    return (intersection + smooth) / (union + smooth)


def compute_precision(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute precision for binary masks."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    
    return (tp + smooth) / (tp + fp + smooth)


def compute_recall(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute recall (sensitivity) for binary masks."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    tp = np.logical_and(pred, target).sum()
    fn = np.logical_and(~pred, target).sum()
    
    return (tp + smooth) / (tp + fn + smooth)


def compute_specificity(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute specificity for binary masks."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    tn = np.logical_and(~pred, ~target).sum()
    fp = np.logical_and(pred, ~target).sum()
    
    return (tn + smooth) / (tn + fp + smooth)


def compute_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute pixel accuracy for binary masks."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    correct = np.logical_or(
        np.logical_and(pred, target),
        np.logical_and(~pred, ~target)
    ).sum()
    
    return correct / pred.size


def compute_hausdorff_distance(pred: np.ndarray, target: np.ndarray, percentile: float = 95) -> float:
    """Compute Hausdorff distance between contours.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        percentile: Percentile for robust Hausdorff (default: 95 for HD95)
        
    Returns:
        Hausdorff distance in pixels
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    # Get contour points
    pred_contours = _get_contour_points(pred)
    target_contours = _get_contour_points(target)
    
    if len(pred_contours) == 0 or len(target_contours) == 0:
        return float('inf')
    
    # Compute directed Hausdorff distances
    d1 = directed_hausdorff(pred_contours, target_contours)[0]
    d2 = directed_hausdorff(target_contours, pred_contours)[0]
    
    return max(d1, d2)


def compute_hausdorff_distance_95(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute 95th percentile Hausdorff distance (HD95)."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    pred_contours = _get_contour_points(pred)
    target_contours = _get_contour_points(target)
    
    if len(pred_contours) == 0 or len(target_contours) == 0:
        return float('inf')
    
    # Compute all pairwise distances
    min_dist_pred = cKDTree(target_contours).query(pred_contours)[0]
    min_dist_target = cKDTree(pred_contours).query(target_contours)[0]
    
    hd95_pred = np.percentile(min_dist_pred, 95)
    hd95_target = np.percentile(min_dist_target, 95)
    
    return max(hd95_pred, hd95_target)


def compute_surface_distance(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute surface distance metrics."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    # Get contour points
    pred_contours = _get_contour_points(pred)
    target_contours = _get_contour_points(target)
    
    if len(pred_contours) == 0 or len(target_contours) == 0:
        return {
            'mean_surface_distance': float('inf'),
            'median_surface_distance': float('inf'),
            'std_surface_distance': float('inf'),
            'max_surface_distance': float('inf')
        }
    
    min_dist_pred = cKDTree(target_contours).query(pred_contours)[0]
    min_dist_target = cKDTree(pred_contours).query(target_contours)[0]
    all_distances = np.concatenate([min_dist_pred, min_dist_target])
    
    return {
        'mean_surface_distance': float(all_distances.mean()),
        'median_surface_distance': float(np.median(all_distances)),
        'std_surface_distance': float(all_distances.std()),
        'max_surface_distance': float(all_distances.max())
    }


def _get_contour_points(mask: np.ndarray) -> np.ndarray:
    """Extract contour points from binary mask."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.array([])
    points = np.vstack(contours).reshape(-1, 2)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    return points


def compute_contour_metrics(pred_contour: np.ndarray, target_contour: np.ndarray) -> Dict[str, float]:
    """Compute symmetric nearest-boundary distances in pixels.

    Symmetric distances avoid assuming that independent contours share a starting
    vertex or point-by-point correspondence.
    """
    from ..data import contour_to_points
    from scipy.spatial.distance import cdist

    pred_resampled = contour_to_points(pred_contour, 100)
    target_resampled = contour_to_points(target_contour, 100)
    distances = cdist(pred_resampled, target_resampled)
    nearest_pred = distances.min(axis=1)
    nearest_target = distances.min(axis=0)
    nearest = np.concatenate((nearest_pred, nearest_target))
    return {
        'mean_contour_distance': float(nearest.mean()),
        'median_contour_distance': float(np.median(nearest)),
        'hausdorff_95_contour_distance': float(np.percentile(nearest, 95)),
        'max_contour_distance': float(nearest.max()),
        'std_contour_distance': float(nearest.std())
    }


def evaluate_segmentation(
    pred: np.ndarray, 
    target: np.ndarray,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """Evaluate segmentation with multiple metrics.
    
    Args:
        pred: Predicted mask (H, W) or (H, W, C)
        target: Ground truth mask (H, W)
        metrics: List of metrics to compute (default: all)
        
    Returns:
        Dictionary of metric names and values
    """
    if metrics is None:
        metrics = ['dice', 'iou', 'precision', 'recall', 'specificity', 'accuracy', 
                   'hausdorff', 'hausdorff_95', 'surface_distance']
    
    results = {}
    
    # Ensure binary
    pred_bin = (pred > 0).astype(np.uint8)
    target_bin = (target > 0).astype(np.uint8)
    if pred_bin.shape != target_bin.shape:
        raise ValueError('Prediction and reference must use the same original coordinates')
    
    if 'dice' in metrics:
        results['dice'] = compute_dice(pred_bin, target_bin)
    if 'iou' in metrics:
        results['iou'] = compute_iou(pred_bin, target_bin)
    if 'precision' in metrics:
        results['precision'] = compute_precision(pred_bin, target_bin)
    if 'recall' in metrics:
        results['recall'] = compute_recall(pred_bin, target_bin)
    if 'specificity' in metrics:
        results['specificity'] = compute_specificity(pred_bin, target_bin)
    if 'accuracy' in metrics:
        results['accuracy'] = compute_accuracy(pred_bin, target_bin)
    if 'hausdorff' in metrics:
        results['hausdorff'] = compute_hausdorff_distance(pred_bin, target_bin)
    if 'hausdorff_95' in metrics:
        results['hausdorff_95'] = compute_hausdorff_distance_95(pred_bin, target_bin)
    if 'surface_distance' in metrics:
        results.update(compute_surface_distance(pred_bin, target_bin))
    if 'area_ratio' in metrics:
        reference_area = int(target_bin.sum())
        predicted_area = int(pred_bin.sum())
        results['area_ratio'] = (predicted_area/reference_area if reference_area else
                                 (1.0 if predicted_area == 0 else float('inf')))
    
    return results


def evaluate_tracking(
    tracked_contours: List[np.ndarray],
    ground_truth_contours: List[np.ndarray],
    frame_indices: Optional[List[int]] = None
) -> Dict[str, float]:
    """Evaluate contour tracking performance.

    ``mean_hausdorff_95_contour_distance`` is the arithmetic mean of per-frame
    contour HD95 values for successful pairs, not a pooled sequence percentile.
    
    Args:
        tracked_contours: List of tracked contours per frame
        ground_truth_contours: List of ground truth contours per frame
        frame_indices: Frame indices for reference
        
    Returns:
        Dictionary of tracking metrics
    """
    if len(tracked_contours) != len(ground_truth_contours):
        raise ValueError("Tracked and reference contour sequences must have equal length")
    if frame_indices is None:
        frame_indices = list(range(len(tracked_contours)))
    elif len(frame_indices) != len(tracked_contours):
        raise ValueError("frame_indices must have one entry per contour")
    
    frame_metrics = []
    
    for i, (tracked, gt) in enumerate(zip(tracked_contours, ground_truth_contours)):
        if len(tracked) == 0 or len(gt) == 0:
            frame_metrics.append({
                'frame': frame_indices[i],
                'contour_distance': float('inf'),
                'center_distance': float('inf'),
                'area_ratio': 0.0
            })
            continue
        
        # Compute contour metrics
        contour_metrics = compute_contour_metrics(tracked, gt)
        
        # Compute center distance
        tracked_center = tracked.mean(axis=0)
        gt_center = gt.mean(axis=0)
        center_dist = np.linalg.norm(tracked_center - gt_center)
        
        # Compute area ratio
        tracked_area = cv2.contourArea(tracked.astype(np.float32))
        gt_area = cv2.contourArea(gt.astype(np.float32))
        area_ratio = tracked_area / gt_area if gt_area > 0 else 0.0
        
        frame_metrics.append({
            'frame': frame_indices[i],
            **contour_metrics,
            'center_distance': float(center_dist),
            'area_ratio': float(area_ratio)
        })
    
    # Aggregate metrics
    valid_metrics = [m for m in frame_metrics if m['center_distance'] != float('inf')]
    
    if not valid_metrics:
        return {
            'mean_contour_distance': float('inf'),
            'median_contour_distance': float('inf'),
            'mean_hausdorff_95_contour_distance': float('inf'),
            'mean_center_distance': float('inf'),
            'mean_area_ratio': 0.0,
            'tracking_success_rate': 0.0
        }
    
    return {
        'mean_contour_distance': float(np.mean([m['mean_contour_distance'] for m in valid_metrics])),
        'median_contour_distance': float(np.median([m['median_contour_distance'] for m in valid_metrics])),
        'mean_hausdorff_95_contour_distance': float(np.mean([
            m['hausdorff_95_contour_distance'] for m in valid_metrics])),
        'mean_center_distance': float(np.mean([m['center_distance'] for m in valid_metrics])),
        'mean_area_ratio': float(np.mean([m['area_ratio'] for m in valid_metrics])),
        'tracking_success_rate': len(valid_metrics) / len(frame_metrics)
    }


class Evaluator:
    """Main evaluator class for ICG pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_config = config.get('evaluation', {})
        self.metrics_to_compute = self.metrics_config.get('metrics', [
            'dice', 'iou', 'precision', 'recall', 'hausdorff_distance', 'surface_distance'
        ])
    
    def evaluate_dataset(
        self, 
        predictions: List[Dict[str, Any]], 
        ground_truths: List[np.ndarray]
    ) -> Dict[str, Any]:
        """Evaluate predictions against ground truth for a dataset."""
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have equal length")
        if not predictions:
            raise ValueError("Cannot evaluate an empty dataset")
        all_metrics = []
        
        for pred, gt in zip(predictions, ground_truths):
            pred_mask = pred['mask']
            metrics = evaluate_segmentation(pred_mask, gt, self.metrics_to_compute)
            all_metrics.append(metrics)
        
        # Aggregate
        aggregated = {}
        for key in all_metrics[0].keys():
            all_values = [m[key] for m in all_metrics]
            values = [v for v in all_values if np.isfinite(v)]
            nonfinite = len(all_values) - len(values)
            aggregated[f'nonfinite_count_{key}'] = nonfinite
            if values:
                aggregated[f'mean_{key}'] = float('inf') if nonfinite else float(np.mean(values))
                aggregated[f'std_{key}'] = float('inf') if nonfinite else float(np.std(values))
                aggregated[f'median_{key}'] = float(np.median(all_values))
                aggregated[f'finite_mean_{key}'] = float(np.mean(values))
            else:
                aggregated[f'mean_{key}'] = float('inf')
                aggregated[f'std_{key}'] = 0.0
                aggregated[f'median_{key}'] = float('inf')
        
        return {
            'per_sample': all_metrics,
            'aggregated': aggregated
        }
    
    def evaluate_tracking_sequence(
        self,
        tracked_results: List[Dict[str, Any]],
        ground_truth_contours: List[Optional[np.ndarray]],
        elapsed_seconds: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Evaluate outputs against independent contours, with optional drift bins.

        Pass ``None`` references for unannotated frames. ``elapsed_seconds`` is
        measured from fluorescence disappearance; it is never inferred from
        tracker output.
        """
        if len(tracked_results) != len(ground_truth_contours):
            raise ValueError("One independent reference contour (or None) is required per frame")
        if elapsed_seconds is not None and len(elapsed_seconds) != len(tracked_results):
            raise ValueError("elapsed_seconds must have one entry per frame")
        tracked_contours = [r.get('tracked_contour') for r in tracked_results]
        valid_indices = [
            i for i, (r, gt) in enumerate(zip(tracked_results, ground_truth_contours))
            if r.get('tracked_contour') is not None and gt is not None
            and len(r['tracked_contour']) >= 3 and len(gt) >= 3
        ]
        statuses = [r.get('tracking_status', 'unknown') for r in tracked_results]
        reference_count = sum(gt is not None and len(gt) >= 3 for gt in ground_truth_contours)
        failure_count = sum(
            status.startswith('tracking_failed')
            or status == 'uninitialized_no_reliable_segmentation'
            for status in statuses
        )
        annotated = {
            'tracking_failure_frequency': failure_count / len(statuses) if statuses else 0.0,
            'tracking_failure_count': failure_count,
            'annotated_frame_count': reference_count,
            'evaluated_frame_count': len(valid_indices),
            'distance_metrics_condition': 'successful_annotated_frames_only',
            'frame_count': len(statuses),
        }
        if valid_indices:
            tracked_valid = [tracked_contours[i] for i in valid_indices]
            gt_valid = [ground_truth_contours[i] for i in valid_indices]
            annotated.update(evaluate_tracking(tracked_valid, gt_valid, valid_indices))
        else:
            annotated.update({
                'mean_contour_distance': float('nan'),
                'median_contour_distance': float('nan'),
                'mean_hausdorff_95_contour_distance': float('nan'),
                'mean_center_distance': float('nan'),
                'mean_area_ratio': float('nan'),
            })
        # evaluate_tracking receives successful pairs only; its local success
        # fraction must not erase failed frames with independent references.
        annotated['tracking_success_rate'] = len(valid_indices)/reference_count if reference_count else float('nan')
        if elapsed_seconds is not None:
            drift: Dict[str, List[float]] = {}
            for index in valid_indices:
                key = str(float(elapsed_seconds[index]))
                value = compute_contour_metrics(tracked_contours[index], ground_truth_contours[index])[
                    'mean_contour_distance'
                ]
                drift.setdefault(key, []).append(value)
            annotated['drift_by_elapsed_seconds'] = {
                key: float(np.mean(values))
                for key, values in sorted(drift.items(), key=lambda item: float(item[0]))
            }
        processing_times = [
            float(result['processing_time_ms']) for result in tracked_results
            if result.get('processing_time_ms') is not None
        ]
        if processing_times:
            annotated['mean_processing_time_ms'] = float(np.mean(processing_times))
            annotated['frames_per_second'] = 1000.0 / annotated['mean_processing_time_ms']
        return annotated


def evaluate_source_split(config, split, predictor=None, output_dir=None):
    """Stream original pairs; never reconstruct inputs from normalized tensors."""
    from pathlib import Path
    from ..data import ICGDataset, read_image, write_image
    from ..inference import ICGPredictor
    if split not in ('train', 'val', 'test'):
        raise ValueError(f'Unknown split: {split}')
    data = config['data']
    pairs = ICGDataset(data[f'{split}_data_dir'], is_train=False,
                       image_size=tuple(data['image_size']),
                       channels=data.get('in_channels', 1)).pairs
    predictor = predictor or ICGPredictor(config)
    evaluator = Evaluator(config)
    rows = []
    for image_path, mask_path in pairs:
        image, target = read_image(image_path), read_image(mask_path, grayscale=True)
        if image.shape[:2] != target.shape:
            raise ValueError(f'Image/mask size mismatch: {image_path}')
        prediction = predictor.predict_image(image)
        row = evaluate_segmentation(prediction['mask'], target, evaluator.metrics_to_compute)
        row.update(image_path=image_path.name, image_size_hw=list(target.shape),
                   empty_prediction=not bool(prediction['mask'].any()),
                   empty_reference=not bool(target.any()))
        row['nonfinite_metrics'] = [k for k, v in row.items()
                                    if isinstance(v, (float, np.floating)) and not np.isfinite(v)]
        rows.append(row)
        if output_dir is not None:
            destination = Path(output_dir)
            write_image(destination / 'masks' / (image_path.stem+'.png'), prediction['mask'])
            overlay = image.copy()
            for mask, color in ((target, (0, 255, 0)), (prediction['mask'], (0, 165, 255))):
                contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 2)
            write_image(destination / 'overlays' / (image_path.stem+'.png'), overlay)
    metric_rows = [{k: v for k, v in row.items() if isinstance(v, (float, np.floating))} for row in rows]
    return {'per_sample': rows, 'aggregated': aggregate_metrics(metric_rows), 'sample_count': len(rows),
            'split': split, 'distance_units': 'original_image_pixels',
            'scope': 'retrospective_source_group_holdout_not_independent_patients'}


def aggregate_metrics(rows):
    """Keep failure-aware means and separately labelled finite-only diagnostics."""
    if not rows:
        raise ValueError('Cannot aggregate an empty dataset')
    result = {}
    for key in rows[0]:
        values = np.asarray([r[key] for r in rows], dtype=float)
        finite = values[np.isfinite(values)]
        failures = len(values)-len(finite)
        result[f'nonfinite_count_{key}'] = failures
        result[f'mean_{key}'] = float('inf') if failures else float(values.mean())
        result[f'std_{key}'] = float('inf') if failures else float(values.std())
        result[f'median_{key}'] = float(np.median(values))
        result[f'finite_mean_{key}'] = float(finite.mean()) if len(finite) else None
    return result
