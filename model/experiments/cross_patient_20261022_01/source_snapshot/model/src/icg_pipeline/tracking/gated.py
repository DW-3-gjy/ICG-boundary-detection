"""Small correction policy around Farneback; no learned temporal memory."""
import cv2
import numpy as np
from . import OpticalFlowTracker


class GatedContourTracker:
    def __init__(self, working_size=(288, 512), min_probability=.75, min_temporal_iou=.5,
                 num_points=100, **flow_options):
        self.working_size = tuple(working_size)
        self.min_probability = float(min_probability)
        self.min_temporal_iou = float(min_temporal_iou)
        if not (0 <= self.min_probability <= 1 and 0 <= self.min_temporal_iou <= 1):
            raise ValueError('Probability and IoU gates must be in [0, 1]')
        self.flow_options = {'min_valid_fraction': .8, **flow_options, 'num_points': num_points}
        self.reset()

    def reset(self):
        self.flow = OpticalFlowTracker(**self.flow_options)
        self.failed = False
        self.source_shape = None

    def _mask(self, contour):
        mask = np.zeros(self.working_size, np.uint8)
        cv2.fillPoly(mask, [np.rint(contour).astype(np.int32).reshape(-1, 1, 2)], 1)
        return mask

    def update(self, frame, contour=None, confidence=None):
        h, w = frame.shape[:2]
        wh, ww = self.working_size
        scale = np.array([ww/w, wh/h], np.float32)
        image = cv2.resize(frame, (ww, wh), interpolation=cv2.INTER_AREA)
        candidate = None if contour is None else np.asarray(contour, np.float32).reshape(-1, 2)*scale
        trusted = (candidate is not None and len(candidate) >= 3 and np.isfinite(candidate).all()
                   and confidence is not None and np.isfinite(confidence) and confidence >= self.min_probability)
        result = {'tracked_contour': None, 'tracking_status': 'uninitialized_no_reliable_segmentation',
                  'correction_status': 'not_available', 'segmentation_interior_probability': confidence,
                  'temporal_iou': None, 'valid_point_fraction': 0., 'forward_backward_error': None,
                  'working_size_hw': list(self.working_size)}
        if self.failed:
            result['tracking_status'] = 'tracking_failed:latched_until_reset'
            return result
        if self.source_shape is not None and self.source_shape != (h, w):
            self.failed = True
            result['tracking_status'] = 'tracking_failed:frame_size_changed'
            return result
        if self.flow.points is None:
            if trusted:
                self.flow.initialize(image, candidate)
                self.source_shape = (h, w)
                result.update(tracked_contour=self.flow.points/scale,
                              tracking_status='segmentation_initialized', correction_status='initialized',
                              valid_point_fraction=1.)
            return result
        propagated = self.flow.track(image)
        result.update(valid_point_fraction=propagated.valid_point_fraction,
                      forward_backward_error=propagated.forward_backward_error)
        if propagated.points is None:
            self.failed = True
            result['tracking_status'] = f'tracking_failed:{propagated.status}'
            return result
        points = propagated.points
        result.update(tracked_contour=points/scale, tracking_status='propagated')
        if candidate is not None:
            a, b = self._mask(points), self._mask(candidate)
            union = np.logical_or(a, b).sum()
            agreement = float(np.logical_and(a, b).sum()/union) if union else 0.
            result.update(temporal_iou=agreement, correction_status='rejected')
            if trusted and agreement >= self.min_temporal_iou:
                self.flow.correct(image, candidate)
                result.update(tracked_contour=self.flow.points/scale, tracking_status='segmentation_corrected',
                              correction_status='corrected')
        return result
