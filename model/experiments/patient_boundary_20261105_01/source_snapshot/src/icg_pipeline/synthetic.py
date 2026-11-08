"""Known-motion sequences from stills; these do not simulate biological ICG decay."""
import cv2
import numpy as np


def generate_sequence(image, mask, scenario='translation', split='train', seed=42, frames=12):
    if scenario not in ('translation', 'rotation', 'deformation', 'brightness', 'occlusion'):
        raise ValueError(f'Unknown synthetic scenario: {scenario}')
    if split not in ('train', 'val', 'test') or frames < 2:
        raise ValueError('A source split and at least two frames are required')
    if image.shape[:2] != mask.shape:
        raise ValueError('Image/mask dimensions must match')
    h, w = mask.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    phase = np.random.default_rng(seed).uniform(-np.pi, np.pi)
    rows = []
    for i in range(frames):
        t = i/(frames-1)
        reference = ((mask > 0)*255).astype(np.uint8)
        frame = image.copy()
        transform = {'scenario': scenario, 'step': i}
        if scenario in ('translation', 'rotation'):
            if scenario == 'translation':
                matrix = np.float32([[1, 0, 2*i], [0, 1, i]])
            else:
                matrix = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), 8*t, 1)
            frame = cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR)
            reference = cv2.warpAffine(reference, matrix, (w, h), flags=cv2.INTER_NEAREST)
            transform['affine_matrix'] = matrix.tolist()
        elif scenario == 'deformation':
            dx = (3*t*np.sin(2*np.pi*yy/h + phase)).astype(np.float32)
            dy = (2*t*np.sin(2*np.pi*xx/w + phase)).astype(np.float32)
            frame = cv2.remap(frame, xx+dx, yy+dy, cv2.INTER_LINEAR)
            reference = cv2.remap(reference, xx+dx, yy+dy, cv2.INTER_NEAREST)
            transform.update(amplitude_xy=[3*t, 2*t], phase=float(phase))
        elif scenario == 'brightness':
            frame = np.clip(frame.astype(np.float32)*(1-.7*t), 0, 255).astype(np.uint8)
            transform['brightness_factor'] = 1-.7*t
        elif scenario == 'occlusion' and 0 < i < frames-1:
            frame[h//4:3*h//4, w//4:3*w//4] = 0
            transform['occluded_rectangle'] = [w//4, h//4, 3*w//4, 3*h//4]
        rows.append({'image': frame, 'mask': reference, 'frame_idx': i, 'source_split': split,
                     'transform': transform, 'synthetic_only': True})
    return rows


def benchmark_synthetic(config, split, output_dir, frames=12, seed=42, initialization='reference'):
    """Reference initialization isolates motion; model initialization tests integration."""
    from pathlib import Path
    import time
    from .data import ICGDataset, read_image, write_image, extract_contours
    from .eval import Evaluator
    from .tracking import GatedContourTracker
    from .utils import write_json
    if initialization not in ('reference', 'model'):
        raise ValueError('initialization must be reference or model')
    predictor = None
    if initialization == 'model':
        from .inference import ICGPredictor
        predictor = ICGPredictor(config)
        if predictor.tracking_policy != 'gated':
            raise ValueError('Model benchmark requires tracking.correction_policy=gated')
    tracking = config.get('tracking', {})
    working_size = tuple(tracking.get('working_size', [288, 512]))
    h, w = working_size
    dataset = ICGDataset(config['data'][f'{split}_data_dir'], is_train=False)
    output = Path(output_dir)
    sequences = []
    for image_path, mask_path in dataset.pairs:
        image = cv2.resize(read_image(image_path), (w, h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(read_image(mask_path, grayscale=True), (w, h), interpolation=cv2.INTER_NEAREST)
        for scenario in ('translation', 'rotation', 'deformation', 'brightness', 'occlusion'):
            sequence = generate_sequence(image, mask, scenario, split, seed, frames)
            tracker = GatedContourTracker(
                working_size, min_probability=tracking.get('min_probability', .75),
                min_temporal_iou=tracking.get('min_temporal_iou', .5),
                num_points=tracking.get('num_points', 100), **tracking.get('optical_flow', {}))
            predictions, references, per_frame = [], [], []
            for row in sequence:
                contours = extract_contours(row['mask'], min_area=1)
                reference = contours[0].reshape(-1, 2) if contours else None
                candidate, confidence = None, None
                if initialization == 'reference' and row['frame_idx'] == 0:
                    candidate, confidence = reference, 1.
                elif predictor is not None:
                    observation = predictor.predict_image(row['image'])
                    candidate, confidence = observation['selected_contour'], observation['segmentation_probability']
                started = time.perf_counter()
                result = tracker.update(row['image'], candidate, confidence)
                result['processing_time_ms'] = (time.perf_counter()-started)*1000
                predictions.append(result)
                references.append(reference)
                per_frame.append(result | {'frame_idx': row['frame_idx'], 'transform': row['transform'],
                                            'reference_contour': reference})
                if result['tracked_contour'] is not None:
                    overlay = row['image'].copy()
                    cv2.polylines(overlay, [np.rint(result['tracked_contour']).astype(np.int32)], True, (0, 165, 255), 1)
                    write_image(output / image_path.stem / scenario / f"{row['frame_idx']:04d}.png", overlay)
            metrics = Evaluator({}).evaluate_tracking_sequence(predictions, references)
            entry = {'source_image': image_path.name, 'source_split': split, 'scenario': scenario,
                     'initialization': initialization, 'synthetic_only': True, 'metrics': metrics}
            sequences.append(entry)
            write_json(output / image_path.stem / scenario / 'frames.json', per_frame)
    report = {'synthetic_only': True, 'distance_units': 'working_image_pixels',
              'working_size_hw': list(working_size), 'sequences': sequences,
              'limitations': 'Known synthetic transformations; brightness change is not biological ICG decay. '
                             'Reference mode uses a manual initial mask, not an automatic segmentation.'}
    write_json(output / 'summary.json', report)
    return report
