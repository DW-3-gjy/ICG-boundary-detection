"""Native-resolution open-interface refinement: bounded displacement + rejection.

Coordinates are original-image (x, y) pixel centers. Labels are explicitly
255=visible target, 0=reviewed negative, 128=unknown. No region filling occurs.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RefinerConfig:
    radius: int = 32
    tangent_radius: int = 16
    hidden: int = 24
    batch_size: int = 128
    reject_threshold: float = 0.5
    max_link_distance: float = 12.0

    def __post_init__(self):
        if self.radius < 1 or self.tangent_radius < 1 or self.hidden < 1 or self.batch_size < 1:
            raise ValueError('Refiner dimensions must be positive')
        if not 0 <= self.reject_threshold <= 1 or self.max_link_distance <= 0:
            raise ValueError('Invalid rejection threshold or link distance')


class BoundaryStripRefiner(nn.Module):
    """Four-channel RGB/coarse-probability strip -> offsets plus reject class."""
    def __init__(self, config=None):
        super().__init__()
        self.config = config or RefinerConfig()
        h = self.config.hidden
        self.features = nn.Sequential(
            nn.Conv2d(4, h, 3, padding=1), nn.ReLU(),
            nn.Conv2d(h, h, 3, padding=1), nn.ReLU(),
            nn.Conv2d(h, h, 3, padding=1), nn.ReLU())
        self.offset = nn.Conv1d(h, 1, 1)
        self.reject = nn.Linear(h, 1)

    def forward(self, strips):
        features = self.features(strips)
        # Strip height indexes signed normal offsets; width indexes tangent.
        offsets = self.offset(features.mean(-1)).squeeze(1)
        return torch.cat([offsets, self.reject(features.mean((2, 3)))], dim=1)


def validate_inputs(image, probability, curves):
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('image must be uint8 BGR HxWx3')
    probability = np.asarray(probability, dtype=np.float32)
    if probability.ndim != 2 or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('probability must be finite HxW in [0, 1]')
    if probability.shape != image.shape[:2]:
        probability = cv2.resize(probability, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    checked = []
    for curve in curves:
        p = np.asarray(curve, dtype=np.float32)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
            raise ValueError('Each curve needs at least two finite (x,y) points')
        if np.any(p < 0) or np.any(p[:, 0] > image.shape[1]-1) or np.any(p[:, 1] > image.shape[0]-1):
            raise ValueError('Curve outside original image')
        checked.append(p)
    return probability, checked


def curve_geometry(curves):
    centers, normals, groups = [], [], []
    for curve in curves:
        ids = np.arange(len(curve))
        tangents = curve[np.minimum(ids+1, len(curve)-1)] - curve[np.maximum(ids-1, 0)]
        lengths = np.linalg.norm(tangents, axis=1)
        normal = np.column_stack([-tangents[:, 1], tangents[:, 0]]) / np.maximum(lengths[:, None], 1e-8)
        centers.append(curve)
        normals.append(normal)
        groups.append(len(curve))
    if not centers:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), []
    return np.concatenate(centers), np.concatenate(normals), groups


def sample_strips(image, probability, centers, normals, config, device='cpu'):
    """Bilinear native RGB/probability sampling, align_corners=True pixel centers."""
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)/255
    source = torch.from_numpy(np.concatenate([rgb, probability[..., None]], axis=-1).transpose(2, 0, 1)).unsqueeze(0).to(device)
    c = torch.as_tensor(centers, device=device, dtype=torch.float32)
    n = torch.as_tensor(normals, device=device, dtype=torch.float32)
    tangent = torch.stack([n[:, 1], -n[:, 0]], -1)
    offsets = torch.arange(-config.radius, config.radius+1, device=device, dtype=torch.float32)
    along = torch.arange(-config.tangent_radius, config.tangent_radius+1, device=device, dtype=torch.float32)
    xy = c[:, None, None, :] + offsets[None, :, None, None]*n[:, None, None, :] + along[None, None, :, None]*tangent[:, None, None, :]
    grid = torch.stack([2*xy[..., 0]/max(width-1, 1)-1, 2*xy[..., 1]/max(height-1, 1)-1], -1)
    return F.grid_sample(source.expand(len(c), -1, -1, -1), grid, padding_mode='border', align_corners=True)


def make_targets(labels, centers, normals, config):
    """Conservative local ray matches; reject only explicitly negative centers.

    Multiple separated positive runs along a ray are ambiguous and ignored.
    Unknown centers never become rejection examples merely for lacking a match.
    """
    if labels.ndim != 2 or not np.isin(labels, [0, 128, 255]).all():
        raise ValueError('Labels must be 2D tri-state 0/128/255')
    result = np.full(len(centers), -100, dtype=np.int64)
    offsets = np.arange(-config.radius, config.radius+1)
    for i, (center, normal) in enumerate(zip(centers, normals)):
        if np.linalg.norm(normal) < 0.5:
            continue
        xy = np.rint(center + offsets[:, None]*normal).astype(int)
        valid = (xy[:, 0]>=0)&(xy[:, 0]<labels.shape[1])&(xy[:, 1]>=0)&(xy[:, 1]<labels.shape[0])
        positive = np.zeros(len(offsets), dtype=bool)
        positive[valid] = labels[xy[valid, 1], xy[valid, 0]] == 255
        indices = np.flatnonzero(positive)
        if len(indices):
            if np.any(np.diff(indices)>1):
                continue
            result[i] = indices[np.argmin(np.abs(offsets[indices]))]
        else:
            x, y = np.rint(center).astype(int)
            if 0<=x<labels.shape[1] and 0<=y<labels.shape[0] and labels[y, x] == 0:
                result[i] = 2*config.radius+1
    return result


def split_fragments(points, keep, max_link_distance):
    """Never close or link across a rejected point, invalid sample or large jump."""
    output, current = [], []
    for point, accepted in zip(points, keep):
        if not accepted or (current and np.linalg.norm(point-current[-1])>max_link_distance):
            if len(current)>=2:
                output.append(np.asarray(current, dtype=np.float32))
            current = []
        if accepted:
            current.append(point)
    if len(current)>=2:
        output.append(np.asarray(current, dtype=np.float32))
    return output


@torch.inference_mode()
def refine_curves(model, image, probability, curves):
    probability, curves = validate_inputs(image, probability, curves)
    c, n, groups = curve_geometry(curves)
    config = model.config
    if not len(c):
        return {'fragments': [], 'points': [], 'accepted': [], 'reject_probability': [], 'candidate_count': 0}
    model.eval()
    device = next(model.parameters()).device
    logits = []
    for start in range(0, len(c), config.batch_size):
        logits.append(model(sample_strips(image, probability, c[start:start+config.batch_size], n[start:start+config.batch_size], config, device)).cpu())
    probabilities = torch.cat(logits).softmax(-1).numpy()
    reject = probabilities[:, -1]
    offsets = probabilities[:, :-1].argmax(-1)-config.radius
    points = c + offsets[:, None]*n
    keep = (reject < config.reject_threshold)&(np.linalg.norm(n, axis=1)>.5)
    keep &= (points[:, 0]>=0)&(points[:, 0]<image.shape[1])&(points[:, 1]>=0)&(points[:, 1]<image.shape[0])
    fragments, start = [], 0
    for count in groups:
        fragments.extend(split_fragments(points[start:start+count], keep[start:start+count], config.max_link_distance))
        start += count
    return {'fragments': [p.tolist() for p in fragments], 'points': points.tolist(), 'accepted': keep.tolist(), 'reject_probability': reject.tolist(), 'candidate_count': len(c)}


def save_refiner(path, model, metadata=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'format_version': 1, 'config': asdict(model.config), 'state_dict': model.state_dict(), 'metadata': metadata or {}}, path)


def load_refiner(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('format_version') != 1:
        raise ValueError('Unsupported refiner checkpoint format')
    model = BoundaryStripRefiner(RefinerConfig(**state['config']))
    model.load_state_dict(state['state_dict'])
    return model.to(device).eval()


def load_manifest(path):
    """JSON list with split/patient/provenance; paths relative to manifest."""
    path = Path(path).resolve()
    rows = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('Manifest must be a nonempty JSON list')
    patient_splits = {}
    for source in rows:
        if not {'patient_id', 'split', 'image', 'probability', 'curves', 'labels', 'coarse_train_patients'} <= source.keys():
            raise ValueError('Missing manifest fields')
        patient, split = source['patient_id'], source['split']
        if not isinstance(patient, str) or not patient or split not in {'train', 'val'}:
            raise ValueError('Only train/val with nonempty string patient IDs allowed')
        trained = source['coarse_train_patients']
        if not isinstance(trained, list) or not all(isinstance(p, str) for p in trained):
            raise ValueError('coarse_train_patients must explicitly list patient IDs')
        if patient in trained:
            raise ValueError('Coarse prediction leakage: patient used to train coarse model')
        if patient in patient_splits and patient_splits[patient] != split:
            raise ValueError('Patient overlap between train and val')
        patient_splits[patient] = split
        for key in ('image', 'probability', 'curves', 'labels'):
            source[key] = str((path.parent/source[key]).resolve())
    val_patients = {p for p, split in patient_splits.items() if split == 'val'}
    if any(val_patients.intersection(row['coarse_train_patients']) for row in rows):
        raise ValueError('Outer validation patient used in a coarse training fold')
    if set(patient_splits.values()) != {'train', 'val'}:
        raise ValueError('Both patient-disjoint train and val required')
    return rows


def read_sample(row):
    image = cv2.imread(row['image'], cv2.IMREAD_COLOR)
    labels = cv2.imread(row['labels'], cv2.IMREAD_GRAYSCALE)
    if image is None or labels is None or image.shape[:2] != labels.shape:
        raise ValueError('Unreadable or mismatched native image/labels')
    probability = np.load(row['probability'], allow_pickle=False)
    curves = json.loads(Path(row['curves']).read_text(encoding='utf-8'))
    probability, curves = validate_inputs(image, probability, curves)
    return image, probability, curves, labels
