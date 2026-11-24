"""Mixed polygon/visible-line supervision, patient sampling and boundary metrics.

Open curves never define a foreground region. Boundary negatives are valid only
after confirmation that all requested visible boundaries have been annotated.
The resulting metrics describe the delivered reference, including polygon
instrument/contact segments; they do not establish amodal anatomical accuracy.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F

from .models.resnet_unet import ResNetUNet
from .training import TverskyCELoss


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mask_boundary(mask):
    """One-pixel interior contour; include all components and holes."""
    mask = np.asarray(mask, np.uint8)
    return mask - cv2.erode(mask, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)


def rasterize_shapes(document, width, height):
    region = np.zeros((height, width), np.uint8)
    boundary = np.zeros_like(region)
    kinds = set()
    for shape in document.get('shapes', []):
        kind = shape.get('shape_type', 'polygon')
        if kind not in ('polygon', 'linestrip'):
            raise ValueError(f'Unsupported annotation: {kind}')
        points = np.asarray(shape['points'], np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError('Invalid points')
        if len(points) < (3 if kind == 'polygon' else 2):
            raise ValueError('Insufficient vertices')
        points = np.rint(points).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        kinds.add(kind)
        if kind == 'polygon':
            cv2.fillPoly(region, [points], 1)
        else:
            cv2.polylines(boundary, [points], False, 1, 1, cv2.LINE_8)
    if not kinds:
        raise ValueError('No supervised shapes; unannotated images are not background')
    if kinds == {'polygon'}:
        boundary = mask_boundary(region)
    elif kinds != {'linestrip'}:
        raise ValueError('Mixed polygon and line semantics require explicit review')
    if not boundary.any():
        raise ValueError('Empty boundary reference')
    return region, boundary, kinds == {'polygon'}


def reference_points(boundary):
    y, x = np.nonzero(boundary)
    return np.column_stack((x, y)).astype(np.float32)


def reduced_boundary(boundary, size):
    # Max-pooling preserves thin lines that nearest-neighbour shrinking can erase.
    tensor = torch.from_numpy(boundary.astype(np.float32))[None, None]
    return F.adaptive_max_pool2d(tensor, (size[1], size[0]))[0, 0].numpy().astype(np.uint8)


def skeletonize(binary):
    """Morphological skeleton at network resolution, identical for all heads."""
    image = np.asarray(binary, np.uint8).copy()
    skeleton = np.zeros_like(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while image.any():
        eroded = cv2.erode(image, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        opened = cv2.dilate(eroded, kernel)
        skeleton |= image & (1 - opened)
        image = eroded
    return skeleton


def prediction_points(probability, threshold, source, original_size):
    binary = (probability >= threshold).astype(np.uint8)
    if source == 'region':
        binary = mask_boundary(binary)
    elif source != 'boundary':
        raise ValueError(source)
    binary = skeletonize(binary)
    points = reference_points(binary)
    # Coordinates are original 4K units, with a declared native-grid sampling.
    h, w = probability.shape
    points[:, 0] = (points[:, 0] + .5) * original_size[0] / w - .5
    points[:, 1] = (points[:, 1] + .5) * original_size[1] / h - .5
    return points


class BoundaryReference:
    def __init__(self, points, original_size):
        self.points = np.asarray(points, np.float32)
        self.original_size = tuple(original_size)
        if not len(self.points):
            raise ValueError('No reference points')
        self.tree = cKDTree(self.points)

    def score(self, prediction, tolerance=20, distances=False):
        if not len(prediction):
            result = dict(precision=0., recall=0., f1=0., empty_prediction=True)
            if distances:
                result.update(hd95=None, assd=None, directed_reference_mean=None,
                              directed_reference_p95=None)
            return result
        dp = self.tree.query(prediction, workers=1)[0]
        dr = cKDTree(prediction).query(self.points, workers=1)[0]
        precision, recall = float(np.mean(dp <= tolerance)), float(np.mean(dr <= tolerance))
        result = dict(precision=precision, recall=recall,
                      f1=2 * precision * recall / (precision + recall) if precision + recall else 0.,
                      empty_prediction=False)
        if distances:
            result.update(hd95=float(max(np.percentile(dp, 95), np.percentile(dr, 95))),
                          assd=float((dp.sum() + dr.sum()) / (len(dp) + len(dr))),
                          directed_reference_mean=float(dr.mean()),
                          directed_reference_p95=float(np.percentile(dr, 95)))
        return result


def train_records(records, group, validation_fold=None):
    result = []
    for row in records:
        if row['planned_partition'] != 'development':
            continue
        if validation_fold and row['planned_validation_fold'] == validation_fold:
            continue
        if not group['include_old_training'] and row['batch'] != '1102':
            continue
        if not group['include_l_training'] and row['task_stratum'] != 'Z_label':
            continue
        result.append(row)
    if not result:
        raise ValueError('No training data')
    return result


class CachedTrainingData:
    def __init__(self, study, rows, seed):
        self.rows = rows
        self.rng = np.random.default_rng(seed)
        self.by_patient = {}
        self.arrays = {}
        for row in rows:
            if row['planned_partition'] != 'development':
                raise ValueError('Test image in training loader')
            sample = row['sample_id']
            with np.load(Path(study) / 'development' / (sample + '.npz'), allow_pickle=False) as data:
                self.arrays[sample] = {k: data[k] for k in ['image', 'region', 'boundary', 'has_region']}
            self.by_patient.setdefault(row['patient_source_id'], []).append(sample)
        self.patients = sorted(self.by_patient)

    def sample(self):
        patient = self.patients[int(self.rng.integers(len(self.patients)))]
        samples = self.by_patient[patient]
        sample = samples[int(self.rng.integers(len(samples)))]
        data = self.arrays[sample]
        image, region, boundary = data['image'].copy(), data['region'].copy(), data['boundary'].copy()
        h, w = region.shape
        if self.rng.random() < .5:
            image, region, boundary = (cv2.flip(a, 1) for a in (image, region, boundary))
        angle = self.rng.uniform(-10, 10)
        matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1.)
        image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        region = cv2.warpAffine(region, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        boundary = cv2.warpAffine(boundary, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        image = (image.astype(np.float32) / 255 - .5) * self.rng.uniform(.9, 1.1) + .5
        image = np.clip(image + self.rng.uniform(-.1, .1), 0, 1)
        image = (image - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        return (torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
                torch.from_numpy(region.astype(np.int64)),
                torch.from_numpy(boundary.astype(np.float32)),
                torch.tensor(bool(data['has_region'])))

    def batch(self, size, device):
        return tuple(torch.stack(items).to(device) for items in zip(*(self.sample() for _ in range(size))))


class MixedBoundaryUNet(ResNetUNet):
    def __init__(self, boundary_head, head_seed, encoder_weights='IMAGENET1K_V1'):
        super().__init__(encoder_weights=encoder_weights, freeze_encoder_batchnorm=True)
        self.boundary_head_enabled = bool(boundary_head)
        if boundary_head:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(head_seed)
                self.boundary_head = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 1))

    def forward(self, image):
        features = self.forward_features(image)
        return {'logits': self.output(features),
                'boundary_logits': self.boundary_head(features)[:, 0] if self.boundary_head_enabled else None}


def soft_contour(probability):
    image = probability[:, None]
    return (F.max_pool2d(image, 3, 1, 1) + F.max_pool2d(-image, 3, 1, 1))[:, 0]


def partial_boundary_loss(probability, line):
    positive = F.max_pool2d(line[:, None], 3, 1, 1)[:, 0] > 0
    uncertainty = F.max_pool2d(line[:, None], 5, 1, 2)[:, 0] > 0
    negative = ~uncertainty
    probability = probability.float().clamp(1e-5, 1 - 1e-5)
    pos_loss = -probability.log()
    neg_loss = -(1 - probability).log()
    dims = (1, 2)
    pos_count, neg_count = positive.sum(dims), negative.sum(dims)
    loss = ((pos_loss * positive).sum(dims) / pos_count.clamp_min(1)
            + (neg_loss * negative).sum(dims) / neg_count.clamp_min(1)) / 2
    valid = (pos_count > 0) & (neg_count > 0)
    return loss[valid].mean() if valid.any() else probability.sum() * 0


class MixedBoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.region = TverskyCELoss(tversky_weight=.5, ce_weight=.4, boundary_weight=.1)

    def forward(self, output, region, boundary, has_region):
        logits = output['logits']
        # Region loss is never evaluated on an open-curve image.
        region_loss = self.region(logits[has_region], region[has_region]) if has_region.any() else logits.sum() * 0
        contour = soft_contour(logits.float().softmax(1)[:, 1])
        partial = partial_boundary_loss(contour, boundary)
        total = region_loss + .1 * partial
        head_loss, consistency = logits.sum() * 0, logits.sum() * 0
        if output['boundary_logits'] is not None:
            head = output['boundary_logits'].float().sigmoid()
            head_loss = partial_boundary_loss(head, boundary)
            # Only confirmed positives and negatives, excluding uncertain band.
            dilated = F.max_pool2d(boundary[:, None], 3, 1, 1)[:, 0] > 0
            outer = F.max_pool2d(boundary[:, None], 5, 1, 2)[:, 0] > 0
            valid = dilated | ~outer
            consistency = ((head - contour).abs() * valid).sum() / valid.sum().clamp_min(1)
            total = total + .1 * head_loss + .05 * consistency
        return total, {'region': float(region_loss.detach()), 'contour': float(partial.detach()),
                       'head': float(head_loss.detach()), 'consistency': float(consistency.detach())}


def normalized_image(array):
    image = array.astype(np.float32) / 255
    image = (image - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


def patient_macro(per_image, key):
    patients = {}
    for row in per_image:
        if row[key] is not None:
            patients.setdefault(row['patient'], []).append(row[key])
    means = {p: float(np.mean(v)) for p, v in patients.items()}
    return {'mean': float(np.mean(list(means.values()))) if means else None, 'by_patient': means}
