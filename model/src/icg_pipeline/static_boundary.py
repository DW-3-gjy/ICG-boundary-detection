"""Static D-model candidates, open curves and optional local region correction."""
from functools import lru_cache

import cv2
import numpy as np
import torch

from .boundary_study import MixedBoundaryUNet, normalized_image, skeletonize
from .refinement import curve_geometry, refine_curves


def load_coarse(path, device, input_size):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if 'input_size' in checkpoint and tuple(checkpoint['input_size']) != tuple(input_size):
        raise ValueError('Input dimensions differ from the coarse checkpoint')
    model = MixedBoundaryUNet(boundary_head=True, head_seed=0, encoder_weights=None)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device).eval(), float(checkpoint['threshold'])


@torch.inference_mode()
def coarse_probabilities(model, image, input_size, device):
    reduced = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), input_size,
                         interpolation=cv2.INTER_AREA)
    output = model(normalized_image(reduced)[None].to(device))
    region = output['logits'].float().softmax(dim=1)[0, 1].cpu().numpy()
    boundary = output['boundary_logits'].float().sigmoid()[0].cpu().numpy()
    return region, boundary


def trace_candidates(probability, threshold, image_shape):
    """Trace skeleton links once. Only an actual cycle returns to its first point."""
    if not 0 < threshold < 1:
        raise ValueError('A boundary threshold must be between 0 and 1')
    skeleton = skeletonize(probability >= threshold)
    pixels = set(zip(*np.nonzero(skeleton)))

    def neighbors(point):
        y, x = point
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                other = y + dy, x + dx
                if other == point or other not in pixels:
                    continue
                if dy and dx and ((y + dy, x) in pixels or (y, x + dx) in pixels):
                    continue
                yield other

    links = {p: tuple(neighbors(p)) for p in sorted(pixels)}
    visited = set()

    def walk(start, other):
        path = [start, other]
        visited.add(frozenset((start, other)))
        while len(links[path[-1]]) == 2 and path[-1] != start:
            current = path[-1]
            following = next(p for p in links[current] if p != path[-2])
            link = frozenset((current, following))
            if link in visited:
                break
            visited.add(link)
            path.append(following)
        return path

    paths = []
    for point, adjacent in links.items():
        if not adjacent:
            paths.append([point])
        elif len(adjacent) != 2:
            paths.extend(walk(point, p) for p in adjacent if frozenset((point, p)) not in visited)
    for point, adjacent in links.items():
        paths.extend(walk(point, p) for p in adjacent if frozenset((point, p)) not in visited)

    h, w = image_shape[:2]
    ph, pw = probability.shape
    scale = np.array([w / pw, h / ph], dtype=np.float32)
    curves = [(np.asarray(path, dtype=np.float32)[:, ::-1] + .5) * scale - .5 for path in paths]
    for curve in curves:
        np.clip(curve, (0, 0), (w - 1, h - 1), out=curve)
    return curves


def sampled_curve(curve, step):
    points = np.asarray(curve, dtype=np.float32)
    if len(points) < 2:
        return points.reshape(-1, 2)
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance) > 1e-6]
    samples = np.arange(0, distance[-1], step)
    samples = np.r_[samples, distance[-1]]
    return np.column_stack([np.interp(samples, distance[keep], points[keep, i]) for i in (0, 1)]).astype(np.float32)


def curve_points(curves, step):
    if not curves:
        return np.empty((0, 2), dtype=np.float32)
    points = np.concatenate([sampled_curve(curve, step) for curve in curves])
    return np.unique(points, axis=0)


def base_region_mask(region, image_shape):
    h, w = image_shape[:2]
    return cv2.resize((region >= .5).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


def refine_open_curves(model, image, boundary, curves, step):
    ordered = [sampled_curve(curve, step) for curve in curves if len(curve) > 1]
    result = refine_curves(model, image, boundary, ordered)
    return ordered, result


@lru_cache(maxsize=4)
def disk(radius):
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    return np.maximum(0, 1 - (offsets[:, None]**2 + offsets[None, :]**2) / radius**2)**2


def corrected_region_mask(region, old_mask, curves, refined, config):
    """Move a supported existing region transition towards the refined line.

    The correction is local, uses no annotation, and never fills a curve as a polygon.
    """
    centers, normals, groups = curve_geometry(curves)
    if len(centers) == 0:
        return old_mask.copy()
    radius = config.radius
    valid = np.asarray(refined['accepted'], dtype=bool)
    all_points = np.asarray(refined['points'], dtype=np.float32)
    # Singleton refinements are not emitted as line segments. Ignore them here too.
    groups_valid = []
    cursor = 0
    for count in groups:
        run = valid[cursor:cursor + count]
        points = all_points[cursor:cursor + count]
        linked = run[:-1] & run[1:] & (np.linalg.norm(np.diff(points, axis=0), axis=1) <= config.max_link_distance)
        groups_valid.extend(np.r_[False, linked] | np.r_[linked, False])
        cursor += count
    use = np.asarray(groups_valid) & (np.asarray(refined['position_mass']) >= .20)
    if not use.any():
        return old_mask.copy()
    refined_points = all_points[use]
    centers, normals = centers[use], normals[use]
    h, w = old_mask.shape
    probability = cv2.resize(region, (w, h), interpolation=cv2.INTER_LINEAR)
    search = min(radius, 16)
    offsets = np.arange(-search, search + 1, dtype=np.float32)
    xy = centers[:, None] + normals[:, None] * offsets[None, :, None]
    profiles = np.concatenate([cv2.remap(probability, chunk[..., 0], chunk[..., 1], cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE)
                               for chunk in np.array_split(xy, (len(xy) - 1) // 2048 + 1)])
    new_offsets = ((refined_points - centers) * normals).sum(axis=1)
    edits = []
    for point, normal, profile, ray, new_offset in zip(centers, normals, profiles, xy, new_offsets):
        if np.any(ray < 0) or np.any(ray > (w - 1, h - 1)):
            continue
        if min(profile[0], profile[-1]) > .40 or max(profile[0], profile[-1]) < .60:
            continue
        crossings = np.flatnonzero((profile[:-1] >= .5) != (profile[1:] >= .5))
        if len(crossings) != 1:
            continue
        k = crossings[0]
        old_offset = offsets[k] + (.5 - profile[k]) / (profile[k + 1] - profile[k])
        delta = float(new_offset - old_offset)
        if .5 <= abs(delta) <= min(radius, 12):
            position = np.rint(point + new_offset * normal).astype(int)
            edits.append((position, delta * normal))
    if not edits:
        return old_mask.copy()

    reach = max(6, min(radius, 24))
    kernel = disk(reach)
    anchor = np.array([p for p, _ in edits])
    x0, y0 = np.maximum(anchor.min(0) - reach, 0)
    x1, y1 = np.minimum(anchor.max(0) + reach + 1, (w, h))
    shape = (y1 - y0, x1 - x0)
    weights = np.zeros(shape, np.float32)
    dx, dy = np.zeros_like(weights), np.zeros_like(weights)
    for (x, y), (vx, vy) in edits:
        left, right = max(0, x - reach), min(w, x + reach + 1)
        top, bottom = max(0, y - reach), min(h, y + reach + 1)
        yy, xx = slice(top - y0, bottom - y0), slice(left - x0, right - x0)
        patch = kernel[top - y + reach:bottom - y + reach, left - x + reach:right - x + reach]
        weights[yy, xx] += patch
        dx[yy, xx] += vx * patch
        dy[yy, xx] += vy * patch
    cols, rows = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
    divisor = weights.clip(min=1)
    mask = old_mask.copy()
    roi = mask[y0:y1, x0:x1]
    moved = cv2.remap(old_mask, cols - dx / divisor, rows - dy / divisor,
                      cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    roi[weights > 0] = moved[weights > 0]
    return mask
