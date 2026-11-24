"""Review references returned in the private Z-stage annotation workspace."""
from __future__ import annotations

import argparse
import base64
import binascii
from collections import Counter
import csv
from pathlib import Path

import cv2
import numpy as np

from evaluate_static_d_locked_regions import prepare_frames
from static_d_baseline import LOGS, ORIGINAL, digest, exclusive_training, load_runner, read_json, verify_file

WORKSPACE = LOGS / 'dice90_reference_intake_20261124_01' / 'annotation_workspace.private'
DRAFT = 'independent_review_draft.private.csv'


def workspace_path(workspace, relative):
    path = (workspace / relative).resolve()
    if not path.is_relative_to(workspace):
        raise ValueError('Annotation pairing refers outside its workspace')
    return path


def check_polygon(path, image, width, height):
    doc = read_json(path)
    if not isinstance(doc, dict):
        raise ValueError('Expected a LabelMe annotation object')
    if Path(doc.get('imagePath') or image.name).name != image.name:
        raise ValueError('LabelMe imagePath points to a different file')
    if (int(doc.get('imageWidth', width)), int(doc.get('imageHeight', height))) != (width, height):
        raise ValueError('LabelMe dimensions differ from the paired original')
    shapes = doc.get('shapes')
    if not isinstance(shapes, list) or not shapes:
        raise ValueError('No completed polygons')
    limit = np.array([width, height])
    for shape in shapes:
        if not isinstance(shape, dict) or shape.get('shape_type', 'polygon') != 'polygon':
            raise ValueError('Only closed LabelMe polygons define a full region')
        points = np.asarray(shape.get('points'), dtype=np.float32)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                or not np.isfinite(points).all() or np.any(points < 0) or np.any(points > limit)
                or cv2.contourArea(points) <= 0):
            raise ValueError('A polygon has invalid or zero-area original-image coordinates')
    if doc.get('imageData'):
        encoded = base64.b64decode(doc['imageData'], validate=True)
        embedded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        actual = cv2.imdecode(np.fromfile(image, np.uint8), cv2.IMREAD_COLOR)
        if embedded is None or not np.array_equal(embedded, actual):
            raise ValueError('The embedded LabelMe image differs from the paired original')


def audit(runner, workspace):
    workspace = workspace.resolve(strict=True)
    with (workspace / 'annotation_pairs.private.csv').open(encoding='utf-8-sig', newline='') as file:
        planned = list(csv.DictReader(file))
    if not planned or len({row['sample_id'] for row in planned}) != len(planned):
        raise ValueError('The pairing table is empty or has duplicate sample IDs')
    if len({row['image_sha256'] for row in planned}) != len(planned):
        raise ValueError('The pairing table has duplicate source images')
    ready, observations = [], []
    for pair in planned:
        image = workspace_path(workspace, pair['image_for_annotation'])
        json_path = workspace_path(workspace, pair['expected_labelme_json'])
        if json_path != image.with_suffix('.json'):
            raise ValueError('The pairing table must link a same-stem LabelMe file')
        mask = workspace_path(workspace, image.with_name(f'{image.stem}_mask.png'))
        try:
            verify_file(image, pair['image_sha256'])
            for reply in (json_path, mask):
                if reply.exists() and not reply.is_file():
                    raise ValueError('A reference path is not a file')
            refs = [(p, kind) for p, kind in ((json_path, 'labelme_polygon'), (mask, 'region_mask')) if p.is_file()]
            if not refs:
                observations.append((image.name, 'pending', 'Awaiting a full reference'))
                continue
            if len(refs) != 1:
                raise ValueError('Both LabelMe and PNG exist; select the intended reference')
            path, kind = refs[0]
            reference_hash = digest(path)
            width, height = int(pair['width']), int(pair['height'])
            if kind == 'labelme_polygon':
                check_polygon(path, image, width, height)
            row = {'sample': pair['sample_id'], 'patient': pair['patient_id'], 'kind': kind,
                   'image': str(image), 'reference': str(path), 'image_sha256': pair['image_sha256'],
                   'reference_sha256': reference_hash}
            frame = prepare_frames(runner, [row], historical=False)[row['sample']]
            if frame.original_size != (width, height):
                raise ValueError('Original dimensions disagree with the pairing table')
            ready.append({'sample_id': row['sample'], 'patient_id': row['patient'], 'image': row['image'],
                          'reference': row['reference'], 'reference_kind': row['kind'],
                          'image_sha256': row['image_sha256'], 'reference_sha256': row['reference_sha256']})
            observations.append((image.name, 'format_valid', kind))
        except (ValueError, TypeError, OverflowError, OSError, binascii.Error, cv2.error) as exc:
            observations.append((image.name, 'review_needed', str(exc)))
    return ready, observations


def draft_manifest(workspace, rows):
    target = workspace / DRAFT
    if target.exists():
        with target.open(encoding='utf-8-sig', newline='') as existing:
            if list(csv.DictReader(existing)) != rows:
                raise FileExistsError('Review the existing draft before replacing a changed cohort')
        return target
    temporary = target.with_suffix('.partial')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=WORKSPACE)
    parser.add_argument('--draft', action='store_true', help='Write a review draft for a fully returned pairing table')
    args = parser.parse_args()
    runner = load_runner(ORIGINAL)
    runner.verify_frozen(ORIGINAL)
    if args.draft:
        workspace = args.workspace.resolve(strict=True)
        with exclusive_training(workspace):
            rows, observations = audit(runner, workspace)
            if len(rows) == len(observations):
                print(f'Wrote review draft: {draft_manifest(workspace, rows)}')
    else:
        rows, observations = audit(runner, args.workspace)
    counts = Counter(status for _, status, _ in observations)
    print({state: counts[state] for state in ('format_valid', 'pending', 'review_needed')})
    for name, state, detail in observations:
        if state == 'review_needed':
            print(f'{name}: {detail}')
    if args.draft and len(rows) != len(observations):
        parser.error('A draft for all listed images requires one valid reference per image')


if __name__ == '__main__':
    main()
