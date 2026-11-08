"""Prepare a private, portable replay of the frozen patient-boundary study.

The public archive contains no patient pixels, coordinates, masks or weights.
An authorized data custodian supplies a PRIVATE CSV with sample_id, image_path,
annotation_or_mask_path and annotation_format (labelme_json or binary_mask).
Relative source paths are resolved against that CSV, never against this repo.
Every source hash must match the public split. A replay is not new independent
patient validation and never overwrites the original study or its test marker.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = MODEL_ROOT / 'experiments/patient_boundary_20261105_01'
MAP_FIELDS = ('sample_id', 'image_path', 'annotation_or_mask_path', 'annotation_format')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def verify_archive(archive):
    """Check public evidence without importing Torch or opening private data."""
    archive = Path(archive).resolve()
    spec = importlib.util.spec_from_file_location('_boundary_public_verifier',
        Path(__file__).with_name('patient_boundary_archive.py'))
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verifier.verify_archive(archive)
    manifest = load(archive / 'manifest_sha256.json')
    for relative, expected in manifest.items():
        target = (archive / relative).resolve()
        if not target.is_relative_to(archive) or sha(target) != expected:
            raise ValueError('Public archive hash mismatch: ' + relative)
    code = load(archive / 'source_snapshot_manifest.json')
    if not code or 'scripts/patient_boundary_study.py' not in code:
        raise ValueError('Missing frozen scientific runner')
    for relative, expected in code.items():
        target = (archive / 'source_snapshot' / relative).resolve()
        if not target.is_relative_to(archive / 'source_snapshot') or sha(target) != expected:
            raise ValueError('Frozen source hash mismatch: ' + relative)
    protocol = load(archive / 'protocol.public.json')
    original = protocol.get('publication', {}).get('original_protocol_sha256')
    if not original or original != load(archive / 'freeze.public.json')['protocol_sha256']:
        raise ValueError('Missing original frozen protocol identity')
    if protocol.get('input_width_height') != [768, 432]:
        raise ValueError('Unexpected input resolution')
    return protocol


def source_mapping(split, source_map):
    """Resolve the explicit custodian map and verify every source byte hash."""
    source_map = Path(source_map).resolve()
    rows = read_rows(source_map)
    expected = {row['sample_id']: row for row in split}
    if len(expected) != len(split):
        raise ValueError('Duplicate split sample')
    mapped = {}
    for row in rows:
        if any(not row.get(field) for field in MAP_FIELDS):
            raise ValueError('Source map requires: ' + ', '.join(MAP_FIELDS))
        sid = row['sample_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', sid) or sid in mapped:
            raise ValueError('Unsafe or duplicate source-map sample ID')
        if sid not in expected:
            raise ValueError('Source map has an unexpected sample: ' + sid)
        kind = row['annotation_format']
        if kind not in ('labelme_json', 'binary_mask'):
            raise ValueError('Unsupported annotation format: ' + sid)
        reference = expected[sid]
        if kind == 'binary_mask' and reference['annotation_kind'] not in ('polygon', 'polygon_derived_region_mask'):
            raise ValueError('An open line cannot be reconstructed from a binary region mask: ' + sid)
        paths = {}
        for field in ('image_path', 'annotation_or_mask_path'):
            value = Path(row[field]).expanduser()
            paths[field] = value.resolve() if value.is_absolute() else (source_map.parent / value).resolve()
            if not paths[field].is_file():
                raise ValueError('Missing private source for sample: ' + sid)
        if sha(paths['image_path']) != reference['image_sha256']:
            raise ValueError('Private image hash mismatch: ' + sid)
        if sha(paths['annotation_or_mask_path']) != reference['annotation_or_mask_sha256']:
            raise ValueError('Private annotation/mask hash mismatch: ' + sid)
        mapped[sid] = {'sample_id': sid, 'annotation_format': kind, **paths}
    if set(mapped) != set(expected):
        raise ValueError('Source map does not cover every frozen sample')
    return mapped


def separate_output(output, archive, sources=(), original_study_id='patient_boundary_20261105_01'):
    output, archive = Path(output).resolve(), Path(archive).resolve()
    if output.exists():
        raise FileExistsError('Replay output already exists; choose a new private directory')
    if output.name == original_study_id:
        raise ValueError('Replay must have a new study identity')
    protected = [archive, *(Path(path).resolve().parent for path in sources)]
    if any(output.is_relative_to(parent) for parent in protected):
        raise ValueError('Replay output must be separate from the archive and source directories')
    return output


def prepare(archive, source_map, output, *, validate_only=False):
    archive = Path(archive).resolve()
    protocol = verify_archive(archive)
    output = separate_output(output, archive, original_study_id=protocol['study_id'])
    split = read_rows(archive / 'split_manifest.csv')
    mapping = source_mapping(split, source_map)
    sources = [item[key] for item in mapping.values() for key in ('image_path', 'annotation_or_mask_path')]
    separate_output(output, archive, sources, protocol['study_id'])
    if validate_only:
        return {'stage': 'private_source_hash_validation_only', 'source_pairs': len(mapping),
                'output_created': False, 'source_pixels_decoded': False,
                'new_independent_patient_validation': False}

    # Preparation uses the original scientific rasterization and downsampling.
    # Import only here: validate-only and public verification require stdlib only.
    import numpy as np
    import cv2
    import torch
    snapshot = archive / 'source_snapshot'
    sys.path.insert(0, str(snapshot / 'src'))
    # Importing frozen code must not write __pycache__ into the sealed archive.
    previous_bytecode_setting = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        from icg_pipeline import boundary_study as scientific
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    if not Path(scientific.__file__).resolve().is_relative_to(snapshot):
        raise ValueError('A different scientific package is already imported; prepare in a fresh process')
    weight = Path(torch.hub.get_dir()) / 'checkpoints/resnet18-f37072fd.pth'
    if not weight.is_file() or sha(weight) != protocol['official_encoder_weights_sha256']:
        raise ValueError('Matching official ImageNet encoder cache is required; no random/download fallback')

    output.mkdir(parents=True)
    (output / '.gitignore').write_text('*\n', encoding='utf-8')
    shutil.copytree(snapshot, output / 'code')
    shutil.copy2(archive / 'split_manifest.csv', output / 'candidate_manifest.csv')
    original_sha = protocol['publication']['original_protocol_sha256']
    protocol = copy.deepcopy(protocol)
    protocol.update(study_id=output.name, data_root=str(output / 'data'),
                    design_status='frozen_private_replay_not_new_independent_validation',
                    training_started=False, final_test_started=False)
    protocol['replay_provenance'] = {
        'original_study_id': load(archive / 'protocol.public.json')['study_id'],
        'original_protocol_sha256': original_sha,
        'public_protocol_sha256': sha(archive / 'protocol.public.json'),
        'public_archive_manifest_sha256': sha(archive / 'manifest_sha256.json'),
        'new_independent_patient_validation': False,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'scientific_source_is_original_frozen_snapshot': True,
    }
    caches, private_paths = {}, []
    size = tuple(protocol['input_width_height'])
    for row in split:
        sid = row['sample_id']; item = mapping[sid]
        image = cv2.imdecode(np.fromfile(str(item['image_path']), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Cannot decode source image: ' + sid)
        h, w = image.shape[:2]
        if (w, h) != (3840, 2160):
            raise ValueError('Source resolution must remain 3840x2160: ' + sid)
        if item['annotation_format'] == 'labelme_json':
            region, boundary, has_region = scientific.rasterize_shapes(load(item['annotation_or_mask_path']), w, h)
            if has_region != (row['annotation_kind'] == 'polygon'):
                raise ValueError('Source annotation kind differs from frozen split: ' + sid)
        else:
            mask = cv2.imdecode(np.fromfile(str(item['annotation_or_mask_path']), np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (h, w):
                raise ValueError('Cannot decode aligned source mask: ' + sid)
            region = (mask > 0).astype(np.uint8)
            boundary, has_region = scientific.mask_boundary(region), True
        folder = output / 'data' / ('sealed_test' if row['planned_partition'] == 'test' else 'development')
        folder.mkdir(parents=True, exist_ok=True)
        cache = folder / (sid + '.npz')
        np.savez_compressed(cache,
            image=cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), size, interpolation=cv2.INTER_AREA),
            region=cv2.resize(region, size, interpolation=cv2.INTER_NEAREST),
            boundary=scientific.reduced_boundary(boundary, size), has_region=np.array(has_region),
            reference_points=scientific.reference_points(boundary),
            packed_region=np.packbits(region.reshape(-1)) if has_region else np.array([], np.uint8),
            original_size=np.array([w, h], np.int32))
        caches[cache.relative_to(output).as_posix()] = sha(cache)
        private_paths.append({key: str(value) for key, value in item.items()})
    source_hashes = load(archive / 'source_snapshot_manifest.json')
    protocol['prepared_cache_manifest_sha256'] = hashlib.sha256(json.dumps(caches, sort_keys=True).encode()).hexdigest()
    write(output / 'protocol.json', protocol)
    write(output / 'cache_manifest.json', caches)
    write(output / 'source_snapshot_manifest.json', source_hashes)
    write(output / 'source_paths.private.json', private_paths)
    gpu = torch.cuda.is_available()
    write(output / 'environment.json', {'torch': torch.__version__, 'numpy': np.__version__,
        'opencv': cv2.__version__, 'python': sys.version,
        'gpu': torch.cuda.get_device_name(0) if gpu else 'not available during preparation',
        'vram_bytes': torch.cuda.get_device_properties(0).total_memory if gpu else 0})
    write(output / 'freeze.json', {
        'protocol_sha256': sha(output / 'protocol.json'),
        'candidate_manifest_sha256': sha(output / 'candidate_manifest.csv'),
        'cache_manifest_sha256': sha(output / 'cache_manifest.json'),
        'source_snapshot_manifest_sha256': sha(output / 'source_snapshot_manifest.json'),
        'source_pairs': len(split), 'preparation_reads_test_sources': True,
        'test_model_predictions_made': False, 'test_labels_used_for_model_selection': False,
        'replay_original_protocol_sha256': original_sha})
    write(output / 'replay_ready.json', protocol['replay_provenance'])
    return {'stage': 'private_replay_prepared', 'source_pairs': len(split),
            'output_created': True, 'output': str(output),
            'new_independent_patient_validation': False}


def run_replay(study):
    study = Path(study).resolve()
    if (study / 'final_test_started.json').exists() or (study / 'final_summary.json').exists():
        raise FileExistsError('Final test already started; replay cannot be restarted')
    ready = load(study / 'replay_ready.json')
    protocol = load(study / 'protocol.json')
    if (not ready.get('original_protocol_sha256') or ready.get('new_independent_patient_validation') is not False
            or protocol.get('replay_provenance') != ready or protocol['study_id'] == ready['original_study_id']
            or protocol['data_root'] != str(study / 'data')):
        raise ValueError('Only a newly prepared private replay may be run')
    freeze = load(study / 'freeze.json')
    if freeze['protocol_sha256'] != sha(study / 'protocol.json'):
        raise ValueError('Replay protocol changed')
    runner = study / 'code/scripts/patient_boundary_study.py'
    source_hashes = load(study / 'source_snapshot_manifest.json')
    if sha(runner) != source_hashes['scripts/patient_boundary_study.py']:
        raise ValueError('Replay runner changed')
    # Do not silently remove a lock left by another live/interrupted owner.
    lock = study / 'replay_execution.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write('run-replay owns training followed by the one final test\n')
    try:
        subprocess.run([sys.executable, str(runner), 'run', '--study', str(study), '--microbatch', '4'], check=True)
        subprocess.run([sys.executable, str(runner), 'test', '--study', str(study)], check=True)
    finally:
        lock.unlink()
    return {'stage': 'replay_training_and_test_complete', 'new_independent_patient_validation': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare-replay')
    p.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    p.add_argument('--source-map', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--validate-only', action='store_true', help='Only verify mapping and byte hashes; no pixels/cache/Torch')
    r = commands.add_parser('run-replay')
    r.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare-replay':
        result = prepare(args.archive, args.source_map, args.output,
                         validate_only=args.validate_only)
    else:
        result = run_replay(args.study)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
