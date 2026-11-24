"""Export completed study evidence without patient data or local paths.

The private source study is read only. Existing exported evidence is never
overwritten; a target containing only Markdown documentation is permitted.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from patient_boundary_archive import load, require, sha, write, seal_archive, verify_archive


def export(study, output):
    study, output = Path(study).resolve(), Path(output).resolve()
    require(not (study == output or study.is_relative_to(output) or output.is_relative_to(study)),
            'Public output must be separate from private study')
    require(not output.exists() or all(p.suffix == '.md' for p in output.rglob('*') if p.is_file()),
            'Refuse to overwrite existing exported evidence; choose a new directory')
    freeze, protocol = load(study / 'freeze.json'), load(study / 'protocol.json')
    require(sha(study / 'protocol.json') == freeze['protocol_sha256'], 'Frozen protocol hash changed')
    require(sha(study / 'candidate_manifest.csv') == freeze['candidate_manifest_sha256'], 'Frozen candidate manifest changed')
    require(sha(study / 'source_snapshot_manifest.json') == freeze['source_snapshot_manifest_sha256'], 'Frozen source manifest changed')
    require(len(list((study / 'runs').glob('*/complete.json'))) == 60, 'Only complete 60-run study can be exported')
    require((study / 'final_summary.json').is_file() and (study / 'analysis/report_data.json').is_file(), 'Final test/analysis not complete')
    lock = load(study / 'checkpoint_lock.json')
    require(len(lock['checkpoints']) == 15, 'Final checkpoint lock incomplete')
    for item in lock['checkpoints']:
        require(sha(study / 'runs' / item['run'] / 'best.pt') == item['checkpoint_sha256'], 'Final checkpoint changed: ' + item['run'])
    output.mkdir(parents=True, exist_ok=True)
    publication = {'kind': 'path_redacted_derivative_not_original_frozen_protocol',
                   'original_protocol_sha256': freeze['protocol_sha256'],
                   'private_pixels_coordinates_masks_weights_included': False,
                   'original_private_records_not_modified': True}
    protocol['data_root'] = '${STUDY_ROOT}/data'
    protocol['metadata_sources_sha256'] = {f'metadata_{i:02}': digest for i, digest in enumerate(protocol['metadata_sources_sha256'].values(), 1)}
    protocol['publication'] = publication
    write(output / 'protocol.public.json', protocol)
    freeze['publication'] = {'original_freeze_sha256': sha(study / 'freeze.json'),
                             'protocol_file_in_this_archive': 'protocol.public.json',
                             'protocol_hash_above_identifies_original_private_protocol': True}
    write(output / 'freeze.public.json', freeze)
    lock['publication'] = {'original_checkpoint_lock_sha256': sha(study / 'checkpoint_lock.json')}
    write(output / 'checkpoint_lock.public.json', lock)
    # Candidate records already use anonymous IDs only; retain original metadata
    # status strings and explain their confirmation override in the README.
    shutil.copyfile(study / 'candidate_manifest.csv', output / 'split_manifest.csv')
    for name in ['source_snapshot_manifest.json', 'cache_manifest.json', 'environment.json',
                 'development_summary.json', 'final_summary.json', 'final_test_started.json']:
        shutil.copyfile(study / name, output / name)
    for relative, digest in load(study / 'source_snapshot_manifest.json').items():
        source = (study / 'code' / relative).resolve()
        require(source.is_relative_to(study / 'code'), 'Source path escapes private snapshot')
        require(sha(source) == digest, 'Frozen source changed: ' + relative)
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    report = load(study / 'analysis/report_data.json')
    report.pop('case_files', None)
    report['publication'] = {'original_report_data_sha256': sha(study / 'analysis/report_data.json'),
                             'case_image_paths_omitted': True,
                             'original_numeric_fields_preserved': True}
    write(output / 'analysis/report_data.public.json', report)
    for folder in sorted((study / 'runs').iterdir()):
        if not folder.is_dir():
            continue
        complete = load(folder / 'complete.json')
        require(sha(folder / 'best.pt') == complete['checkpoint_sha256'], 'Completed checkpoint changed: ' + folder.name)
        destination = output / 'runs' / folder.name
        destination.mkdir(parents=True, exist_ok=True)
        for name in ['config.json', 'complete.json', 'history.json', 'initialization.json']:
            shutil.copyfile(folder / name, destination / name)
        if complete['fold'] == 'refit':
            (destination / 'test').mkdir()
            for name in ['evaluation.json', 'per_image.csv']:
                shutil.copyfile(folder / 'test' / name, destination / 'test' / name)
    sealed = seal_archive(output)
    result = verify_archive(output)
    return {**result, **sealed, 'exported': True,
            'original_protocol_sha256': publication['original_protocol_sha256'],
            'private_payload_included': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(export(args.study, args.output), ensure_ascii=False, indent=2))
