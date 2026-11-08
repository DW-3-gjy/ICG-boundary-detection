"""Portable source mapping and final-test protection, without patient pixels."""
import copy
import csv
import importlib.util
import subprocess
from pathlib import Path

import pytest

MODEL_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('patient_boundary_replay',
    MODEL_ROOT / 'scripts/replay_patient_boundary_study.py')
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def mapped_fixture(tmp_path):
    private = tmp_path / 'private'
    private.mkdir()
    (private / 'image.png').write_bytes(b'not decoded during source-hash validation')
    (private / 'annotation.json').write_text('{"shapes": []}', encoding='utf-8')
    split = [{'sample_id': 'sample_1', 'annotation_kind': 'linestrip',
              'image_sha256': replay.sha(private / 'image.png'),
              'annotation_or_mask_sha256': replay.sha(private / 'annotation.json')}]
    row = {'sample_id': 'sample_1', 'image_path': 'image.png',
           'annotation_or_mask_path': 'annotation.json', 'annotation_format': 'labelme_json'}
    source_map = private / 'source_map.private.csv'
    write_map(source_map, [row])
    return private, split, row, source_map


def write_map(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=replay.MAP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_source_map_is_explicit_relative_to_custodian_csv_and_checks_bytes(tmp_path):
    private, split, _, source_map = mapped_fixture(tmp_path)
    mapping = replay.source_mapping(split, source_map)
    assert mapping['sample_1']['image_path'] == (private / 'image.png').resolve()
    (private / 'image.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='image hash mismatch'):
        replay.source_mapping(split, source_map)


def test_frozen_historical_polygon_derived_masks_keep_their_existing_semantics(tmp_path):
    _, split, row, source_map = mapped_fixture(tmp_path)
    split[0]['annotation_kind'] = 'polygon_derived_region_mask'
    row['annotation_format'] = 'binary_mask'
    write_map(source_map, [row])
    assert replay.source_mapping(split, source_map)['sample_1']['annotation_format'] == 'binary_mask'


@pytest.mark.parametrize('violation', ['duplicate', 'missing', 'extra', 'unsafe_id', 'format', 'line_as_mask', 'annotation_hash'])
def test_mapping_rejects_wrong_identity_missing_data_and_semantic_substitution(tmp_path, violation):
    private, split, row, source_map = mapped_fixture(tmp_path)
    rows = [copy.deepcopy(row)]
    if violation == 'duplicate': rows *= 2
    elif violation == 'missing': rows = []
    elif violation == 'extra': rows[0]['sample_id'] = 'unknown'
    elif violation == 'unsafe_id': rows[0]['sample_id'] = '../sample_1'
    elif violation == 'format': rows[0]['annotation_format'] = 'auto'
    elif violation == 'line_as_mask': rows[0]['annotation_format'] = 'binary_mask'
    else: (private / 'annotation.json').write_text('changed', encoding='utf-8')
    write_map(source_map, rows)
    with pytest.raises(ValueError):
        replay.source_mapping(split, source_map)


def test_output_cannot_overwrite_or_nest_inside_evidence_or_patient_sources(tmp_path):
    archive = tmp_path / 'archive'; archive.mkdir()
    source = tmp_path / 'private' / 'image.png'; source.parent.mkdir(); source.write_bytes(b'x')
    with pytest.raises(FileExistsError): replay.separate_output(archive, archive)
    with pytest.raises(ValueError, match='separate'): replay.separate_output(archive / 'new', archive)
    with pytest.raises(ValueError, match='separate'): replay.separate_output(source.parent / 'new', archive, [source])
    with pytest.raises(ValueError, match='new study identity'):
        replay.separate_output(tmp_path / 'patient_boundary_20261105_01', archive)


def test_validate_only_does_not_decode_images_or_create_output(tmp_path, monkeypatch):
    _, split, _, source_map = mapped_fixture(tmp_path)
    archive = tmp_path / 'archive'; archive.mkdir()
    with (archive / 'split_manifest.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(split[0])); writer.writeheader(); writer.writerows(split)
    monkeypatch.setattr(replay, 'verify_archive', lambda path: {'study_id': 'original'})
    target = tmp_path / 'new_replay'
    result = replay.prepare(archive, source_map, target, validate_only=True)
    assert result['source_pairs'] == 1 and result['source_pixels_decoded'] is False
    assert not target.exists()


def replay_fixture(tmp_path):
    runner = tmp_path / 'code/scripts/patient_boundary_study.py'
    runner.parent.mkdir(parents=True); runner.write_text('# frozen fixture\n', encoding='utf-8')
    ready = {'original_study_id': 'original', 'original_protocol_sha256': 'a' * 64,
             'new_independent_patient_validation': False}
    protocol = {'study_id': 'new_replay', 'replay_provenance': ready,
                'data_root': str(tmp_path.resolve() / 'data')}
    replay.write(tmp_path / 'protocol.json', protocol)
    replay.write(tmp_path / 'replay_ready.json', ready)
    replay.write(tmp_path / 'freeze.json', {'protocol_sha256': replay.sha(tmp_path / 'protocol.json')})
    replay.write(tmp_path / 'source_snapshot_manifest.json', {'scripts/patient_boundary_study.py': replay.sha(runner)})
    return protocol


def test_original_or_already_tested_study_is_rejected_before_launch(tmp_path, monkeypatch):
    replay_fixture(tmp_path)
    calls = []
    monkeypatch.setattr(replay.subprocess, 'run', lambda *args, **kwargs: calls.append(args))
    marker = tmp_path / 'final_test_started.json'; marker.write_text('{}', encoding='utf-8')
    with pytest.raises(FileExistsError, match='Final test already started'): replay.run_replay(tmp_path)
    assert calls == []
    marker.unlink()
    protocol = replay.load(tmp_path / 'protocol.json'); protocol['study_id'] = 'original'
    replay.write(tmp_path / 'protocol.json', protocol)
    with pytest.raises(ValueError, match='newly prepared'): replay.run_replay(tmp_path)
    assert calls == []


def test_failed_training_never_launches_final_test_and_releases_owned_lock(tmp_path, monkeypatch):
    replay_fixture(tmp_path)
    calls = []
    def failure(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(replay.subprocess, 'run', failure)
    with pytest.raises(subprocess.CalledProcessError): replay.run_replay(tmp_path)
    assert len(calls) == 1 and calls[0][-2:] == ['--microbatch', '4']
    assert 'test' not in calls[0]
    assert not (tmp_path / 'replay_execution.lock').exists()


def test_replay_executes_frozen_run_then_test_and_refuses_second_owner(tmp_path, monkeypatch):
    replay_fixture(tmp_path)
    calls = []
    monkeypatch.setattr(replay.subprocess, 'run', lambda command, **kwargs: calls.append(command))
    lock = tmp_path / 'replay_execution.lock'; lock.write_text('other owner', encoding='utf-8')
    with pytest.raises(FileExistsError): replay.run_replay(tmp_path)
    assert calls == [] and lock.read_text(encoding='utf-8') == 'other owner'
    lock.unlink()
    result = replay.run_replay(tmp_path)
    assert [command[2] for command in calls] == ['run', 'test']
    assert result['new_independent_patient_validation'] is False


def test_protocol_or_scientific_runner_tampering_is_rejected(tmp_path, monkeypatch):
    replay_fixture(tmp_path)
    calls = []
    monkeypatch.setattr(replay.subprocess, 'run', lambda *args, **kwargs: calls.append(args))
    runner = tmp_path / 'code/scripts/patient_boundary_study.py'; runner.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='runner changed'): replay.run_replay(tmp_path)
    assert calls == []
