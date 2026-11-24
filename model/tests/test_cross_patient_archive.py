"""Guard the real anonymous archive, leakage checks and reporting conventions."""
import copy
import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('cross_patient_archive', ROOT / 'scripts/cross_patient_study.py')
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)
CAPSULE = ROOT / 'experiments/cross_patient_20261022_01'


def test_all_published_runs_reaggregate_without_patient_pixels_or_weights():
    result = archive.verify(CAPSULE)
    assert result['runs_checked'] == 24 and result['primary_Z_frames'] == 25
    assert result['test_patients'] == 1


@pytest.mark.parametrize('violation', ['patient', 'source_group', 'pixels', 'phase', 'strata', 'runs'])
def test_rejects_leakage_or_lost_primary_test_strata(violation):
    p = archive.load(CAPSULE / 'protocol.public.json')
    train = next(r for r in p['records'] if r['split'] == 'train')
    val = next(r for r in p['records'] if r['split'] == 'val')
    test = next(r for r in p['records'] if r['split'] == 'test')
    if violation == 'patient':test['patient'] = train['patient']
    elif violation == 'source_group':val['source_group'] = train['source_group']
    elif violation == 'pixels':test['decoded_image_sha256'] = train['decoded_image_sha256']
    elif violation == 'phase':train['phase'] = 'H'
    elif violation == 'strata':p['test_subgroups']['Z_instrument_occluded'].append(p['test_subgroups']['Z_original_label'][0])
    else:p['runs'].append(copy.deepcopy(p['runs'][0]))
    with pytest.raises(ValueError):archive.validate_protocol(p)


def test_uncertainty_is_sample_sd_and_single_training_has_no_invented_sd():
    result = archive.stats([.7021750040418074, .6946573215883219, .6739696126172108])
    assert result['mean'] == pytest.approx(.6902673127491133)
    assert result['sd'] == pytest.approx(.01460616854482806)
    assert archive.stats([.7427237894206837])['sd'] is None


def test_undefined_distance_is_not_silently_dropped_or_replaced_with_zero():
    result = archive.stats([500., None, 600.])
    assert result['mean'] is None and result['sd'] is None
    assert result['nonfinite_count'] == 1 and result['finite_mean'] == 550.


def test_aggregation_keeps_empty_predictions_and_zero_division_precision():
    p = {'runs': [{'condition_id': 'tiny', 'seed': s, 'run_id': str(s)} for s in [42, 43, 44]],
         'test_subgroups': {'primary': ['a', 'b']}}
    rows = {}
    for seed in [42, 43, 44]:
        rows[str(seed)] = {
            'a': {k: 1. for k in archive.METRICS} | {'empty_prediction': True, 'hausdorff_95': None},
            'b': {k: .5 for k in archive.METRICS} | {'empty_prediction': False}}
    values = archive.aggregate(p, rows)['tiny']['groups']['primary']
    assert values['dice']['mean'] == .75
    assert values['precision']['mean'] == .25  # raw empty precision=1 is not rewarded
    assert values['hausdorff_95']['mean'] is None
    assert values['empty_predictions_by_seed'] == [1, 1, 1]


def test_final_test_marker_refuses_replay_before_any_records_are_written(tmp_path):
    marker = tmp_path / 'test_evaluation_started.json'
    marker.write_text('{"locked": true}', encoding='utf-8')
    before = marker.read_bytes()
    with pytest.raises(FileExistsError, match='Final test already started'):
        archive.run(tmp_path)
    assert marker.read_bytes() == before and sorted(p.name for p in tmp_path.iterdir()) == [marker.name]


def test_existing_replay_output_and_nested_output_are_refused(tmp_path):
    existing = tmp_path / 'existing'
    existing.mkdir()
    with pytest.raises(FileExistsError):archive.prepare(CAPSULE, tmp_path / 'private', existing)
    with pytest.raises(ValueError, match='must be separate'):
        archive.prepare(CAPSULE, tmp_path / 'private', tmp_path / 'private/nested')


def test_dataset_rebase_changes_only_paths_and_keeps_protocol_hyperparameters():
    p = archive.load(CAPSULE / 'protocol.public.json')
    q = archive.replace_roots(copy.deepcopy(p), Path('/private-data'), Path('/new-output'))
    for a, b in zip(p['runs'], q['runs']):
        assert a['seed'] == b['seed'] and a['condition_id'] == b['condition_id']
        assert a['config']['model'] == b['config']['model']
        assert a['config']['data']['augmentations'] == b['config']['data']['augmentations']
        assert a['config']['training']['loss_params'] == b['config']['training']['loss_params']
        assert b['config']['data']['test_data_dir'] == str(Path('/private-data')) + '/fullres/test'
