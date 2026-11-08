"""Public evidence must reject tampering, patient leakage and invented regions."""
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('patient_boundary_archive', Path(__file__).parents[1] / 'scripts/patient_boundary_archive.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def test_patient_macro_does_not_weight_patients_by_frame_count():
    rows = [{'patient': 'one', 'f1': .9} for _ in range(12)] + [{'patient': 'two', 'f1': .1}]
    value, patients = archive.patient_macro(rows, 'f1')
    assert value == pytest.approx(.5)
    assert patients == {'one': .9, 'two': .1}


def test_region_macro_skips_missing_reference_without_inventing_zero():
    value, patients = archive.patient_macro([
        {'patient': 'line_only', 'dice': ''}, {'patient': 'region', 'dice': '.8'}], 'dice')
    assert value == pytest.approx(.8)
    assert set(patients) == {'region'}


def test_nonfinite_metrics_rejected():
    with pytest.raises(ValueError, match='Non-finite'):
        archive.patient_macro([{'patient': 'one', 'f1': 'nan'}], 'f1')


def test_public_reference_url_is_not_a_windows_drive_path(tmp_path):
    (tmp_path / 'references.json').write_text('{"paper":"https://arxiv.org/abs/1907.05740"}', encoding='utf-8')
    archive.seal_archive(tmp_path)
    assert archive.verify_hashes(tmp_path) == 1


def test_manifest_detects_tampering_and_unlisted_payload(tmp_path):
    (tmp_path / 'summary.json').write_text('{"mean":0.4}', encoding='utf-8')
    archive.seal_archive(tmp_path)
    assert archive.verify_hashes(tmp_path) == 1
    (tmp_path / 'summary.json').write_text('{"mean":0.9}', encoding='utf-8')
    with pytest.raises(ValueError, match='hash mismatch'):
        archive.verify_hashes(tmp_path)
    archive.seal_archive(tmp_path)
    (tmp_path / 'extra.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='coverage'):
        archive.verify_hashes(tmp_path)


@pytest.mark.parametrize('name,content', [
    ('data.json', '{"imageData":"not-public"}'),
    ('data.json', '{"points":[[1,2]]}'),
    ('config.json', '{"root":"C:/Users/Someone/private"}'),
    ('weights.pt', 'not-a-real-checkpoint'),
])
def test_private_payload_or_local_paths_rejected(tmp_path, name, content):
    (tmp_path / name).write_text(content, encoding='utf-8')
    with pytest.raises(ValueError):
        archive.seal_archive(tmp_path)


def test_split_rejects_patient_leakage_even_when_image_ids_differ():
    rows = []
    folds = {'fold_1': ['old_01', 'new_02', 'new_03'], 'fold_2': ['old_02', 'new_06', 'new_07'], 'fold_3': ['new_01', 'new_08', 'new_09']}
    for i in range(106):
        fold = list(folds)[i % 3]
        patient = folds[fold][(i // 3) % 3]
        rows.append({'sample_id': str(i), 'planned_partition': 'development', 'patient_source_id': patient,
                     'image_sha256': f'{i:064x}', 'annotation_or_mask_sha256': f'{i+1000:064x}',
                     'planned_validation_fold': fold})
    for i in range(23):
        rows.append({'sample_id': str(i+106), 'planned_partition': 'test',
                     'patient_source_id': ['new_04', 'new_05', 'new_10'][i % 3],
                     'image_sha256': f'{i+106:064x}', 'annotation_or_mask_sha256': f'{i+2000:064x}',
                     'task_stratum': 'Z_label' if i < 15 else 'L_visible_fluorescence',
                     'annotation_kind': 'linestrip' if i < 12 else 'polygon'})
    protocol = {'fixed_new_batch_test_patient_sources': ['new_04', 'new_05', 'new_10'], 'planned_development_validation_folds': folds}
    archive.verify_split(rows, protocol)
    rows[-1]['patient_source_id'] = 'old_01'
    with pytest.raises(ValueError, match='patient counts|crosses'):
        archive.verify_split(rows, protocol)
