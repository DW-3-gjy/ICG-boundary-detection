"""Verify archived source bytes and recompute the published patient-level scores."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean


ARCHIVE = Path(__file__).resolve().parent
REPO = ARCHIVE.parents[2]
RECORDS = ARCHIVE / 'records'


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, message):
    require(abs(actual - expected) < 1e-12, message)


def grouped(rows):
    output = defaultdict(list)
    for row in rows:
        output[row['patient']].append(row)
    return output


def verify_development(rows, report, strata):
    require(len(rows) == 66 and len({r['image'] for r in rows}) == 66, 'Development frame count')
    patients = grouped(rows)
    require(len(patients) == 5, 'Development patient count')
    require(Counter(r['fold'] for r in rows) == {'fold_1': 32, 'fold_2': 29, 'fold_3': 5}, 'Development fold sizes')
    require(all(len({row['fold'] for row in frames}) == 1 for frames in patients.values()), 'Patient crosses folds')
    by_stratum = defaultdict(list)
    for row in rows:
        by_stratum[row['stratum']].append(row)
    comparisons = [(report['overall'], patients)]
    comparisons.extend((entry['metrics'], grouped(by_stratum[name])) for name, entry in strata['strata'].items())
    for score_table, by_patient in comparisons:
        for method, metrics in score_table.items():
            for metric, value in metrics.items():
                computed = mean(mean(r['scores'][method][metric] for r in frames) for frames in by_patient.values())
                equal(computed, value, f'Development {method} {metric}')
    for name, entry in strata['strata'].items():
        require(len(by_stratum[name]) == entry['frames'], f'{name} frame count')
        require(len(grouped(by_stratum[name])) == entry['patients'], f'{name} patient count')


def verify_historical(data):
    rows = data['per_image']
    patients = grouped(rows)
    require(data['scope'] == 'historical', 'Historical scope')
    require(len(rows) == data['frames'] == 11 and len({r['image'] for r in rows}) == 11, 'Historical frame count')
    require(data['patient_count'] == 2 and sorted(map(len, patients.values())) == [4, 7], 'Historical patient count')
    by_patient = {entry['patient']: entry for entry in data['patients']}
    require(set(by_patient) == set(patients), 'Historical summaries have different patients')
    for method, expected in data['overall'].items():
        scores = []
        for label, items in patients.items():
            require(all(0 <= r[method] <= 1 for r in items), 'Dice out of range')
            score = mean(r[method] for r in items)
            equal(score, by_patient[label][method], f'{label} {method}')
            scores.append(score)
        equal(mean(scores), expected, f'Historical {method}')


def main():
    sources = load(ARCHIVE / 'source_manifest.json')
    for name, expected in sources.items():
        path = (REPO / name).resolve()
        require(path.is_relative_to(REPO / 'model' / 'scripts') and path.suffix == '.py', 'Invalid source entry')
        require(digest(path) == expected, f'Source bytes changed: {name}')

    def source(name):
        return sources[f'model/scripts/{name}.py']

    large = RECORDS / 'static_d_sam2_large_20261124_01' / 'results.json'
    training = RECORDS / 'static_d_sam2_large_precision_20261124_01' / 'results.json'
    candidate = load(RECORDS / 'static_d_sam2_large_refit_20261124_01' / 'candidate.json')
    protocol = candidate['protocol']
    require(protocol['source_sha256'] == source('refit_static_d_sam2_large_decoder'), 'Refit source receipt')
    require(load(large)['source_sha256'] == source('evaluate_static_d_sam2_large'), 'Large comparison source receipt')
    require(load(training)['source_sha256'] == source('train_static_d_sam2_large_precision'), 'Training source receipt')
    require(protocol['development_results_sha256'] == digest(training), 'Training results receipt')
    require(protocol['large_candidate']['development_result_sha256'] == digest(large), 'Large results receipt')
    require(protocol['refit_steps'] == 180 and candidate['training_mask_count'] == 66, 'Refit budget or cohort')
    archived = load(ARCHIVE / 'paired_scores.json')
    require(archived['development']['scope'] == 'development-only', 'Development scope')
    verify_development(archived['development']['per_image'], load(training), load(training.with_name('stratum_summary.json')))
    verify_historical(archived['historical'])
    print(f'Verified {len(sources)} source files, 66 development scores and 11 historical scores.')


if __name__ == '__main__':
    main()
