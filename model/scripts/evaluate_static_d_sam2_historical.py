"""Replay the locked Large decoder on the original complete-mask test frames."""

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2

import evaluate_static_d_locked_regions as cohorts
import refit_static_d_sam2_large_decoder as refit
from static_d_baseline import ORIGINAL, load_runner, read_json, verify_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dependencies', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    args = parser.parse_args()
    source, deps, weights = (path.resolve() for path in (args.source, args.dependencies, args.weights))
    cv2.setNumThreads(2)
    runner = load_runner(ORIGINAL)
    runner.verify_frozen(ORIGINAL)
    candidate = read_json(refit.CANDIDATE)
    if candidate['protocol'] != refit.protocol(source, deps, weights):
        raise ValueError('Replay requires the pinned candidate and runtime')
    verify_file(refit.CHECKPOINT, candidate['decoder_sha256'])
    rows = cohorts.read_cohort(runner, manifest=None)
    if sorted(Counter(row['patient'] for row in rows).values()) != [4, 7]:
        raise ValueError('Expected the archived 11 complete masks across two patients')
    frames = cohorts.prepare_frames(runner, rows, historical=True)
    predictor = refit.frozen_large.load_large(source, deps, weights)
    result = refit.score(runner, candidate, rows, frames, predictor)
    print(json.dumps({'scope': 'historical', 'method': refit.METHOD, **result}, indent=2))


if __name__ == '__main__':
    main()
