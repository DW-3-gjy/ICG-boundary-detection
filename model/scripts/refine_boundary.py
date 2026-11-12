#!/usr/bin/env python
"""Train/predict a native strip refiner from exported coarse curves, or smoke it."""
import argparse
from collections import Counter
from dataclasses import asdict
from functools import lru_cache
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from icg_pipeline.refinement import (
    RefinerConfig, BoundaryStripRefiner, curve_geometry, make_targets,
    sample_strips, refine_curves, save_refiner, load_refiner, load_manifest, read_sample)


class StripDataset(Dataset):
    def __init__(self, rows, config):
        self.rows, self.config, self.entries = rows, config, []
        for row_index, row in enumerate(rows):
            image, probability, curves, labels = read_sample(row)
            centers, normals, _ = curve_geometry(curves)
            targets = make_targets(labels, centers, normals, config)
            for i in np.flatnonzero(targets != -100):
                self.entries.append((row_index, centers[i], normals[i], int(targets[i]), row['patient_id']))
        if not self.entries:
            raise ValueError('No supervised candidate strips; check tri-state labels and capture radius')
        counts = Counter(e[4] for e in self.entries)
        missing = {r['patient_id'] for r in rows}-set(counts)
        if missing:
            raise ValueError(f'Patients without supervised candidates: {sorted(missing)}')
        self.weights = [1/counts[e[4]] for e in self.entries]

    def __len__(self):
        return len(self.entries)

    @lru_cache(maxsize=2)
    def sample(self, i):
        return read_sample(self.rows[i])[:2]

    def __getitem__(self, i):
        row, center, normal, target, patient = self.entries[i]
        image, probability = self.sample(row)
        strip = sample_strips(image, probability, center[None], normal[None], self.config)[0]
        return strip, target, patient


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = RefinerConfig(radius=args.radius, tangent_radius=args.tangent_radius, hidden=args.hidden, batch_size=args.batch_size)
    rows = load_manifest(args.manifest)
    training = StripDataset([r for r in rows if r['split']=='train'], config)
    validation = StripDataset([r for r in rows if r['split']=='val'], config)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(training, batch_size=args.batch_size, sampler=WeightedRandomSampler(training.weights, len(training), replacement=True, generator=generator))
    val_loader = DataLoader(validation, batch_size=args.batch_size)
    model = BoundaryStripRefiner(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history, best = [], float('inf')
    for epoch in range(args.epochs):
        model.train()
        total, count = 0., 0
        for strips, targets, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(strips.to(args.device)), targets.to(args.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += loss.item()*len(targets)
            count += len(targets)
        model.eval()
        sums, counts = Counter(), Counter()
        with torch.inference_mode():
            for strips, targets, patients in val_loader:
                losses = F.cross_entropy(model(strips.to(args.device)), targets.to(args.device), reduction='none').cpu().tolist()
                for p, loss in zip(patients, losses):
                    sums[p] += loss
                    counts[p] += 1
        validation_loss = sum(sums[p]/counts[p] for p in counts)/len(counts)
        row = {'epoch':epoch+1, 'train_ce':total/count, 'val_patient_macro_ce':validation_loss}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_loss < best:
            best = validation_loss
            save_refiner(output/'best_refiner.pt', model, {'seed':args.seed, 'epoch':epoch+1, 'manifest':str(Path(args.manifest).resolve()), 'selection':'patient_macro_validation_ce'})
    (output/'training_history.json').write_text(json.dumps({'config':asdict(config), 'history':history, 'train_candidates':len(training), 'val_candidates':len(validation)}, indent=2), encoding='utf-8')


def predict(args):
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot read image')
    model = load_refiner(args.checkpoint, args.device)
    probability = np.load(args.probability, allow_pickle=False)
    curves = json.loads(Path(args.curves).read_text(encoding='utf-8'))
    result = refine_curves(model, image, probability, curves)
    result.update({'coordinate_system':'original_image_xy_pixel_centers', 'original_size':list(image.shape[:2]), 'checkpoint':str(Path(args.checkpoint).resolve())})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    if args.overlay:
        for fragment in result['fragments']:
            cv2.polylines(image, [np.rint(fragment).astype(np.int32).reshape(-1,1,2)], False, (0,220,255), 1)
        destination = Path(args.overlay)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), image):
            raise OSError('Could not write overlay')


def smoke(args):
    """Synthetic optimizer/checkpoint/inference exercise; never clinical evidence."""
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, split in enumerate(('train','val')):
        image = np.zeros((48,64,3), dtype=np.uint8)
        image[24:, :, 1] = 220
        probability = np.zeros((24,32), dtype=np.float32)
        probability[10:] = .8
        labels = np.full((48,64), 128, dtype=np.uint8)
        labels[24,8:56] = 255
        labels[8,8:56] = 0
        curves = [[[x,21] for x in range(10,54,2)], [[x,8] for x in range(10,54,2)]]
        cv2.imwrite(str(output/f'{index}.png'), image)
        cv2.imwrite(str(output/f'{index}_labels.png'), labels)
        np.save(output/f'{index}.npy', probability)
        (output/f'{index}.json').write_text(json.dumps(curves), encoding='utf-8')
        manifest.append({'patient_id':f'SYNTHETIC_{index}', 'split':split, 'image':f'{index}.png', 'probability':f'{index}.npy', 'curves':f'{index}.json', 'labels':f'{index}_labels.png', 'coarse_train_patients':[]})
    manifest_path = output/'synthetic_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    training = argparse.Namespace(manifest=str(manifest_path), output=str(output/'train'), seed=42, radius=4, tangent_radius=2, hidden=8, batch_size=16, epochs=2, lr=1e-3, device='cpu')
    train(training)
    predict(argparse.Namespace(checkpoint=str(output/'train/best_refiner.pt'), device='cpu', image=str(output/'1.png'), probability=str(output/'1.npy'), curves=str(output/'1.json'), output=str(output/'SYNTHETIC_prediction.json'), overlay=str(output/'SYNTHETIC_overlay.png')))
    (output/'SMOKE_ONLY.json').write_text(json.dumps({'scope':'synthetic execution only; no patient efficacy', 'checkpoint_reload':True, 'prediction_written':True}, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train')
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--radius', type=int, default=32)
    p.add_argument('--tangent-radius', type=int, default=16)
    p.add_argument('--hidden', type=int, default=24)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--device', default='cpu')
    p.set_defaults(run=train)
    p = sub.add_parser('predict')
    for key in ('checkpoint','image','probability','curves','output'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--overlay')
    p.set_defaults(run=predict)
    p = sub.add_parser('smoke')
    p.add_argument('--output', required=True)
    p.set_defaults(run=smoke)
    args = parser.parse_args()
    if hasattr(args, 'epochs') and (args.epochs<1 or args.lr<=0):
        parser.error('epochs and lr must be positive')
    torch.set_num_threads(2)
    args.run(args)


if __name__ == '__main__':
    main()
