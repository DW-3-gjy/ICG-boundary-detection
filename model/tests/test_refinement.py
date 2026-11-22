import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]/'src'))
import numpy as np
import pytest
import torch
from icg_pipeline.refinement import *


def test_targets_unknown_negative_ambiguous_and_signed_displacement():
    cfg = RefinerConfig(radius=4, tangent_radius=2)
    labels = np.full((20,20),128,np.uint8)
    labels[12,10] = 255
    labels[5,5] = 0
    c = np.array([[10,10],[5,5],[2,2]],np.float32)
    n = np.array([[0,1]]*3,np.float32)
    assert make_targets(labels,c,n,cfg).tolist() == [6,9,-100]
    labels[8,10] = 255
    assert make_targets(labels,c,n,cfg)[0] == -100


def test_rejection_splits_without_bridging():
    points = np.array([[i,0] for i in range(7)],np.float32)
    fragments = split_fragments(points,[True,True,False,True,True,False,True],12)
    assert [p[:,0].tolist() for p in fragments] == [[0,1],[3,4]]


def test_native_strip_offsets_gradient_checkpoint(tmp_path):
    cfg = RefinerConfig(radius=3,tangent_radius=2,hidden=4)
    image = np.zeros((20,30,3),np.uint8)
    image[11,:,2] = 255
    probability = np.zeros((20,30),np.float32)
    c = np.array([[10,10],[12,10]],np.float32)
    n = np.array([[0,1],[0,1]],np.float32)
    strips = sample_strips(image,probability,c,n,cfg)
    assert strips.shape == (2,4,7,5)
    assert strips[0,0,4,2] == 1
    model = BoundaryStripRefiner(cfg)
    loss = torch.nn.functional.cross_entropy(model(strips),torch.tensor([4,7]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    save_refiner(tmp_path/'r.pt',model)
    restored = load_refiner(tmp_path/'r.pt')
    assert torch.allclose(model(strips),restored(strips))
    result = refine_curves(restored,image,probability,[c])
    assert result['candidate_count']==2


def test_manifest_blocks_patient_leakage(tmp_path):
    row={'patient_id':'p','split':'train','image':'a','labels':'b','probability':'c','curves':'d','coarse_train_patients':['p']}
    path=tmp_path/'m.json'
    path.write_text(json.dumps([row]),encoding='utf8')
    with pytest.raises(ValueError,match='leakage'):
        load_manifest(path)
