"""Single-image CLI behavior for Windows paths used by the local project."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from icg_pipeline.data import read_image, write_image
from icg_pipeline.inference import run_inference
from icg_pipeline.models import create_model


@pytest.mark.parametrize('explicit_output', [False, True])
def test_image_inference_reads_and_exports_unicode_paths(tmp_path, explicit_output):
    source = tmp_path / '胸腔镜原图' / '样本一.png'
    checkpoint = tmp_path / '模型权重' / '模型.pt'
    checkpoint.parent.mkdir()
    image = np.random.default_rng(42).integers(0, 256, (48, 80, 3), dtype=np.uint8)
    write_image(source, image)
    config = {
        'device': {'use_cuda': False},
        'model': {'name': 'unet', 'in_channels': 3, 'out_channels': 2,
                  'base_filters': 4, 'depth': 2, 'dropout': 0.0},
        'data': {'image_size': [32, 32], 'in_channels': 3},
        'inference': {'model_path': str(checkpoint),
                      'output_dir': str(tmp_path / '预测结果'),
                      'save_overlay': not explicit_output, 'min_contour_area': 4},
        'tracking': {'enabled': False},
    }
    model = create_model(config)
    torch.save({'model_state_dict': model.state_dict(), 'config': config}, checkpoint)
    output = str(tmp_path / '指定导出目录' / '边界叠加.png') if explicit_output else None
    result = run_inference('', str(source), output_path=output, config_override=config)
    assert result['mask'].shape == image.shape[:2]
    np.testing.assert_array_equal(read_image(result['mask_path'], grayscale=True), result['mask'])
    assert read_image(result['overlay_path']).shape == image.shape
    assert Path(result['mask_path']).name == '样本一_mask.png'
    if explicit_output:
        assert result['overlay_path'] == output
