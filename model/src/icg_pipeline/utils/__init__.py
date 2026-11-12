"""Utility functions for the ICG pipeline."""

import yaml
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np
import random


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path).resolve()
    with config_file.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {config_file}")
    for section, keys in {
        'data': ('train_data_dir', 'val_data_dir', 'test_data_dir'),
        'training': ('checkpoint_dir',),
        'inference': ('model_path', 'output_dir'),
        'logging': ('log_dir',),
        'refinement': ('checkpoint',),
    }.items():
        for key in keys:
            value = config.get(section, {}).get(key)
            if value and not Path(value).is_absolute():
                config[section][key] = str((config_file.parent.parent / value).resolve())
    return config


def setup_logging(log_dir: str, level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path / 'pipeline.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_config: Dict[str, Any]) -> torch.device:
    """Get torch device based on configuration."""
    if device_config.get('use_cuda', True) and torch.cuda.is_available():
        device_id = device_config.get('device_id', 0)
        return torch.device(f'cuda:{device_id}')
    return torch.device('cpu')


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state: Dict[str, Any], filepath: str):
    """Save model checkpoint."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
