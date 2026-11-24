"""Training pipeline for ICG U-Net segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import time
import json
import platform
from importlib.metadata import version, PackageNotFoundError

from ..utils import AverageMeter, EarlyStopping, save_checkpoint, get_device, setup_logging
from ..models import UNet
from ..data import get_dataloader

logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth: float = 1e-6, include_background: bool = True):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W), targets: (B, H, W)
        probs = F.softmax(logits.float(), dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        if not self.include_background:
            dice = dice[:, 1:]
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    """Combined Dice and Cross Entropy loss."""
    
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5, smooth: float = 1e-6,
                 include_background: bool = True):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(smooth, include_background)
        self.ce_loss = nn.CrossEntropyLoss()
        self.last_components = {}
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits.float(), targets)
        self.last_components = {'dice_loss': dice.detach().item(), 'ce_loss': ce.detach().item()}
        return self.dice_weight * dice + self.ce_weight * ce


class TverskyLoss(nn.Module):
    """Foreground soft Tversky, with explicitly named FP and FN weights."""

    def __init__(self, fp_weight=0.7, fn_weight=0.3, smooth=1e-6):
        super().__init__()
        if fp_weight <= 0 or fn_weight <= 0 or smooth <= 0:
            raise ValueError('Tversky weights and smooth must be positive')
        self.fp_weight, self.fn_weight, self.smooth = fp_weight, fn_weight, smooth

    def forward(self, logits, targets):
        p = logits.float().softmax(1)[:, 1]
        y = (targets == 1).float()
        tp = (p * y).sum((1, 2))
        fp = (p * (1-y)).sum((1, 2))
        fn = ((1-p) * y).sum((1, 2))
        return (1 - (tp + self.smooth) /
                (tp + self.fp_weight * fp + self.fn_weight * fn + self.smooth)).mean()


class BoundaryCELoss(nn.Module):
    """Per-image CE in a reference mask's morphological boundary band."""

    def __init__(self, width=2):
        super().__init__()
        if width < 1:
            raise ValueError('boundary width must be positive')
        self.width = int(width)

    def forward(self, logits, targets):
        y = (targets == 1).float().unsqueeze(1)
        k = 2 * self.width + 1
        band = (F.max_pool2d(y, k, 1, self.width) +
                F.max_pool2d(-y, k, 1, self.width)).squeeze(1)
        nondegenerate = ((y.sum((1, 2, 3)) > 0) & (y.sum((1, 2, 3)) < targets[0].numel()))
        band = band * nondegenerate[:, None, None]
        ce = F.cross_entropy(logits.float(), targets, reduction='none')
        return ((ce * band).sum((1, 2)) / band.sum((1, 2)).clamp_min(1)).mean()


class TverskyCELoss(nn.Module):
    def __init__(self, tversky_weight=0.5, ce_weight=0.5, boundary_weight=0.0,
                 fp_weight=0.7, fn_weight=0.3, smooth=1e-6, boundary_width=2):
        super().__init__()
        weights = (tversky_weight, ce_weight, boundary_weight)
        if min(weights) < 0 or sum(weights) <= 0:
            raise ValueError('loss coefficients must be nonnegative with a positive sum')
        self.weights = weights
        self.tversky = TverskyLoss(fp_weight, fn_weight, smooth)
        self.boundary = BoundaryCELoss(boundary_width)
        self.last_components = {}

    def forward(self, logits, targets):
        tv = self.tversky(logits, targets)
        ce = F.cross_entropy(logits.float(), targets)
        boundary = self.boundary(logits, targets) if self.weights[2] else logits.float().sum() * 0
        self.last_components = {'tversky_loss': tv.detach().item(), 'ce_loss': ce.detach().item(),
                                'boundary_ce_loss': boundary.detach().item()}
        return self.weights[0]*tv + self.weights[1]*ce + self.weights[2]*boundary


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits.float(), targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def get_loss_function(config: Dict[str, Any]) -> nn.Module:
    """Get loss function based on config."""
    loss_config = config['training']
    loss_name = loss_config['loss_function']
    params = loss_config.get('loss_params', {})
    
    if loss_name == 'dice':
        return DiceLoss(**params)
    elif loss_name == 'dice_ce':
        return DiceCELoss(**params)
    elif loss_name in ('tversky_ce', 'tversky_ce_boundary'):
        defaults = {'ce_weight': 0.4, 'boundary_weight': 0.1} if loss_name.endswith('_boundary') else {}
        return TverskyCELoss(**(defaults | params))
    elif loss_name == 'focal':
        return FocalLoss(**params)
    elif loss_name == 'ce':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Get optimizer based on config."""
    train_config = config['training']
    opt_name = train_config['optimizer']
    lr = train_config['learning_rate']
    weight_decay = train_config.get('weight_decay', 1e-5)
    parameters = model.parameters()
    if hasattr(model, 'encoder'):
        encoder_ids = {id(p) for p in model.encoder.parameters()}
        parameters = [
            {'params': model.encoder.parameters(), 'lr': train_config.get('encoder_learning_rate', lr)},
            {'params': [p for p in model.parameters() if id(p) not in encoder_ids], 'lr': lr},
        ]
    
    if opt_name == 'adamw':
        return AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def get_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> torch.optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler based on config."""
    train_config = config['training']
    sched_name = train_config['scheduler']
    params = train_config.get('scheduler_params', {})
    
    if sched_name == 'cosine_annealing':
        return CosineAnnealingLR(optimizer, **params)
    elif sched_name == 'reduce_on_plateau':
        return ReduceLROnPlateau(optimizer, mode='max', **params)
    elif sched_name == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, **params)
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    gradient_clip: float = 1.0,
    mixed_precision: bool = False
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    component_meters = {}
    optimizer_updates = 0
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if mixed_precision and scaler is not None:
            with autocast():
                logits = model(images)
                loss = criterion(logits, masks)
            
            scaler.scale(loss).backward()
            
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            scaler.step(optimizer)
            old_scale = scaler.get_scale()
            scaler.update()
            optimizer_updates += int(scaler.get_scale() >= old_scale)
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            
            loss.backward()
            
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            optimizer_updates += 1
        
        # Compute dice score
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            dice = compute_dice_score(preds, masks)
        
        loss_meter.update(loss.item(), images.size(0))
        dice_meter.update(dice, images.size(0))
        for key, value in getattr(criterion, 'last_components', {}).items():
            component_meters.setdefault(key, AverageMeter()).update(value, images.size(0))
    
    return {
        'loss': loss_meter.avg,
        'dice': dice_meter.avg,
        'optimizer_updates': optimizer_updates,
        **{key: meter.avg for key, meter in component_meters.items()},
    }


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    mixed_precision: bool = False
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    component_meters = {}
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            
            if mixed_precision:
                with autocast():
                    logits = model(images)
                    loss = criterion(logits, masks)
            else:
                logits = model(images)
                loss = criterion(logits, masks)
            
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            
            dice = compute_dice_score(preds, masks)
            iou = compute_iou_score(preds, masks)
            
            loss_meter.update(loss.item(), images.size(0))
            dice_meter.update(dice, images.size(0))
            iou_meter.update(iou, images.size(0))
            for key, value in getattr(criterion, 'last_components', {}).items():
                component_meters.setdefault(key, AverageMeter()).update(value, images.size(0))
    
    return {
        'loss': loss_meter.avg,
        'dice': dice_meter.avg,
        'iou': iou_meter.avg,
        **{key: meter.avg for key, meter in component_meters.items()},
    }


def compute_dice_score(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> float:
    """Compute foreground-only Dice for class-index masks."""
    preds, targets = (preds == 1).float(), (targets == 1).float()
    intersection = (preds * targets).sum(dim=(1, 2))
    union = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def compute_iou_score(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> float:
    """Compute foreground-only IoU for class-index masks."""
    preds, targets = (preds == 1).float(), (targets == 1).float()
    intersection = (preds * targets).sum(dim=(1, 2))
    union = (preds + targets - preds * targets).sum(dim=(1, 2))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


class Trainer:
    """Main trainer class for ICG U-Net."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_device(config['device'])
        self.logger = setup_logging(config['logging']['log_dir'], config['logging']['level'])
        
        # Set seed for reproducibility
        from ..utils import set_seed
        set_seed(int(config['training'].get('seed', 42)))
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.logger.info(f"Model created on {self.device}: {self.model.count_parameters():,} parameters")
        
        # Create dataloaders
        self.train_loader = get_dataloader(config, 'train', training=True)
        self.val_loader = get_dataloader(config, 'val')
        
        # Create loss, optimizer, scheduler
        self.criterion = get_loss_function(config).to(self.device)
        self.optimizer = get_optimizer(self.model, config)
        self.scheduler = get_scheduler(self.optimizer, config)
        
        # Mixed precision
        self.mixed_precision = bool(config['training'].get('mixed_precision', False)) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.mixed_precision else None
        
        # Early stopping
        early_stop_config = config['training'].get('early_stopping', {})
        self.early_stopping_enabled = early_stop_config.get('enabled', True)
        self.early_stopping = EarlyStopping(
            patience=early_stop_config.get('patience', 15),
            min_delta=early_stop_config.get('min_delta', 1e-4),
            mode=early_stop_config.get('mode', 'max')
        )
        
        # Checkpointing
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = config['training'].get('save_top_k', 3)
        self.best_metrics = []
        
        # Training state
        self.current_epoch = 0
        self.best_val_dice = float('-inf')
        self.optimizer_updates = 0
        self.history = []
        self.metadata = {'python': platform.python_version(), 'packages': {},
                         'preprocessing': {key: config['data'].get(key, default) for key, default in
                                           [('normalization', 'legacy'), ('image_size', [256, 256]),
                                            ('in_channels', config['model'].get('in_channels', 1))]},
                         'encoder_weights': config['model'].get('encoder_weights'),
                         'split_manifest': config.get('experiment', {}).get('split_manifest')}
        for package in ('torch', 'torchvision', 'numpy', 'opencv-python', 'scipy', 'PyYAML'):
            try:
                self.metadata['packages'][package] = version(package)
            except PackageNotFoundError:
                self.metadata['packages'][package] = None
        
        # TensorBoard
        self.writer = None
        if config['logging'].get('use_tensorboard', False):
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(config['logging']['log_dir'])
    
    def train(self) -> Dict[str, float]:
        """Run full training loop."""
        self.logger.info("Starting training...")
        start_time = time.time()
        
        train_config = self.config['training']
        num_epochs = train_config['epochs']
        log_interval = self.config['logging'].get('log_interval', 10)
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = train_one_epoch(
                self.model, self.train_loader, self.optimizer, self.criterion,
                self.device, self.scaler,
                train_config.get('gradient_clip', 1.0),
                self.mixed_precision
            )
            self.optimizer_updates += train_metrics['optimizer_updates']
            
            # Validation
            val_metrics = validate(
                self.model, self.val_loader, self.criterion,
                self.device, self.mixed_precision
            )
            self.history.append({'epoch': epoch+1, 'train': train_metrics, 'val': val_metrics,
                                 'optimizer_updates': self.optimizer_updates})
            (self.checkpoint_dir / 'history.json').write_text(json.dumps(self.history, indent=2), encoding='utf-8')
            self.logger.info('Loss components: %s', self.history[-1])
            
            # Learning rate scheduling
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics['dice'])
            else:
                self.scheduler.step()
            
            # Logging
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Dice: {train_metrics['dice']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Dice: {val_metrics['dice']:.4f} | "
                f"Val IoU: {val_metrics['iou']:.4f} | "
                f"LR: {lr:.2e}"
            )
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('train/loss', train_metrics['loss'], epoch)
                self.writer.add_scalar('train/dice', train_metrics['dice'], epoch)
                self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('val/dice', val_metrics['dice'], epoch)
                self.writer.add_scalar('val/iou', val_metrics['iou'], epoch)
                self.writer.add_scalar('lr', lr, epoch)
            
            # Checkpointing
            val_dice = val_metrics['dice']
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.save_checkpoint('best_model.pt', val_metrics)
            
            # Save top-k checkpoints
            self.save_topk_checkpoint(epoch, val_metrics)
            
            # Early stopping
            if self.early_stopping_enabled and self.early_stopping(val_dice):
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        
        if self.writer:
            self.writer.close()
        
        return {'best_val_dice': self.best_val_dice}
    
    def save_checkpoint(self, filename: str, metrics: Dict[str, float]):
        """Save model checkpoint."""
        filepath = self.checkpoint_dir / filename
        save_checkpoint({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'optimizer_updates': self.optimizer_updates,
            'metadata': self.metadata,
        }, str(filepath))
        self.logger.info(f"Saved checkpoint: {filepath}")
    
    def save_topk_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save checkpoint if in top-k."""
        val_dice = metrics['dice']
        self.best_metrics.append((val_dice, epoch))
        self.best_metrics.sort(key=lambda x: x[0], reverse=True)
        
        if len(self.best_metrics) > self.save_top_k:
            # Remove worst
            worst_dice, worst_epoch = self.best_metrics.pop()
            worst_path = self.checkpoint_dir / f'epoch_{worst_epoch}_dice_{worst_dice:.4f}.pt'
            if worst_path.exists():
                worst_path.unlink()
        
        # Save current if in top-k
        if (val_dice, epoch) in self.best_metrics:
            filename = f'epoch_{epoch}_dice_{val_dice:.4f}.pt'
            self.save_checkpoint(filename, metrics)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        from ..utils import load_checkpoint
        checkpoint = load_checkpoint(filepath, self.model, self.optimizer)
        self.current_epoch = checkpoint.get('epoch', -1) + 1
        self.best_val_dice = checkpoint.get('metrics', {}).get('dice', 0.0)
        self.optimizer_updates = checkpoint.get('optimizer_updates', 0)
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.logger.info(f"Loaded checkpoint from {filepath}")


def create_model(config: Dict[str, Any]) -> UNet:
    """Factory function to create model (imported from models)."""
    from ..models import create_model as create_model_fn
    return create_model_fn(config)
