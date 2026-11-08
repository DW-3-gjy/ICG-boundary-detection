"""ICG segmentation and optical-flow research baseline."""

__version__ = "1.0.0"
__author__ = "ICG Pipeline Team"

from .utils import load_config, setup_logging, set_seed, get_device, count_parameters, save_checkpoint, load_checkpoint
from .models import UNet, UNetWithAttention, AttentionGate, DoubleConv, Down, Up, OutConv, create_model
from .data import ICGDataset, get_dataloaders, preprocess_image, postprocess_mask, extract_contours, contour_to_points
from .tracking import OpticalFlowTracker, TrackedContour
from .training import Trainer, train_one_epoch, validate, get_loss_function, get_optimizer, get_scheduler
from .inference import ICGPredictor, run_inference
from .eval import Evaluator, evaluate_segmentation, evaluate_tracking, compute_dice, compute_iou, compute_hausdorff_distance

__all__ = [
    # Config & Utils
    'load_config',
    'setup_logging',
    'set_seed',
    'get_device',
    'count_parameters',
    'save_checkpoint',
    'load_checkpoint',
    
    # Models
    'UNet',
    'UNetWithAttention',
    'AttentionGate',
    'DoubleConv',
    'Down',
    'Up',
    'OutConv',
    'create_model',
    
    # Data
    'ICGDataset',
    'get_dataloaders',
    'preprocess_image',
    'postprocess_mask',
    'extract_contours',
    'contour_to_points',
    
    # Tracking
    'OpticalFlowTracker',
    'TrackedContour',
    
    # Training
    'Trainer',
    'train_one_epoch',
    'validate',
    'get_loss_function',
    'get_optimizer',
    'get_scheduler',
    
    # Inference
    'ICGPredictor',
    'run_inference',
    
    # Evaluation
    'Evaluator',
    'evaluate_segmentation',
    'evaluate_tracking',
    'compute_dice',
    'compute_iou',
    'compute_hausdorff_distance',
]