"""Binary-region data utilities missing from the published baseline.

Open interface annotations belong to the refinement module, never these masks.
"""
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def preprocess_image(image, image_size, channels=1):
    if image is None or image.size == 0:
        raise ValueError("Image is empty")
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        resized = cv2.resize(image, (image_size[1], image_size[0]))[..., None]
    elif channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image, (image_size[1], image_size[0]))
    else:
        raise ValueError("channels must be 1 or 3")
    return torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1))).float().div(255).unsqueeze(0)


def postprocess_mask(logits, original_size):
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 2:
        raise ValueError("Expected one two-class output")
    mask = logits.argmax(1)[0].detach().cpu().numpy().astype(np.uint8) * 255
    return cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)


def extract_contours(mask, min_area=100):
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return sorted((c for c in contours if cv2.contourArea(c) >= min_area), key=cv2.contourArea, reverse=True)


def contour_to_points(contour, num_points=100):
    """Resample a CLOSED region contour; not for open interface curves."""
    p = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    if len(p) < 2 or not np.isfinite(p).all() or num_points < 2:
        raise ValueError("Contour must have finite points")
    p = np.concatenate([p, p[:1]])
    distance = np.linalg.norm(np.diff(p, axis=0), axis=1)
    keep = np.r_[True, distance > 1e-6]
    p = p[keep]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if s[-1] == 0:
        raise ValueError("Contour has zero length")
    positions = np.linspace(0, s[-1], num_points, endpoint=False)
    return np.column_stack([np.interp(positions, s, p[:, j]) for j in range(2)]).astype(np.float32)


class ICGDataset(Dataset):
    def __init__(self, data_dir, image_size=(256, 256), in_channels=1, is_train=False, augmentations=None):
        self.image_size, self.channels = image_size, in_channels
        self.is_train, self.augmentations = is_train, augmentations or {}
        root = Path(data_dir)
        images = sorted(p for p in (root / 'images').glob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'})
        masks = {p.stem: p for p in (root / 'masks').glob('*') if p.is_file()}
        missing = [p.name for p in images if p.stem not in masks]
        if missing:
            raise ValueError(f"Missing masks for {missing}")
        if not images:
            raise ValueError(f"No images in {root / 'images'}")
        self.pairs = [(p, masks[p.stem]) for p in images]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape[:2] != mask.shape:
            raise ValueError(f"Unreadable or mismatched image/mask: {image_path}")
        # Only paired geometry-preserving flips; other augmentations are not inferred.
        if self.is_train:
            for axis, key in ((1, 'horizontal_flip'), (0, 'vertical_flip')):
                if np.random.random() < self.augmentations.get(key, 0):
                    image, mask = np.flip(image, axis).copy(), np.flip(mask, axis).copy()
        image = preprocess_image(image, self.image_size, self.channels)[0]
        mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
        return {'image': image, 'mask': torch.from_numpy((mask > 0).astype(np.int64)), 'image_path': str(image_path)}


def get_dataloaders(config):
    c = config['data']
    return tuple(DataLoader(
        ICGDataset(c[f'{split}_data_dir'], c['image_size'], c.get('in_channels', 1), split == 'train', c.get('augmentations')),
        batch_size=c.get('batch_size', 8), shuffle=split == 'train',
        num_workers=c.get('num_workers', 0), pin_memory=c.get('pin_memory', False)
    ) for split in ('train', 'val', 'test'))
