"""Data augmentation for pixel-based RL.
Implements random shift (pad + random crop), the core technique from
DrQ (Data-regularized Q).
"""

import torch
import torch.nn.functional as F

def random_shift(images: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Randomly shift the image by a small amount."""
    batch_size, channels, height, width = images.shape
    padded = F.pad(images, [pad, pad, pad, pad], mode='replicate')

    crop_top = torch.randint(0, 2 * pad + 1, size=(batch_size,))
    crop_left = torch.randint(0, 2 * pad + 1, size=(batch_size,))
    
    cropped = torch.empty_like(images)
    for i in range(batch_size):
        t = crop_top[i]
        l = crop_left[i]
        cropped[i] = padded[i, :, t:t+height, l:l+width]
    return cropped