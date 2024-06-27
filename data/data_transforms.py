import numpy as np
import torch
from torchvision.transforms.functional import resize, center_crop

from data.operators import create_operator, create_noise_schedule

class ImageDataTransform:
    # Designed for ImageNet data with U-Net
    # See typical ImageNet preprocessing here: 
    # https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
    
    def __init__(self, 
                 is_train, 
                 operator_config,
                 noise_config=None,
                 dt=1e-3,
                ):
        self.is_train = is_train
        self.fwd_operator = create_operator(operator_config)
        self.noise_scheduler = create_noise_schedule(noise_config)
        self.dt = dt
        self.rng = np.random.RandomState()

    def __call__(self, 
                 image, 
                 fname=None
                ):
        
        # Crop image to square 
        shorter = min(image.size)
        image = center_crop(image, shorter)

        # Crop image to square 
        shorter = min(image.size)
        image = center_crop(image, shorter)
        
        # Resize images to uniform size
        image = resize(image, (256, 256))
        
        # Convert to ndarray and permute dimensions to C, H, W
        image = np.array(image)
        image = image.transpose(2, 0, 1)
        
        # Normalize image to range [0, 1]
        image = image / 255.
    
        # Convert to tensor
        image = torch.from_numpy(image.astype(np.float32))
        image = image.unsqueeze(0)
        
        # Generate degraded noisy images
        t_this = torch.rand(1)
        t_next = t_this - self.dt
        t_next = t_next if t_next > 0.0 else torch.zeros_like(t_this)
        degraded_this = self.fwd_operator(image, t_this).squeeze(0) 
        degraded_next = self.fwd_operator(image, t_next).squeeze(0) 
        z, noise_std = self.noise_scheduler(t_this, image.shape)
        degraded_noisy = degraded_this + z.to(image.device)
        
        image = image.squeeze(0)
        degraded_noisy = degraded_noisy.squeeze(0)
        out_dict = {
                'clean': image, 
                'degraded_noisy': degraded_noisy, 
                'noise_std': noise_std,
                'degraded_this': degraded_this, 
                't_this': t_this,
                'degraded_next':degraded_next, 
                't_next': t_next,
                'fname': fname,
               }
        return out_dict
    
    def set_initial_seed(self, seed):
        self.rng.seed(seed)