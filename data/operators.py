import torch
from torch import nn
import scipy.ndimage
import numpy as np

class NoiseScheduler:
    def __init__(self, sigma_min, sigma_max):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def __call__(self, t, noise_shape, seed=None):
        assert 0.0 <= t <= 1.0
        std = self.get_std(t)
        if seed is not None:
            rand_state = torch.get_rng_state()
            torch.random.manual_seed(seed)
            z = torch.randn(noise_shape) * std
            torch.set_rng_state(rand_state)
            return z, std
        else:    
            return torch.randn(noise_shape) * std, std
    
    def get_std(self, t):
        if self.sigma_min == self.sigma_max == 0.0:
            return torch.tensor(0.0)
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        return std

class Blurkernel(nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size//2),
            nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3)
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2,self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std, truncate=6.0)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)
                f.requires_grad_(False)
        elif self.blur_type == "motion":
            raise ValueError('Unsupported blur type.')

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k)
        for name, f in self.named_parameters():
            f.data.copy_(k)
            f = f.to(k.device)
            f.requires_grad_(False)

    def get_kernel(self):
        return self.k
    
class GaussianBlurOperator:
    def __init__(self, 
                 kernel_size,
                 std_schedule,
                 from_file=None,
                 **kwargs,
                ):
        self.kernel_size = kernel_size
        if from_file is None:
            self.std_schedule = std_schedule
        else:
            assert from_file is not None
            self.t_vals, self.std_vals = torch.from_numpy(np.loadtxt(from_file)[:, 0]), torch.from_numpy(np.loadtxt(from_file)[:, 1])
            self.std_schedule = lambda t: self.lerp_std(t)
        self.conv = None
        
    def update_kernel(self, t):
        self.conv = Blurkernel(blur_type='gaussian',
                       kernel_size=self.kernel_size,
                       std=self.std_from_t(t),
                      )
        self.kernel = self.conv.get_kernel().to(t.device)
        self.conv.update_weights(self.kernel.type(torch.float32))
        self.conv.to(t.device)
        
    def __call__(self, data, t):
        return self.forward(data, t)

    def forward(self, data, t):
        assert 0.0 <= t <= 1.0
        self.update_kernel(t)
        return self.conv(data)

    def get_kernel(self, t):
        self.update_kernel(t)
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)
    
    def std_from_t(self, t):
        std = self.std_schedule(t)
        std_np = float(std) 
        return std_np
    
    def lerp_std(self, t):
        assert 0.0 <= t <= 1.0
        self.std_vals = self.std_vals.to(t.device)
        self.t_vals = self.t_vals.to(t.device)
        if t == 0.0:
            return self.std_vals[0]
        elif t == 1.0:
            return self.std_vals[-1]
        else:
            # linear interpolation
            t_end_index = (self.t_vals >= t).nonzero(as_tuple=False)[0]
            t_start_index = t_end_index - 1
            std_out = self.std_vals[t_start_index] + (self.std_vals[t_end_index]- self.std_vals[t_start_index]) * (t - self.t_vals[t_start_index]) / (self.t_vals[t_end_index] - self.t_vals[t_start_index])
            return std_out

        
class InpaintingOperator:
    def __init__(self, 
                 mask_type='box',
                 box_size=None,
                 mask_min_std=None,
                 mask_max_std=None,
                 mask_pow=None,
                 mask_schedule=None,
                 from_file=None,
                 img_size=256,
                 **kwargs,
                ):
        self.box_size = box_size
        self.mask_type = mask_type
        if self.mask_type == 'box':
            assert box_size is not None
            self.box_size = box_size
        elif self.mask_type == 'gaussian':
            assert mask_min_std is not None
            assert mask_max_std is not None
            assert mask_pow is not None
            self.mask_min_std = mask_min_std
            self.mask_max_std = mask_max_std
            self.mask_pow = mask_pow
        if from_file is None:
            assert mask_schedule is not None
            self.mask_schedule = mask_schedule
        else:
            self.t_vals, self.mask_factor_vals = torch.from_numpy(np.loadtxt(from_file)[:, 0]), torch.from_numpy(np.loadtxt(from_file)[:, 1])
            self.mask_schedule = lambda t: self.lerp_mask(t)
        self.img_size = img_size
        
    def __call__(self, data, t):
        return self.forward(data, t)

    def forward(self, data, t):
        assert 0.0 <= t <= 1.0
       # t = self.mask_schedule(t)
        mask_t = self.mask_from_t(t)
        return data * mask_t.to(data.device)
    
    def mask_from_t(self, t):
        if t == 0.0:
            return torch.ones((self.img_size, self.img_size))
        if self.mask_type == 'box':
            mask1 = torch.ones((self.img_size, self.img_size))
            mask1 = self.set_center_box_to_val(mask1, torch.floor(self.box_size * t), 0) # Zero mask in the center
            mask2 = torch.ones((self.img_size, self.img_size))
            mask2 = self.set_center_box_to_val(mask2, torch.floor(self.box_size * t) + 1, 1 - self.box_size * t + torch.floor(self.box_size * t)) # linear mask on the perimeter
            return mask1 * mask2
        elif self.mask_type == 'gaussian':
            mask = self.gaussian_mask_from_t(t, (self.img_size, self.img_size))
            return mask
        else:
            raise ValueError('Inpainting mask type not implemented.')
    
    def lerp_mask(self, t):
        assert 0.0 <= t <= 1.0
        if t == 0.0:
            return self.mask_factor_vals[0]
        elif t == 1.0:
            return self.mask_factor_vals[-1]
        else:
            # linear interpolation
            assert 0.0 <= t <= 1.0
            self.mask_factor_vals = self.mask_factor_vals.to(t.device)
            self.t_vals = self.t_vals.to(t.device)
            t_end_index = (self.t_vals >= t).nonzero(as_tuple=False)[0]
            t_start_index = t_end_index - 1
            mask_factor_out = self.mask_factor_vals[t_start_index] + (self.mask_factor_vals[t_end_index]- self.mask_factor_vals[t_start_index]) * (t - self.t_vals[t_start_index]) / (self.t_vals[t_end_index] - self.t_vals[t_start_index])
            return mask_factor_out
    
    def gaussian_mask_from_t(self, t, shape):
        h, w = shape
        def crop_center(img,cropx,cropy):
            y,x = img.shape
            startx = x//2-(cropx//2)
            starty = y//2-(cropy//2)    
            return img[starty:starty+cropy,startx:startx+cropx]

        kernel_size_h, kernel_size_w = h * 2, w * 2
        std = float(self.mask_min_std + t * (self.mask_max_std - self.mask_min_std))
        n = np.zeros((kernel_size_h, kernel_size_w))
        n[kernel_size_h // 2, kernel_size_w // 2] = 1
        k = scipy.ndimage.gaussian_filter(n, sigma=std)
        k = 1.0 - k / k.max()
        k = crop_center(k, h, w)
        k = k ** self.mask_pow
        return torch.from_numpy(k).float().view(h, w)
    
    @staticmethod
    def set_center_box_to_val(x, box_sz, val):
        h, w = x.shape[-2:]
        box_sz = int(box_sz)
        start_h = h//2-(box_sz//2)
        start_w = w//2-(box_sz//2)    
        x[start_h:start_h+box_sz,start_w:start_w+box_sz] = val
        return x
        
def create_operator(config):
    if config['type'] == 'gaussian_blur':
        MIN_STD = 0.3 # below this the filter is truncated and we get identity mapping
        if config['scheduling'] == 'linear':
            std_schedule = lambda t: (config['max_std'] - MIN_STD) * t + MIN_STD
            return GaussianBlurOperator(config['kernel_size'], std_schedule)
        elif config['scheduling'] == 'from_file':
            return GaussianBlurOperator(config['kernel_size'], std_schedule=None, from_file=config['schedule_path'])
        elif config['scheduling'] == 'fixed':
            std_schedule = lambda t: config['max_std']
            return GaussianBlurOperator(config['kernel_size'], std_schedule)
    elif config['type'] == 'inpainting':
        # set defaults
        if 'img_size' not in config:
            config['img_size'] = 256
        if 'box_size' not in config:
            config['box_size'] = None
        if 'mask_min_std' not in config:
            config['mask_min_std'] = None
        if 'mask_min_std' not in config:
            config['mask_max_std'] = None
        if 'mask_pow' not in config:
            config['mask_pow'] = None
        if config['scheduling'] == 'linear':
            config['mask_schedule'] = (lambda t: t)
            config['from_file'] = None
        elif config['scheduling'] == 'from_file':
            config['mask_schedule'] = None
            config['from_file'] = config['schedule_path']
        elif config['scheduling'] == 'fixed':
            config['mask_schedule'] = (lambda t:  1.0)
            config['from_file'] = None
        else:
            raise ValueError('Unkown inpainting scheduling type.')
        return InpaintingOperator(**config)
    else:
        raise ValueError('Unsupported operator in config.')
        
def create_noise_schedule(config):
    noise_schedule = NoiseScheduler(config['sigma_min'], config['sigma_max'])
    return noise_schedule
