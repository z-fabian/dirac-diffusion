import torch
import lpips
from torchmetrics.functional import structural_similarity_index_measure as ssim_metric

def psnr(img1, img2):
    assert 3 <= len(img1.shape) <= 4
    if len(img1.shape) == 3:
        mse = torch.mean((img1 - img2) ** 2)
    else:
        mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3))
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def mse(img1, img2):
    assert 3 <= len(img1.shape) <= 4
    if len(img1.shape) == 3:
        return (img1 - img2).pow(2).mean()
    else:
        return torch.mean((img1 - img2).pow(2), dim=(1, 2, 3))
    
def nmse(gt, pred):
    assert 3 <= len(gt.shape) <= 4
    if len(gt.shape) == 3:
        return (gt - pred).pow(2).sum() / gt.pow(2).sum()
    else:
        return torch.sum((gt - pred).pow(2), dim=(1, 2, 3)) / torch.sum(gt.pow(2), dim=(1, 2, 3))

def ssim(img1, img2):
    assert len(img1.shape) == len(img1.shape)
    assert 3 <= len(img1.shape) <= 4
    if len(img1.shape) == 3:
        return ssim_metric((img1.unsqueeze(0)), (img2.unsqueeze(0)), data_range=1.0)
    else:
        return ssim_metric(img1, img2, data_range=1.0)

class LPIPS:
    def __init__(self, net='vgg', input_min=0.0, input_max=1.0):
        self.loss = lpips.LPIPS(net=net)
        self.device = next(self.loss.parameters()).device
        self.input_min = input_min
        self.input_max = input_max
        self.range = self.input_max - self.input_min
        assert self.range > 0.0
        
    def normalize(self, img):
        # Normalize image to [-1, 1] range
        return ((img - self.input_min) / self.range) * 2.0 - 1.0
        
    def __call__(self, img1, img2):
        if self.device != img1.device:
            self.loss = self.loss.to(img1.device)
            self.device = next(self.loss.parameters()).device
            
        # Normalize to [-1, 1]
        return self.loss(
            self.normalize(img1), 
            self.normalize(img2)
        )