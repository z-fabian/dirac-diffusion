import torch
from models.ncsn_models import SongUNet as NCSN
import lightning.pytorch as pl
from data.metrics import psnr, LPIPS, ssim
from data.operators import create_operator, create_noise_schedule
from scripts.utils import load_config_from_yaml

LOSS_KEYS = [
            "mse_loss", 
            "weighted_mse_loss", 
            "mse_x_loss", 
            "weighted_mse_x_loss",
            "inc_recon_loss", 
            "weighted_inc_recon_loss",
            "psnr_loss",
            "ssim_loss", 
            "lpips_loss",
]

class NCSN_Module(pl.LightningModule):

    def __init__(
        self,
        dt,
        loss_type,
        lr,
        experiment_config_file=None,
        operator_config=None,
        noise_config=None,
        model_arch='base',
        residual_prediction=True,
        model_conditioning='noise',
        weight_decay=0.0,
        logger_type='wandb',
        full_val_only_last_epoch=True,
        num_log_images=10,
    ):
        """
        Pytorch Lightning module to train and evaluate HUMUS-Net. 
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_pools: Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            sens_chans: Number of channels for sensitivity map U-Net.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
        """    
        super().__init__()
        self.save_hyperparameters()
        self.full_val_only_last_epoch = full_val_only_last_epoch
        self.logger_type = logger_type
        self.num_log_images = num_log_images
        if self.logger_type == 'wandb':
            global wandb
            import wandb
            
        self.dt = dt 
        if experiment_config_file is not None:       
            exp_config = load_config_from_yaml(experiment_config_file)
            operator_config = exp_config['operator']
            noise_config = exp_config['noise']
        else:
            assert operator_config is not None, 'Must provide operator config.'
            assert noise_config is not None, 'Must provide noise config.'
            
        print('Loaded operator: ', operator_config)
        print('Loaded noise schedule: ', noise_config)

        self.fwd_operator = create_operator(operator_config)
        self.noise_schedule = create_noise_schedule(noise_config)
        self.residual_prediction = residual_prediction
        self.lr = lr
        self.weight_decay = weight_decay
        self.model_conditioning = model_conditioning
        self.loss_type = loss_type
                        
        model_arch = load_config_from_yaml('configs/models/models.yaml')[model_arch]
        self.denoiser_fn = NCSN(**model_arch)
        
        # move VGG model to GPU for LPIPS metric eval 
        self.lpips = LPIPS('vgg')
        
    def get_prediction(self, y, cond_labels):
        x0 = self.denoiser_fn(y, cond_labels, class_labels=None)
        if self.residual_prediction:
            x0 += y
        return x0.type(torch.float32)
        
    def get_losses(self, batch):
        if self.model_conditioning == 'noise':
            cond = batch['noise_std']
            weights = 1 / (cond**2 + 1e-9)
        elif self.model_conditioning == 'time':
            cond = batch['t_mapped']
            weights = torch.clip(1 / (cond**2 + 1e-9), 1.0, 100.0)
        
        x0 = self.get_prediction(y=batch['degraded_noisy'], cond_labels=cond)
        b = x0.shape[0]
        weighted_mse_loss = 0
        mse_loss = 0
        weighted_mse_x_loss = 0
        mse_x_loss = 0
        psnr_loss = 0
        ssim_loss = 0
        lpips_loss = 0
        inc_recon_loss = 0
        weighted_inc_recon_loss = 0
        for i in range(b): # Replace with batch implementation!
            # get degraded reconstructions
            y_hat = self.fwd_operator(data=x0[i], t=batch['t_this'][i])
            y_hat_next = self.fwd_operator(data=x0[i], t=batch['t_next'][i])
            y = batch['degraded_this'][i]
            y_next = batch['degraded_next'][i]
            
            # get mse losses
            error = (y - y_hat).pow(2).sum() 
            mse_loss += error
            weighted_mse_loss += error * weights[i]
            
            # get mse_x losses
            error_x = (x0[i] - batch['clean'][i]).pow(2).sum()
            mse_x_loss += error_x
            weighted_mse_x_loss += error_x * weights[i]
            
            # get incremental reconstruction loss
            error_next = (y_next - y_hat_next).pow(2).sum() 
            inc_recon_loss += error_next
            inc_recon_weight = weights[i]
            weighted_inc_recon_loss += error_next * inc_recon_weight
                                                                 
            # get image quality losses
            psnr_loss += psnr(x0[i], batch['clean'][i])
            ssim_loss += ssim(x0[i], batch['clean'][i])            
            lpips_loss += self.lpips(x0[i], batch['clean'][i]) 
        mse_loss /= b
        weighted_mse_loss /= b
        mse_x_loss /= b
        weighted_mse_x_loss /= b
        psnr_loss /= b
        ssim_loss /= b
        lpips_loss /= b
        inc_recon_loss /= b
        weighted_inc_recon_loss /= b
        return {
            'prediction': x0,
            'mse_loss': mse_loss, 
            'weighted_mse_loss': weighted_mse_loss, 
            'mse_x_loss': mse_x_loss, 
            'weighted_mse_x_loss': weighted_mse_x_loss, 
            'inc_recon_loss': inc_recon_loss, 
            'weighted_inc_recon_loss': weighted_inc_recon_loss, 
            'psnr_loss': psnr_loss,
            'ssim_loss': ssim_loss,
            'lpips_loss': lpips_loss,
        }
    
    def inference(self, degraded_noisy):
        # This is a simplified inference function to evaluate the model during training
        # See reconstruction/reverse_diffusion.py for a more general implementation
        num_steps = int(1.0 / self.dt)
        y = degraded_noisy.clone()
        for i in range(num_steps):
            # get new variables
            t_this = torch.tensor(1.0 - self.dt * i).to(y.device)
            t_next = t_this - self.dt
            std_this = self.noise_schedule.get_std(t_this).clone().detach().to(y.device)
            std_next = self.noise_schedule.get_std(t_next).clone().detach().to(y.device)
            if self.model_conditioning == 'noise':
                cond = std_this.view(1, )
            elif self.model_conditioning == 'time':
                cond = t_this.view(1, )
            x0_pred = self.get_prediction(y=y, cond_labels=cond)[0]
            
            # diffusion term
            diffusion = torch.randn_like(degraded_noisy) * torch.sqrt(std_this**2 - std_next**2) # This is 0 for InDI as epsilon is constant
            
            # denoising term
            y_hat = self.fwd_operator(data=x0_pred, t=t_this).unsqueeze(0)
            if std_this != std_next: # This is 0 for InDI as epsilon is constant
                denoising = (y_hat - y)
                denoising *= (std_this**2 - std_next**2) / std_this**2
            else:
                denoising = 0
            
            # degradation update term
            degr_update = self.fwd_operator(x0_pred, t=t_next).unsqueeze(0) - y_hat

            # update y
            y = y + degr_update + denoising + diffusion
        return y           
                                                     
    def training_step(self, batch, batch_idx):
        optimizer_idx=0
        if optimizer_idx == 0:
            losses = self.get_losses(batch)
            
            if batch_idx < self.num_log_images and self.global_rank == 0:
                noised_im = batch['degraded_noisy'][0].unsqueeze(0).detach()
                denoised_im = losses['prediction'][0].unsqueeze(0).detach()
                self.log_image(f"train/{batch_idx}/target", batch["clean"][0].unsqueeze(0))
                self.log_image(f"train/{batch_idx}/degraded_noisy", noised_im)
                self.log_image(f"train/{batch_idx}/prediction", denoised_im)

            self.log_losses(losses, 'train')
            return losses[self.loss_type]
        
    def validation_step(self, batch, batch_idx):
        losses = self.get_losses(batch)
        
        if batch_idx == 0:
            # run full validation on a single image
            degraded_y = self.fwd_operator(batch['clean'][0], torch.ones_like(batch['t_this'][0]))
            z, _ = self.noise_schedule(1.0, batch['clean'][0].shape)
            degraded_noisy = degraded_y + z.to(batch['clean'][0].device)
            degraded_noisy = degraded_noisy.unsqueeze(0)
            
            recon = self.inference(degraded_noisy)
            self.log_image(f"val_recon/{batch_idx}/target", batch["clean"][0].unsqueeze(0))
            self.log_image(f"val_recon/{batch_idx}/degraded_noisy", degraded_noisy)
            self.log_image(f"val_recon/{batch_idx}/recon", recon)
            self.log("val_recon/psnr_x", psnr(recon[0], batch["clean"][0]), on_epoch=True, sync_dist=True)
            self.log("val_recon/ssim_x", ssim(recon[0], batch["clean"][0]), on_epoch=True, sync_dist=True)
            self.log("val_recon/lpips", self.lpips(recon[0], batch["clean"][0]), on_epoch=True, sync_dist=True)
        
        if batch_idx < self.num_log_images and self.global_rank == 0:
            noised_im = batch['degraded_noisy'][0].unsqueeze(0).detach()
            denoised_im = losses['prediction'][0].unsqueeze(0).detach()
            self.log_image(f"val/{batch_idx}/target", batch["clean"][0].unsqueeze(0))
            self.log_image(f"val/{batch_idx}/degraded_noisy", noised_im)
            self.log_image(f"val/{batch_idx}/prediction", denoised_im)
            

        self.log_losses(losses, 'val')
            
        return {"val_loss": losses[self.loss_type]}
    
    def log_losses(self, losses, folder_name):
        for key in LOSS_KEYS:
            self.log('{}/{}'.format(folder_name, key), losses[key], sync_dist=True)
    
    def on_train_start(self):
        self.optimizers().param_groups[0]['lr'] = self.lr    

    def configure_optimizers(self):
        optims = []
        optim_denoiser = torch.optim.Adam(
            self.denoiser_fn.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        optims.append(optim_denoiser)

        return optims, []

    def log_image(self, name, image):
        if self.logger_type == 'wandb':
            # wandb logging
            self.logger.experiment.log({name:  wandb.Image(image)})
        else:
            # tensorboard logging (default)
            self.logger.experiment.add_image(name, image, global_step=self.global_step)