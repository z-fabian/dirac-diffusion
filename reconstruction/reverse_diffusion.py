from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import torch
from pl_modules.ncsn_module import NCSN_Module
from data.metrics import psnr, mse, LPIPS, ssim, nmse
import yaml
from scripts.utils import str2int, load_np_to_tensor, load_config_from_yaml
import torch_fidelity
from tqdm import tqdm

MIN_BEST = {'mse', 'mse_x0h', 'mse_y_degraded', 'mse_y_x0h_degraded', 'mse_y_noisy', 'mse_y_x0h_noisy', 'lpips', 'lpips_x0h'}
MAX_BEST = {'ssim', 'ssim_x0h', 'psnr', 'psnr_x0h'}

class ReverseDiffusion:

    def __init__(self, 
                 config_dict,
                 device='cuda:0',
                 output_path='outputs', 
                 experiment_name='reconstruction'
                ):
        self.device = device
        
        self.config = config_dict
        self.model_ckpt_path = config_dict.pop('model_ckpt')
        
        train_params = torch.load(self.model_ckpt_path, map_location='cpu')['hyper_parameters']
        if 'experiment_config_file' in train_params:
            experiment_config_file = train_params['experiment_config_file']
            exp_config = load_config_from_yaml(experiment_config_file)
            self.operator_config = exp_config['operator']
            self.noise_config = exp_config['noise']
        else:
            assert 'operator_config' in train_params and 'noise_config' in train_params
            self.operator_config = train_params['operator_config']
            self.noise_config = train_params['noise_config']
        
        self.model = NCSN_Module.load_from_checkpoint(self.model_ckpt_path, operator_config=self.operator_config, noise_config=self.noise_config).to(self.device)
        self.model.eval()
        self.output_path = Path(output_path)
        self.experiment_name = experiment_name
        self.dataloader = None
        self.fwd_operator = self.model.fwd_operator
        self.noise_schedule = self.model.noise_schedule
        self.lpips = LPIPS('vgg')
        
        if 'start_t' in self.config:
            self.start_t = self.config['start_t']
        else:
            self.start_t = 1.0
            
    def load_data(self, dataset):
        self.dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=4,
        ) 
            
    def run_single(self, y, x=None, save=False, save_intermediate_times=None, image_id='example'):
        results = self.reverse_diff(y=y, x=x, model=self.model, image_id=image_id, save_intermediate_times=save_intermediate_times, **self.config)
        if save:
            self.save_result(results, self.output_path, self.experiment_name, image_id)
        return results
    
    def run_batch(self, num_images, save=False, evaluate=False, evaluate_gen_metrics=False, save_intermediate_times=None, im_range=None):
        if self.dataloader is None:
            raise ValueError('Load data first before running reconstruction.')
        
        fnames = {}
        
        if im_range is None:
            im_range = (0, num_images-1)

        # use tqdm to iterate over dataset
        tqdm_loader = tqdm(self.dataloader, total=num_images, desc='Reconstructing images')
            
        for i, sample in enumerate(tqdm_loader):                
            if i >= num_images:
                break
            if (im_range[0] > i) or (i > im_range[1]):
                continue
            im_batch = sample
            fnames[i] = im_batch['fname'][0]
            x_batch = im_batch['clean'].to(self.device)

            degraded_y_batch = self.fwd_operator(x_batch, self.start_t * torch.ones((1, 1)).to(self.device))
            z_batch, _ = self.noise_schedule(self.start_t, x_batch.shape, seed=str2int(im_batch['fname'][0]))
            y_batch = degraded_y_batch + z_batch.to(x_batch.device)

            self.run_single(y=y_batch, x=x_batch, save=save, save_intermediate_times=save_intermediate_times, image_id=i)

        if save:
            print('Saving results to '+str(self.output_path))
            # Save list of images in root dir
            with open(self.output_path / 'images.yml', 'w') as outfile:
                yaml.dump(fnames, outfile)
            
            # Save reconstruction config in experiment folder
            with open(self.output_path / self.experiment_name / 'recon_config.yml', 'w') as outfile:
                yaml.dump(self.config, outfile)
                
            if evaluate:
                print('Evaluating results.')
                self.evaluate(self.output_path, self.experiment_name, device=self.device, evaluate_gen_metrics=evaluate_gen_metrics)
                            
    def reverse_diff(self, y,                               # Batch of initial degraded, noisy image (t=1.0)
                     dt,                                    # Time-step
                     model,                                 # Trained model
                     degr_update_method='look_ahead',       # How to calculate reconstruction term. Dict. See Appendix I for details.
                     dc_corr=None,                          # Configuration of data consistency correction via guided diffusion. Dict.        
                     fwd_operator=None,                     # Load operator from pre-trained model if not given.
                     noise_schedule=None,                   # Load noise schedule from pre-trained model if not given.
                     x=None,                                # Batch of corresponding ground truth images. Used for metric calculations if given.
                     stop_t=0.0,                            # Reverse diffusion stopping time. 
                     x0_pred_in_last_step=True,             # If True, x0_pred is used in the last step. If False, y_recon is used.
                     save_intermediate_times=None,          # If set to an integer, this many times will intermediate reconstructions be saved (image only)
                     image_id=None,                         # Used to generate file name for intermediate reconstructions
                    ):
        # Set up parameters
        b = y.shape[0]
        assert b==1

        if fwd_operator is None:
            fwd_operator = self.fwd_operator

        if noise_schedule is None:
            noise_schedule = self.noise_schedule

        if dc_corr is None:
            dc_corr = {'use': False}

        # Reconstruction
        y_recon = y.clone() # Initialize reconstruction with degraded, noisy image
        num_steps = int(self.start_t / dt)
        metrics = {
            'ssim': [], 
            'psnr': [], 
            'mse': [], 
            'lpips': [], 
            'mse_y_degraded': [], 
            'mse_y_noisy': [], 
            'ssim_x0h': [], 
            'psnr_x0h': [], 
            'mse_x0h': [], 
            'lpips_x0h': [], 
            'mse_y_x0h_degraded': [], 
            'mse_y_x0h_noisy': []
        }
        ts = []
        # Evaluate metrics on degraded, noisy initial image
        if x is not None:
            metrics['ssim'].append(ssim(x, y).mean().detach().cpu().numpy())
            metrics['mse'].append(mse(x, y).detach().mean().cpu().numpy())
            metrics['psnr'].append(psnr(x, y).detach().mean().cpu().numpy())
            metrics['lpips'].append(self.lpips(x, y).detach().mean().squeeze().cpu().numpy())
                    
        if save_intermediate_times is not None:
            assert image_id is not None
            t_saves = list(np.linspace(stop_t, self.start_t, save_intermediate_times))
            
        for i in range(num_steps):
            # get new variables
            t_this = torch.tensor(self.start_t - dt * i).to(y_recon.device)
            t_next = t_this - dt
            std_this = torch.tensor(noise_schedule.get_std(t_this)).repeat(b).to(y_recon.device)
            std_next = torch.tensor(noise_schedule.get_std(t_next)).repeat(b).to(y_recon.device)

            if dc_corr['use']:
                y_recon.requires_grad = True
                model.zero_grad()
                
            if self.model.model_conditioning == 'noise':
                cond = std_this.view(b, )
            elif self.model.model_conditioning == 'time':
                cond = t_this.view(b, )
            x0_pred = model.get_prediction(y=y_recon, cond_labels=cond)

            # For dc correction
            if dc_corr['use']:
                if dc_corr['noiseless_meas_available']:
                    y_noiseless = fwd_operator(x, self.start_t * torch.ones((1, 1)).to(x0_pred.device))
                    error = y_noiseless - fwd_operator(x0_pred, self.start_t * torch.ones_like(t_this))
                else:
                    error = y - fwd_operator(x0_pred, self.start_t * torch.ones_like(t_this))
                dc_loss = error.pow(2).sum()
                grad_y = torch.autograd.grad(dc_loss, y_recon)[0]
                y_recon.requires_grad = False
                if dc_corr['scale_relative']:
                    # scale step size with error 
                    scaler = dc_corr['step'] / torch.sqrt(dc_loss)
                else:
                    # scale step size with noise std
                    std_max = noise_schedule(self.start_t, (1,))[1]
                    scaler = dc_corr['step'] * 0.5 / std_max**2 
                dc_correction = scaler * grad_y

            # Time derivatives if needed
            if degr_update_method['type']=='look_back_exact':
                func = lambda t: fwd_operator(x0_pred.detach(), t)
                dt_y = torch.autograd.functional.jacobian(func, t_this)

            with torch.no_grad():
                # diffusion term
                diffusion = torch.randn_like(y_recon) * torch.sqrt(std_this**2 - std_next**2).view(b, 1, 1, 1)

                # denoising term
                y_hat = fwd_operator(x0_pred, t_this)
                if std_this != std_next:
                    denoising = (y_hat - y_recon)
                    denoising *= ((std_this**2 - std_next**2) / std_this**2).view(b, 1, 1, 1)
                    if dc_corr['use']:
                        denoising += (std_this**2 - std_next**2) * dc_correction
                else:
                    denoising = 0

                # degradation update term (incremental reconstruction)
                if degr_update_method['type']=='look_ahead':
                    degr_update = fwd_operator(x0_pred, t_next) - y_hat
                elif degr_update_method['type']=='small_look_ahead':
                    del_t = degr_update_method['del_t']
                    degr_update = (fwd_operator(x0_pred, t_this - del_t) - y_hat) * dt / del_t
                elif degr_update_method['type']=='look_back':
                    if t_this + dt >= 1.0: # In the first update do look ahead approach
                        degr_update = fwd_operator(x0_pred, t_next) - y_hat
                    else:
                        degr_update = y_hat - fwd_operator(x0_pred, t_this + dt)
                elif degr_update_method['type']=='small_look_back':
                    del_t = degr_update_method['del_t']
                    if t_this + del_t >= 1.0: # In the first update do look ahead approach
                        degr_update = fwd_operator(x0_pred, t_next) - y_hat
                    else:
                        degr_update = (y_hat - fwd_operator(x0_pred, t_this + del_t)) * dt / del_t
                elif degr_update_method['type']=='look_back_exact':
                    degr_update = - dt * dt_y
                elif degr_update_method['type']=='identity':
                    degr_update = 0.0

                # update y
                final_iteration = ((i == num_steps - 1) or (stop_t > t_next))
                if final_iteration: # this is the last step
                    if x0_pred_in_last_step:
                        y_recon = x0_pred.detach() # Don't noise and degrade in last step
                    else:
                        y_recon = y_recon + degr_update + denoising + diffusion
                else:
                    y_recon = y_recon + degr_update + denoising + diffusion

                # Evaluate metrics for this step
                y_recon_degraded = fwd_operator(y_recon, self.start_t * torch.ones((1, 1)).to(y_recon.device))
                x0h_degraded = fwd_operator(x0_pred, self.start_t * torch.ones((1, 1)).to(x0_pred.device))
                if x is not None:
                    # we have ground truth, find metrics
                    metrics['ssim'].append(ssim(x, y_recon).mean().cpu().numpy())
                    metrics['mse'].append(mse(x, y_recon).mean().cpu().numpy())
                    metrics['psnr'].append(psnr(x, y_recon).mean().cpu().numpy())
                    metrics['lpips'].append(self.lpips(x, y_recon).mean().squeeze().cpu().numpy())
                    metrics['ssim_x0h'].append(ssim(x, x0_pred).mean().cpu().numpy())
                    metrics['mse_x0h'].append(mse(x, x0_pred).mean().cpu().numpy())
                    metrics['psnr_x0h'].append(psnr(x, x0_pred).mean().cpu().numpy())
                    metrics['lpips_x0h'].append(self.lpips(x, x0_pred).mean().cpu().numpy())
                    degraded_y = fwd_operator(x, self.start_t * torch.ones((1, 1)).to(x.device))
                    metrics['mse_y_degraded'].append(mse(degraded_y, y_recon_degraded).mean().cpu().numpy())
                    metrics['mse_y_x0h_degraded'].append(mse(degraded_y, x0h_degraded).mean().cpu().numpy())

                metrics['mse_y_noisy'].append(mse(y, y_recon_degraded).mean().cpu().numpy())
                metrics['mse_y_x0h_noisy'].append(mse(y, x0h_degraded).mean().cpu().numpy())

                ts.append(t_next.cpu().numpy())
                
                # Save intermediate reconstruction if needed
                if save_intermediate_times is not None:
                    if (True in [(t_this >= t > t_next) for t in t_saves]) or final_iteration:
                        self.save_intermediate_recons(y_recon, x0_pred, self.output_path, self.experiment_name, image_id, t_next)
                
                if stop_t > t_next:
                    return {'noisy': y.detach(), 'recon': y_recon, 'target': x, 'metrics': metrics, 'ts': ts}
        return {'noisy': y.detach(),'recon': y_recon, 'target': x, 'metrics': metrics, 'ts': ts}
            
    @staticmethod
    def save_result(result_dict, root, expname='reconstruction', image_id=0):
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / 'target').mkdir(parents=True, exist_ok=True)
        (Path(root) / 'noisy').mkdir(parents=True, exist_ok=True)
        (Path(root) / expname).mkdir(parents=True, exist_ok=True)
        image_id = str(image_id)
        if not (Path(root) / 'target' / (image_id+'.npy')).exists():
            np.save(str((Path(root) / 'target' / (image_id+'.npy'))), result_dict['target'][0].permute(1,2,0).cpu().numpy())
            plt.imsave(str((Path(root) / 'target' / (image_id+'.png'))), result_dict['target'][0].permute(1,2,0).cpu().numpy().clip(0.0, 1.0))

        if not (Path(root) / 'noisy' / (image_id+'.npy')).exists():
            np.save(str((Path(root) / 'noisy' / (image_id+'.npy'))), result_dict['noisy'][0].permute(1,2,0).cpu().numpy())
            plt.imsave(str((Path(root) / 'noisy' / (image_id+'.png'))), result_dict['noisy'][0].permute(1,2,0).cpu().numpy().clip(0.0, 1.0))

        np.save(str(Path(root) / expname / (image_id+'.npy')), result_dict['recon'][0].permute(1,2,0).cpu().numpy())    
        plt.imsave(str(Path(root)/ expname / (image_id+'.png')), result_dict['recon'][0].permute(1,2,0).cpu().numpy().clip(0.0, 1.0))

        res = {k:v for k, v in result_dict['metrics'].items()}
        res['ts'] = result_dict['ts']
        np.save(str(Path(root)/ expname / (image_id+'_metrics.npy')), res)
        
    def save_intermediate_recons(self, y_recon, x_pred, root, expname, image_id, t):
        time_folder = "{:.4f}".format(t)
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / expname).mkdir(parents=True, exist_ok=True)
        (Path(root) / expname / 'intermediate').mkdir(parents=True, exist_ok=True)
        (Path(root) / expname / 'intermediate' / time_folder).mkdir(parents=True, exist_ok=True)
        (Path(root) / expname / 'intermediate' / time_folder / 'yt').mkdir(parents=True, exist_ok=True)
        (Path(root) / expname / 'intermediate' / time_folder / 'xpred').mkdir(parents=True, exist_ok=True)
        image_id = str(image_id)
        plt.imsave(str(Path(root)/ expname / 'intermediate' / time_folder / 'yt' /(image_id+'.png')), y_recon[0].permute(1,2,0).cpu().numpy().clip(0.0, 1.0))
        plt.imsave(str(Path(root)/ expname / 'intermediate' / time_folder / 'xpred' /(image_id+'.png')), x_pred[0].permute(1,2,0).cpu().numpy().clip(0.0, 1.0))
        with open((Path(root) / expname / 'intermediate' / time_folder / 't.yaml'), 'w') as outfile:
            yaml.dump({'t': float(t)}, outfile)

    @staticmethod
    def evaluate(output_dir, experiment_name, device='cuda:0', evaluate_gen_metrics=False):
        lpips = LPIPS('vgg')
        metric_available = True

        (Path(output_dir) / experiment_name / 'eval').mkdir(parents=True, exist_ok=True)
        target_files = sorted(list((Path(output_dir) / 'target').glob('*.npy')))
        recon_files = sorted([f for f in list((Path(output_dir) / experiment_name).glob('*.npy')) if 'metrics' not in str(f.stem)])
        if metric_available:
            metric_files = sorted([f for f in list((Path(output_dir) / experiment_name).glob('*.npy')) if 'metrics' in str(f.stem)])
        else:
            metric_files = [None for _ in recon_files]
        ssim_arr = []
        psnr_arr = []
        nmse_arr = []
        lpips_arr = []
        if metric_available:
            results_mean = {}
            assert len(target_files) == len(recon_files) == len(metric_files)
        assert len(target_files) == len(recon_files)

        num_images = len(target_files)
        print('Number of images found: ', num_images)
        with torch.no_grad():
            for target, recon, metric in zip(target_files, recon_files, metric_files):
                assert str(target.stem) == str(recon.stem)
                target_arr = load_np_to_tensor(target, device)
                recon_arr = load_np_to_tensor(recon, device)
                if metric_available:
                    result = np.load(metric, allow_pickle=True)
                ssim_arr.append(ssim(target_arr, recon_arr).cpu().numpy())
                psnr_arr.append(psnr(target_arr, recon_arr).cpu().numpy())
                nmse_arr.append(nmse(target_arr, recon_arr).cpu().numpy())
                lpips_arr.append(lpips(target_arr, recon_arr).cpu().numpy())
                if metric_available:
                    for k, v in result[()].items():
                        if k == 'ts':
                            results_mean[k] = np.array(v)
                        elif k not in results_mean:
                            results_mean[k] = np.array(v) / num_images
                        elif k != 'ts':
                            results_mean[k] += np.array(v) / num_images

            # Plot aggregated reverse process evolution
            if metric_available:
                results_log = {}
                for k, _ in results_mean.items():
                    if k != 'ts':
                        if len(results_mean['ts']) < len(results_mean[k]):
                            t = [1.0] + list(results_mean['ts'])
                        else:
                            t = results_mean['ts']
                        best_val_ind = np.argmax(results_mean[k]) if k in MAX_BEST else np.argmin(results_mean[k])
                        results_log[k] = {}
                        results_log[k]['best_val'] = float(results_mean[k][best_val_ind])
                        results_log[k]['best_t'] = float(t[best_val_ind])
                        results_log[k]['last_val'] = float(results_mean[k][-1])
                        
                        plt.plot(t, results_mean[k])
                        plt.title(k)
                        plt.savefig(str(Path(output_dir) / experiment_name / 'eval' / (k + '_plot.png')))
                        plt.close()

            # Aggregate results
            ssim_final = np.array(ssim_arr).mean()
            psnr_final = np.array(psnr_arr).mean()
            nmse_final = np.array(nmse_arr).mean()
            lpips_final = np.array(lpips_arr).mean()
            print('SSIM: ', ssim_final, 'PSNR: ', psnr_final, 'NMSE: ', nmse_final, 'LPIPS: ', lpips_final)
            
            # Compute generative/distributional metrics
            if evaluate_gen_metrics:
                gen_res = {}
                # evaluate final reconstructions
                paths = [str(Path(output_dir) / experiment_name), 
                         str(Path(output_dir) / 'target')
                        ]
                gen_res['final'] = torch_fidelity.calculate_metrics(input1=str(Path(output_dir) / experiment_name),
                                                       input2=str(Path(output_dir) / 'target'),
                                                       cuda=True,
                                                       isc=False,
                                                       fid=True,
                                                       kid=False,
                                                       verbose=False,
                                                      )
                # if given, evaluate intermediate reconstructions
                if (Path(output_dir) / experiment_name / 'intermediate').exists():
                    gen_res['intermediate'] = []
                    paths = [f for f in (Path(output_dir) / experiment_name / 'intermediate').iterdir() if f.is_dir()]
                    
                    for p in paths:
                        print('Calculating FID for folder ', str(p))
                        t = load_config_from_yaml(p / 't.yaml')['t']
                        y_res = torch_fidelity.calculate_metrics(input1=str(p / 'yt'),
                                                       input2=str(Path(output_dir) / 'target'),
                                                       cuda=True,
                                                       isc=False,
                                                       fid=True,
                                                       kid=False,
                                                       verbose=False,
                                                      )
                        print('y_res: ', y_res)
                        xpred_res = torch_fidelity.calculate_metrics(input1=str(p / 'xpred'),
                               input2=str(Path(output_dir) / 'target'),
                               cuda=True,
                               isc=False,
                               fid=True,
                               kid=False,
                               verbose=False,
                              )
                        print('xpred_res: ', xpred_res)
                        gen_res['intermediate'].append({'t': t, 'iterates': y_res, 'xpred': xpred_res})
                        
            
            # Add results to dict for saving
            results_final = {}
            results_final['ssim'] = float(ssim_final)
            results_final['psnr'] = float(psnr_final)
            results_final['nmse'] = float(nmse_final)
            results_final['lpips'] = float(lpips_final)
            if evaluate_gen_metrics:
                results_final['generative'] = gen_res
            results = {}
            results['saved_images'] = results_final
            results['logs'] = results_log
            
            with open(Path(output_dir) / experiment_name / 'eval' / 'final_metrics.yml', 'w') as outfile:
                yaml.dump(results, outfile)

            if metric_available:
                np.save(str(Path(output_dir) / experiment_name / 'eval' / 'final_time_evol.npy'), results_mean)