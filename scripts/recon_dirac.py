import os, sys
import pathlib
from pathlib import Path
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute())   )
from scripts.utils import load_config_from_yaml
from reconstruction.reverse_diffusion import ReverseDiffusion
from data.data_transforms import ImageDataTransform
from data.image_data import ImageNetDataset, CelebaDataset, FFHQDataset
from argparse import ArgumentParser

def cli_main(args):
    print(args.__dict__)
    config_dict = load_config_from_yaml(args.config_path)
    exp_name = str(Path(args.config_path).stem) if args.experiment_name is None else args.experiment_name
    output_path = os.path.join('outputs/recons', exp_name) if args.output_path is None else args.output_path
            
    print('Running reconstruction {} with config {}.'.format(exp_name, config_dict))
    reconstructor = ReverseDiffusion(config_dict=config_dict, 
                                     device=args.device,
                                     output_path=output_path, 
                                     experiment_name=exp_name,
                                    )

    data_transform = ImageDataTransform(is_train=False, 
                                        operator_config=reconstructor.operator_config, 
                                        noise_config=reconstructor.noise_config,
                                        dt=reconstructor.config['dt']
                                        )

    if args.dataset == 'imagenet':
        eval_dataset = ImageNetDataset(
            split=args.split,
            transform=data_transform,
            num_images_per_class=1,
            )
        assert len(eval_dataset) == 1000, 'Expected 1000 images in test dataset (1 per ImageNet class), got {}'.format(len(eval_dataset))
    elif args.dataset == 'celeba256':
        eval_dataset = CelebaDataset(
            split=args.split,
            transform=data_transform,
            )
    elif args.dataset == 'ffhq':
        eval_dataset = FFHQDataset(
            split=args.split,
            transform=data_transform,
            )
    else:
        raise ValueError('Invalid dataset: {}'.format(args.dataset))
    

    reconstructor.load_data(eval_dataset)
    reconstructor.run_batch(len(eval_dataset), 
                                        save=True, 
                                        evaluate=True,
                                        evaluate_gen_metrics=True,
                                        save_intermediate_times=args.fid_eval_times,
                                        im_range=None,
                                        )

def build_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--config_path', 
        type=str,          
        help='Reconstruction configuration will be loaded from this file.',
    )
    parser.add_argument(
        '--dataset', 
        type=str,          
        help='Dataset to reconstruct. Options: celeba256, ffhq, imagenet.',
    )
    parser.add_argument(
        '--split', 
        type=str,
        default='test',          
        help='Which split to evaluate on. Options: val, test. Default: test.',
    )
    parser.add_argument(
        '--output_path', 
        type=str,    
        default=None,      
        help='Root folder where outputs will be generated. Target and noisy images will be stored here.',
    )
    parser.add_argument(
        '--experiment_name', 
        type=str,          
        default=None,
        help='Used to name the folder for the output reconstructions. If not given, the reconstruction config name will be used.',
    )
    parser.add_argument(
        '--device', 
        default='cuda:0',   
        type=str,          
        help='Which device to run the reconstruction on.',
    )      
    parser.add_argument(
        '--fid_eval_times', 
        default=None,   
        type=int,         
        help='If set to an integer > 2, FID will be evaluated at this many intermediate steps during reconstruction.',
    )
    args = parser.parse_args()
    return args

def run_cli():
    args = build_args()

    # ---------------------
    # RUN RECONSTRUCTION
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()