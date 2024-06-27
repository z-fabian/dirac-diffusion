from pathlib import Path
import random
import torch
from PIL import Image
from scripts.utils import load_config_from_yaml

class ImageNetDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        split,
        sample_rate=None,
        num_images_per_class=None,
        transform=None,
        shuffle_seed=0
    ):

        self.transform = transform
        self.examples = []

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        else:
            assert num_images_per_class is None # either use subsampling, or give number of images per class
        
        root = load_config_from_yaml('configs/data/dataset_config.yaml')['imagenet']
        root_folder = (Path(root) / 'train') if split == 'train' else (Path(root) / 'val') # create test data out of validation dataset
        for subfolder in sorted(list(root_folder.iterdir())):
            if subfolder.is_dir():
                files_in_class = [file for file in sorted(list(Path(subfolder).iterdir())) if file.suffix in ['.JPG', '.JPEG', '.jpg', '.jpeg']]
                if num_images_per_class is None:
                    self.examples.extend(files_in_class)
                else:
                    if split in ['train', 'val']: # use start of the dataset
                        self.examples.extend(files_in_class[:num_images_per_class])
                    elif split == 'test': # use end of dataset
                        self.examples.extend(files_in_class[-num_images_per_class:])
                    else:
                        raise ValueError("Invalid split.")

        # shuffle 
        state = random.getstate()
        random.seed(shuffle_seed)
        random.shuffle(self.examples)
        
        # subsample if desired
        if sample_rate < 1.0: 
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
            
        random.setstate(state)
            
        print('{} images loaded from {} for {} split.'.format(len(self.examples), str(root), split))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        file = self.examples[i]
        fname = str(file).replace(str(file.parents[1]), '').replace(str(file.suffix), '')
        im = Image.open(file).convert("RGB") # Will load grayscale images as RGB!

        if self.transform is None:
            raise ValueError('Must define forward model and pass in DataTransform.')
        else:
            sample = self.transform(im, fname)
            
        return sample
    
    def get_filenames(self):
        filenames = [str(file).replace(str(file.parents[1]), '').replace(str(file.suffix), '') for file in self.examples]
        return filenames
    
class CelebaDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        split,
        transform,
        sample_rate=None,
        train_val_seed=0
    ):

        self.transform = transform
        self.examples = []

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0

        root = load_config_from_yaml('configs/data/dataset_config.yaml')['celeba256']
        for file in list(Path(root).iterdir()):
            if file.suffix in ['.JPG', '.JPEG', '.jpg', '.jpeg']:
                self.examples.append(file)
        self.examples = sorted(self.examples)
        
        # pick desired split
        state = random.getstate()
        random.seed(train_val_seed)
        random.shuffle(self.examples)
        len_train = round(len(self.examples) * 0.8)
        len_val = round(len(self.examples) * 0.1)
        if split == 'train':
            self.examples = self.examples[:len_train]
        elif split == 'val':
            self.examples = self.examples[len_train:len_train+len_val]
        elif split == 'test':
            self.examples = self.examples[len_train+len_val:]
        else:
            raise ValueError('Unknown split in CelebaDataset.')
            
        # restore state
        # this is important so that workers generate the same subsampled datasets later
        random.setstate(state)

        # subsample if desired
        if sample_rate < 1.0: 
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
            
        print('{} images loaded from {} as {} split.'.format(len(self.examples), str(root), split))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        file = self.examples[i]
        fname = str(file).replace(str(file.parents[1]), '').replace(str(file.suffix), '')
        im = Image.open(file).convert("RGB") # Will load grayscale images as RGB!

        if self.transform is None:
            raise ValueError('Must define forward model and pass in DataTransform.')
        else:
            sample = self.transform(im, fname)
            
        return sample
    
    def get_filenames(self):
        filenames = [str(file).replace(str(file.parents[1]), '').replace(str(file.suffix), '') for file in self.examples]
        return filenames
    
class FFHQDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        split,
        transform,
        sample_rate=None,
    ):

        self.transform = transform
        self.examples = []

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0

        root = load_config_from_yaml('configs/data/dataset_config.yaml')['ffhq']
        folders = sorted(list(Path(root).iterdir()))
        assert len(folders) == 70
        if split == 'train':
            folders = folders[8:]
        elif split == 'val':
            folders = folders[1: 8]
        else:
            assert split == 'test'
            folders = [folders[0]]
            
        for folder in folders:
            for file in folder.iterdir():
                if file.suffix in ['.PNG','.png']:
                    self.examples.append(file)
        self.examples = sorted(self.examples)
        
        # subsample if desired
        if sample_rate < 1.0: 
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
            
        print('{} images loaded from {} as {} split.'.format(len(self.examples), str(root), split))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        file = self.examples[i]
        fname = str(file).replace(str(file.parents[1]), '').replace(str(file.suffix), '')
        im = Image.open(file).convert("RGB") # Will load grayscale images as RGB!

        if self.transform is None:
            raise ValueError('Must define forward model and pass in DataTransform.')
        else:
            sample = self.transform(im, fname)
            
        return sample
    
    def get_filenames(self):
        filenames = [str(file).replace(str(file.parents[1]), '').replace(str(file.suffix), '') for file in self.examples]
        return filenames