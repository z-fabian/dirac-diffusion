import lightning.pytorch as pl
import torch
from data.image_data import ImageNetDataset, CelebaDataset, FFHQDataset
from data.data_transforms import ImageDataTransform
from scripts.utils import load_config_from_yaml

class CelebaDataModule(pl.LightningDataModule):

    def __init__(
        self, 
        dt,
        experiment_config_file,
        batch_size, 
        sample_rate_dict, 
        distributed_sampler,
        num_workers=4,
    ):

        super().__init__()

        exp_config = load_config_from_yaml(experiment_config_file)
        operator_config = exp_config['operator']
        noise_config = exp_config['noise']
        
        train_transform = ImageDataTransform(is_train=True, operator_config=operator_config, noise_config=noise_config, dt=dt)
        val_transform = ImageDataTransform(is_train=False, operator_config=operator_config, noise_config=noise_config, dt=dt)
        test_transform = ImageDataTransform(is_train=False, operator_config=operator_config, noise_config=noise_config, dt=dt)

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.sample_rate_dict = sample_rate_dict
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler

    def _create_data_loader(
        self,
        data_split,
        data_transform,
        batch_size=None
    ) :
        sample_rate = self.sample_rate_dict[data_split]
        dataset = CelebaDataset(
            split=data_split,
            transform=data_transform,
            sample_rate=sample_rate,
        )

        is_train = (data_split == 'train')
        batch_size = self.batch_size if batch_size is None else batch_size
        sampler = torch.utils.data.DistributedSampler(dataset) if self.distributed_sampler else None
        
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            drop_last=False
        )
        return dataloader

    def train_dataloader(self):
        return self._create_data_loader(data_split='train', data_transform=self.train_transform)

    def val_dataloader(self):
        return self._create_data_loader(data_split='val', data_transform=self.val_transform)

    def test_dataloader(self):
        return self._create_data_loader(data_split='test', data_transform=self.test_transform)

    
class ImageNetDataModule(pl.LightningDataModule):

    def __init__(
        self, 
        dt,
        experiment_config_file,
        batch_size, 
        sample_rate_dict, 
        distributed_sampler,
        num_workers=4,
    ):

        super().__init__()
        
        exp_config = load_config_from_yaml(experiment_config_file)
        operator_config = exp_config['operator']
        noise_config = exp_config['noise']
        
        train_transform = ImageDataTransform(is_train=True, operator_config=operator_config, noise_config=noise_config, dt=dt)
        val_transform = ImageDataTransform(is_train=False, operator_config=operator_config, noise_config=noise_config, dt=dt)
        test_transform = ImageDataTransform(is_train=False, operator_config=operator_config, noise_config=noise_config, dt=dt)
        
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.sample_rate_dict = sample_rate_dict
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler

    def _create_data_loader(
        self,
        data_split,
        data_transform,
        batch_size=None
    ) :
        sample_rate = self.sample_rate_dict[data_split]
        dataset = ImageNetDataset(
            split=data_split,
            transform=data_transform,
            sample_rate=sample_rate,
        )
        batch_size = self.batch_size if batch_size is None else batch_size
        sampler = torch.utils.data.DistributedSampler(dataset) if self.distributed_sampler else None
        
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            drop_last=False
        )
        return dataloader

    def train_dataloader(self):
        return self._create_data_loader(data_split='train', data_transform=self.train_transform)

    def val_dataloader(self):
        return self._create_data_loader(data_split='val', data_transform=self.val_transform)

    def test_dataloader(self):
        return self._create_data_loader(data_split='test', data_transform=self.test_transform)

    
class FFHQDataModule(pl.LightningDataModule):

    def __init__(
        self, 
        dt,
        experiment_config_file,
        batch_size, 
        sample_rate_dict, 
        distributed_sampler,
        num_workers=4,
    ):

        super().__init__()

        exp_config = load_config_from_yaml(experiment_config_file)
        operator_config = exp_config['operator']
        noise_config = exp_config['noise']
        
        train_transform = ImageDataTransform(is_train=True, operator_config=operator_config, noise_config=noise_config, dt=dt)
        val_transform = ImageDataTransform(is_train=False, operator_config=operator_config, noise_config=noise_config, dt=dt)
        test_transform = ImageDataTransform(is_train=False, operator_config=operator_config, noise_config=noise_config, dt=dt)
        
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.sample_rate_dict = sample_rate_dict
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler

    def _create_data_loader(
        self,
        data_split,
        data_transform,
        batch_size=None
    ) :
        sample_rate = self.sample_rate_dict[data_split]
        dataset = FFHQDataset(
            split=data_split,
            transform=data_transform,
            sample_rate=sample_rate,
        )

        batch_size = self.batch_size if batch_size is None else batch_size
        sampler = torch.utils.data.DistributedSampler(dataset) if self.distributed_sampler else None
        
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            drop_last=False
        )
        return dataloader

    def train_dataloader(self):
        return self._create_data_loader(data_split='train', data_transform=self.train_transform)

    def val_dataloader(self):
        return self._create_data_loader(data_split='val', data_transform=self.val_transform)

    def test_dataloader(self):
        return self._create_data_loader(data_split='test', data_transform=self.test_transform)