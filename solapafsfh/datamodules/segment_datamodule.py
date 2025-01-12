import lightning.pytorch as pl
import albumentations as A
import albumentations.pytorch.transforms
import timm.data

from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from solapafsfh.datasets.lawn_paving_dataset import LawnAndPavingDataset

class LawnAndPavingDataModule(pl.LightningDataModule):
    def __init__(self, data_path):
        super().__init__()
        self._data_path = data_path

        self.augmentations = A.Compose([
            A.Resize(width=512, height=512),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.RandomRotate90(p=1.0),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD),
            albumentations.pytorch.transforms.ToTensorV2(),

            # I dopiero na końcu normalizacja do ImageNet mean/std
        ])
        
        self.transforms = A.Compose([
            A.Resize(width=512, height=512),
            A.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD),
            albumentations.pytorch.transforms.ToTensorV2(),
        ])
        
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def setup(self, stage):
        dataset_path = Path(self._data_path)
        train_path = sorted((dataset_path / 'train' / 'images').glob('*.jpg'))
        
        train_path, valid_path = train_test_split(train_path, test_size=0.2, random_state=42)
        valid_path, test_path = train_test_split(valid_path, test_size=0.5, random_state=42)

        self.train_dataset = LawnAndPavingDataset(train_path, self.augmentations)
        self.valid_dataset = LawnAndPavingDataset(valid_path, self.transforms)
        self.test_dataset = LawnAndPavingDataset(test_path, self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=32, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32, num_workers=8)
