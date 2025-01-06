import torch
import numpy as np
import albumentations as A
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

class LawnAndPavingDataset(Dataset):
    def __init__(self, images_paths: list[Path], transforms: A.Compose):
        self._images_paths = images_paths
        self._transforms = transforms

    def __len__(self):
        return len(self._images_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path = self._images_paths[index]
        image = np.asarray(Image.open(image_path).convert('RGB'))
        
        mask_path = image_path.parent.parent / 'masks' / f'{image_path.stem}_mask.png'
        mask = np.asarray(Image.open(mask_path))
        
        transformed = self._transforms(image=image, mask=mask)
        
        return transformed['image'], transformed['mask']
