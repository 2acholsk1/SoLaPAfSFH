
import transformers

import lightning.pytorch as pl
import lightning.pytorch.loggers
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
import torch
import lightning.pytorch as pl
import monai
import torchmetrics
import torch.nn.functional as F
from typing import List, Optional
from solapafsfh.metrics.iou_metric import IOUMetric
import lightning.pytorch as pl
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image








class SegmentationModel(pl.LightningModule):
    def __init__(self,
                 model_name: str = "nvidia/mit-b0",
                 input_channels: str = 3,
                 classes: List[str] = ['background', 'lawn', 'paving'],
                 loss_func: str = "builtin from SegFormer",
                 lr: float = 0.001):
        super().__init__()
        
        self.model_name = model_name
        self._lr = lr

        config = transformers.SegformerConfig(num_channels=input_channels).from_pretrained(self.model_name)
        config.id2label = {idx: classname for idx, classname in enumerate(classes)}
        config.num_labels = len(config.id2label)
        config.label2id = {v: k for k, v in config.id2label.items()}

        self.network = transformers.SegformerForSemanticSegmentation(config).from_pretrained(self.model_name, ignore_mismatched_sizes=True,
                                                         num_labels=len(config.id2label), id2label=config.id2label, label2id=config.label2id,
                                                         reshape_last_stage=True).train()
        
        
        metrics = torchmetrics.MetricCollection({
            'iou' : IOUMetric(num_classes=config.num_labels),
        })
        self.dice_loss = monai.losses.DiceLoss(
            include_background=True,
            to_onehot_y = True,
            softmax= True,
        ) 
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=config.num_labels)
        
        self.train_metrics = metrics.clone('train_')
        self.valid_metrics = metrics.clone('valid_')
        self.test_metrics = metrics.clone('test_')
        
        self.save_hyperparameters()

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        if labels is None:
            return self.network.forward(x)
        return self.network.forward(x, labels=labels)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Optional[torch.Tensor]:
        images, masks = batch['pixel_values'], batch['labels']
        labels = masks.long()
        outputs = self.forward(images, labels) 

        loss, logits = outputs[0], outputs[1]
        upsampled_logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        predicted = upsampled_logits
        loss = self.dice_loss(predicted, labels.unsqueeze(1))
        if torch.isinf(loss):
            return None
        self.accuracy.update(predicted, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_acc', self.accuracy, prog_bar=True)
        self.log_dict(self.train_metrics(predicted, labels))
        
        return loss


    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        images, masks = batch['pixel_values'], batch['labels']
        labels = masks.long()
        outputs = self.forward(images, labels) 

        loss, logits = outputs[0], outputs[1]
        upsampled_logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        predicted = upsampled_logits
        loss = self.dice_loss(predicted, labels.unsqueeze(1))

        if torch.isinf(loss):
            return None
        self.accuracy.update(predicted, labels)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('valid_acc', self.accuracy, prog_bar=True)
        self.log_dict(self.valid_metrics(predicted, labels))
        

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        images, masks = batch['pixel_values'], batch['labels']
        labels = masks.long()
        outputs = self.forward(images, labels)
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        predicted = upsampled_logits
        loss = self.dice_loss(predicted, labels.unsqueeze(1))

        if torch.isinf(loss):
            return None
        self.accuracy.update(predicted, labels)
        self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_acc', self.accuracy, prog_bar=True)
        self.log_dict(self.test_metrics(predicted, labels))
        

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self._lr)


class LawnAndPavingDataModule(pl.LightningDataModule):
    def __init__(self, data_path, model_name = None, batch_size = 8):
        super().__init__()
        self._data_path = data_path
        self.feature_extractor = transformers.SegformerFeatureExtractor().from_pretrained(model_name)
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.batch_size = batch_size

    def setup(self, stage):
        dataset_path = Path(self._data_path)
        train_path = sorted((dataset_path / 'train' / 'images').glob('*.jpg'))
        train_path, valid_path = train_test_split(train_path, test_size=0.2, random_state=42)
        valid_path, test_path = train_test_split(valid_path, test_size=0.5, random_state=42)

        self.train_dataset = LawnAndPavingDataset(train_path, self.feature_extractor)
        self.valid_dataset = LawnAndPavingDataset(valid_path, self.feature_extractor)
        self.test_dataset = LawnAndPavingDataset(test_path, self.feature_extractor)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8)

class LawnAndPavingDataset(Dataset):
    def __init__(self, images_paths: list[Path], feature_extractor: transformers.SegformerFeatureExtractor):
        self._images_paths = images_paths
        self._feature_extractor = feature_extractor

    def __len__(self):
        return len(self._images_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path = self._images_paths[index]
        image = Image.open(image_path).convert('RGB')
        
        mask_path = image_path.parent.parent / 'masks' / f'{image_path.stem}_mask.png'
        mask = Image.open(mask_path)
        
        encoded_input = self._feature_extractor(image, mask, return_tensors="pt")
        for k, _ in encoded_input.items():
          encoded_input[k].squeeze_()
        return encoded_input



model_summary_callback = pl.callbacks.ModelSummary(max_depth=-1)

early_stopping_callback = EarlyStopping(
    monitor="valid_loss",
    patience=10,
    mode="min"
)

checkpoint_callback = ModelCheckpoint(
    monitor="valid_loss",
    dirpath="./checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    mode="min"
)

# logger = lightning.pytorch.loggers.NeptuneLogger(
#     project=config.logger.project,
#     log_model_checkpoints=False,
#     tags=["solapafsh_1"]
#     )

trainer = pl.Trainer(
    # logger=logger,
    callbacks=[model_summary_callback, early_stopping_callback, checkpoint_callback],
    accelerator='gpu',
    benchmark=True,
    log_every_n_steps=20,
    max_epochs=30)


pl.seed_everything(42, workers=True)
model = SegmentationModel()

data_module = LawnAndPavingDataModule(
    data_path=Path(__file__).parent.parent.parent.joinpath("data").absolute(),
    model_name=model.model_name
)



torch.set_float32_matmul_precision('medium')
trainer.fit(model, datamodule=data_module)
trainer.test(model, data_module, ckpt_path='best')

# logger.finalize('success')
