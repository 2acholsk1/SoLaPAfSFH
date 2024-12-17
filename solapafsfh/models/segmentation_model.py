import torch
import lightning.pytorch as pl
import segmentation_models_pytorch as smp
import torchmetrics
import torch.nn.functional as F
from typing import List, Optional
from solapafsfh.metrics.iou_metric import IOUMetric
from solapafsfh.losses.dice_loss import DiceLoss

class SegmentationModel(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 encoder_name: str,
                 input_channels: str,
                 classes: List[str],
                 loss_func: str,
                 lr: float,
                 ):
        super().__init__()
        
        self._model_name = model_name
        self._encoder_name = encoder_name
        self._input_channels = input_channels
        self._classes = classes
        self._loss_func = loss_func
        self._lr = lr
        
        match self._model_name:
            case "UNet":
                self.network = smp.Unet
            case _:
                raise NotImplementedError(
                    f'Not supported model: {self._model_name}'
                )

        self.network = self.network(
            encoder_name = self._encoder_name,
            encoder_weights = "imagenet",
            in_channels = self._input_channels,
            classes = len(self._classes),
            activation = None
        )
        
        match loss_func:
            case 'CrossEntropy':
                self.loss = torch.nn.CrossEntropyLoss()
            case 'Dice':
                self.loss = DiceLoss()
            case _:
                raise NotImplementedError(
                    f'Not supported loss function: {self._loss_func}'
                )
        
        metrics = torchmetrics.MetricCollection({
            'iou' : IOUMetric(num_classes=len(self._classes)),
        })
        
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=len(self._classes))
        
        self.train_metrics = metrics.clone('train_')
        self.valid_metrics = metrics.clone('valid_')
        self.test_metrics = metrics.clone('test_')
        
        self.save_hyperparameters()

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network.forward(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Optional[torch.Tensor]:
        inputs, labels = batch
        labels = labels.long()
        outputs = self.forward(inputs)
        # label_one_hot = F.one_hot(labels, num_classes=len(self._classes)).permute(0, 3, 1, 2)

        labels = labels.unsqueeze(1)
        loss = self.loss(outputs, labels)

        if torch.isinf(loss):
            return None
        self.accuracy.update(outputs, labels.squeeze())
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_acc', self.accuracy, prog_bar=True)
        self.log_dict(self.train_metrics(outputs, labels.squeeze()))
        
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        inputs, labels = batch
        labels = labels.long()
        outputs = self.forward(inputs)
        # outputs = outputs.squeeze(1)
        # label_one_hot = F.one_hot(labels, num_classes=len(self._classes)).permute(0, 3, 1, 2)
        labels = labels.unsqueeze(1)
        loss = self.loss(outputs, labels)

        self.accuracy.update(outputs, labels.squeeze())
        self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('valid_acc', self.accuracy, prog_bar=True)
        self.log_dict(self.valid_metrics(outputs, labels.squeeze()))
        

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        inputs, labels = batch
        labels = labels.long()
        outputs = self.forward(inputs)
        # label_one_hot = F.one_hot(labels, num_classes=len(self._classes)).permute(0, 3, 1, 2)
        labels = labels.unsqueeze(1)

        loss = self.loss(outputs, labels)

        self.accuracy.update(outputs, labels.squeeze())
        self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_acc', self.accuracy, prog_bar=True)
        self.log_dict(self.test_metrics(outputs, labels.squeeze()))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr)
