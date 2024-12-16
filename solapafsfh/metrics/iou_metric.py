import torch
import torch.nn.functional as F
from torchmetrics import Metric

class IOUMetric(Metric):
    def __init__(self, num_classes: int, smooth: float = 1e-8):
        super().__init__()
        self._num_classes = num_classes
        self._smooth = smooth
        
        self.add_state('intersection', torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('union', torch.zeros(num_classes), dist_reduce_fx='sum')

    def update(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs = torch.argmax(inputs, dim=1)

        inputs = F.one_hot(inputs, num_classes=self._num_classes).permute(0, 3, 1, 2)
        targets = F.one_hot(targets, num_classes=self._num_classes).permute(0, 3, 1, 2)

        inputs = inputs.reshape(-1, self._num_classes)
        targets = targets.reshape(-1, self._num_classes)

        intersection = torch.sum(inputs * targets, dim=0)
        union = torch.sum(inputs, dim=0) + torch.sum(targets, dim=0) - intersection

        self.intersection += intersection
        self.union += union

    def compute(self):
        iou = (self.intersection + self._smooth) / (self.union + self._smooth)
        return torch.mean(iou)
