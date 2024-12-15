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
        assert inputs.shape == targets.shape
        
        inputs = torch.argmax(inputs, dim=1)
        inputs = F.one_hot(inputs, num_classes=self._num_classes)

        targets = torch.argmax(targets, dim=1)
        targets = F.one_hot(targets, num_classes=self._num_classes)
        
        targets = torch.flatten(targets, 1)
        inputs = torch.flatten(inputs, 1)
        
        intersection = torch.sum(targets * inputs, dim=1)
    
        union = torch.sum(targets, dim=1) + torch.sum(inputs, dim=1) - intersection
        
        self.intersection += intersection
        self.union += union

    def compute(self):
        iou = (self.intersection + self._smooth) / (self.union + self._smooth)
        return torch.mean(iou)
