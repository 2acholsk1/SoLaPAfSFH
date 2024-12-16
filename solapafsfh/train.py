import lightning.pytorch as pl
from solapafsfh.datamodules.segment_datamodule import LawnAndPavingDataModule
from solapafsfh.models.segmentation_model import SegmentationModel

def train():
    data_module = LawnAndPavingDataModule()
    
    model = SegmentationModel('UNet', 'resnet34', 3, ['background', 'lawn', 'paving'], 'Dice', 0.001)
    
    trainer = pl.Trainer(accelerator='gpu', max_epochs=10)
    trainer.fit(model, datamodule=data_module)

train()