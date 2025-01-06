import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl
import lightning.pytorch.loggers
from solapafsfh.datamodules.segment_datamodule import LawnAndPavingDataModule
from solapafsfh.models.segmentation_model import SegmentationModel

def train(config: DictConfig):
    pl.seed_everything(42, workers=True)
    data_module = LawnAndPavingDataModule(
        data_path=config.data_path
    )
    
    model = SegmentationModel(
        config.model.name,
        config.model.encoder_name,
        config.model.in_channels,
        config.model.classes,
        config.model.loss_func,
        config.model.lr
    )

    model_summary_callback = pl.callbacks.ModelSummary(max_depth=-1)
    logger = lightning.pytorch.loggers.NeptuneLogger(
        project=config.logger.project,
        log_model_checkpoints=False,
        )
    
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[model_summary_callback],
        accelerator='gpu',
        precision=config.trainer.precision,
        benchmark=True,
        max_epochs=config.trainer.max_epochs)

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, data_module, ckpt_path='best')

    logger.finalize('success')

@hydra.main(config_path='../../configs/', config_name='config.yaml', version_base=None)
def main(config: DictConfig):
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934

    # Train model
    return train(config)


if __name__ == '__main__':
    main()