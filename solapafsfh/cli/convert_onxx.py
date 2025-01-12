import yaml
import torch.onnx
import click

from pathlib import Path

from solapafsfh.models.segmentation_model import SegmentationModel

@click.command()
@click.argument('checkpoint_path', type=click.Path(exists=True, path_type=Path))
@click.option('--export_model_name', type=click.STRING, default='onxx_model')
def export_model_to_onxx(checkpoint_path: Path, export_model_name: str):
    with open("configs/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    model = SegmentationModel(
            config['model']['name'],
            config['model']['encoder_name'],
            config['model']['in_channels'],
            config['model']['classes'],
            config['model']['loss_func'],
            config['model']['lr'],
        )
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['state_dict'])
    model.eval()

    x = torch.rand(1, 3, 512, 512)
    _ = model(x)
    export_model_name = export_model_name + '.onnx'
    torch.onnx.export(model,
                    x,
                    'onxx_models/'+export_model_name,
                    export_params=True,
                    opset_version=15,
                    input_names=['input'],
                    output_names=['output'],
                    do_constant_folding=False)

if __name__ == '__main__':
    export_model_to_onxx()
