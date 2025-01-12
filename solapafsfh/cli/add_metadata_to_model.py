import onnx
import click
import json
import timm.data
from pathlib import Path

@click.command
@click.argument('onnx_model_path', type=click.Path(exists=True, path_type=Path))
@click.option('--export_model_name', type=click.STRING, default='onxx_model_meta')
def metadata_to_model(onnx_model_path: Path, export_model_name: str):
    model = onnx.load(onnx_model_path)

    class_names = {
        0: 'background',
        1: 'lawn',
        2: 'paving',
    }

    m1 = model.metadata_props.add()
    m1.key = 'model_type'
    m1.value = json.dumps('Segmentor')

    m2 = model.metadata_props.add()
    m2.key = 'class_names'
    m2.value = json.dumps(class_names)

    m3 = model.metadata_props.add()
    m3.key = 'standardization_mean'
    m3.value = json.dumps(timm.data.IMAGENET_DEFAULT_MEAN)

    m4 = model.metadata_props.add()
    m4.key = 'standardization_std'
    m4.value = json.dumps(timm.data.IMAGENET_DEFAULT_STD)

    onnx.save(model, 'onxx_models/' + export_model_name + '.onnx')

if __name__ == '__main__':
    metadata_to_model()
