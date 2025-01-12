from solapafsfh.models.segmentation_model import SegmentationModel
import torch.onnx

model = SegmentationModel(
        'UNet',
        'resnet18',
        3,
        ['background', 'lawn', 'paving'],
        'Dice',
        1e-3
    )
model.load_state_dict(torch.load('checkpoints/best-checkpoint-v16.ckpt', map_location='cpu')['state_dict'])
model.eval()

x = torch.rand(1, 3, 512, 512)
_ = model(x)

torch.onnx.export(model,
                x,
                'onxx_model',
                export_params=True,
                opset_version=15,
                input_names=['input'],
                output_names=['output'],
                do_constant_folding=False)
checkpoint = torch.load('checkpoints/best-checkpoint-v16.ckpt', map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['state_dict'])
