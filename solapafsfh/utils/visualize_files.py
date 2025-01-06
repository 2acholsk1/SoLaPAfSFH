import matplotlib.pyplot as plt
import albumentations as A
import albumentations.pytorch.transforms
from solapafsfh.datamodules.segment_datamodule import LawnAndPavingDataModule

transform = A.Compose([
    A.Resize(width=512, height=512),
    A.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.], max_pixel_value=255.),
    albumentations.pytorch.transforms.ToTensorV2()
])

datamodule = LawnAndPavingDataModule('data/')
datamodule.setup('fit')
index = 0

def update_image(index):
    image, mask = datamodule.train_dataset[index]

    axes[0].cla()
    axes[1].cla()

    axes[0].imshow(image.permute(1, 2, 0))
    axes[0].set_title("Picture")
    axes[0].axis('off')

    axes[1].imshow(mask.numpy(), cmap='gray')
    axes[1].set_title("Mask")
    axes[1].axis('off')

    plt.draw()

def on_key(event):
    global index
    if event.key == 'right':
        if index < len(datamodule.train_dataset) - 1:
            index += 1
    elif event.key == 'left':
        if index > 0:
            index -= 1
    elif event.key == 'q':
        plt.close(fig)
        return
    
    update_image(index)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

update_image(index)

fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()
