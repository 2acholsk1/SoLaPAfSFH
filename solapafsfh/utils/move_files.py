import os
import shutil
import click

@click.command()
@click.argument('source_folder', type=click.Path(exists=True))
@click.argument('images_folder', type=click.Path())
@click.argument('masks_folder', type=click.Path())
def split_images_and_masks(source_folder, images_folder, masks_folder):
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)

        if os.path.isfile(file_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                if '_mask' not in filename:
                    shutil.move(file_path, os.path.join(images_folder, filename))
                    print(f"Moving image: {filename}")
                else:
                    shutil.move(file_path, os.path.join(masks_folder, filename))
                    print(f"Moving mask: {filename}")

if __name__ == '__main__':
    split_images_and_masks()
