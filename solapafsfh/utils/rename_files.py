import os
import click

@click.command()
@click.argument('folder_path', type=click.Path(exists=True))
@click.option('--prefix', default='img', help='Prefix to names(default "img")')
def rename_files(folder_path, prefix):
    image_files = []
    mask_files = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            image_files.append(filename)
        elif filename.endswith('_mask.png'):
            base_name = filename.replace('_mask.png', '')
            mask_files[base_name] = filename

    image_files.sort()

    for index, original_image_name in enumerate(image_files, start=1):
        new_image_name = f'{prefix}_{index}.jpg'
        
        original_image_path = os.path.join(folder_path, original_image_name)
        new_image_path = os.path.join(folder_path, new_image_name)
        
        os.rename(original_image_path, new_image_path)
        print(f'Renamed {original_image_name} to {new_image_name}')
        
        base_name = original_image_name.replace('.jpg', '')
        if base_name in mask_files:
            original_mask_name = mask_files[base_name]
            new_mask_name = f'{prefix}_{index}_mask.png'
            
            original_mask_path = os.path.join(folder_path, original_mask_name)
            new_mask_path = os.path.join(folder_path, new_mask_name)
            
            os.rename(original_mask_path, new_mask_path)
            print(f'Renamed {original_mask_name} to {new_mask_name}')
        else:
            print(f'No file of {original_image_name}')

if __name__ == '__main__':
    rename_files()
