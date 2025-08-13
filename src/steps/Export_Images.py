
from src import load_data
import os
import numpy as np
from tifffile import imwrite


def main(
        receipt, 
        step_name, 
        new_params:dict = None, 
        p_range = None, 
        t_range = None,
        run_in_parallel:bool = False
        ):
    data = load_data(receipt)

    # required updates to the receipt
    if step_name not in receipt['step_order']:
        receipt['step_order'].append(step_name)
    if step_name not in receipt['steps'].keys():
        receipt['steps'][step_name] = {}
    receipt['steps'][step_name]['task_name'] = 'export_images'
    if new_params:
        for k, v in new_params.items():
            receipt['steps'][step_name][k] = v


    # extract parameters
    channel_order = receipt['steps'][step_name].get('channel_order', 'ptczyx')
    export_format = receipt['steps'][step_name].get('export_format', 'tif')
    ptczyx_image = data['images']
    save_dir = receipt['dirs']['results_dir']

    if export_format == 'cellpose':
        channel_order = 'ptzcyx'

    # reorder images
    axis_map = {a: i for i, a in enumerate('ptczyx')}
    permute_order = [axis_map[a] for a in channel_order]
    reordered_image = np.transpose(ptczyx_image, permute_order)

    if export_format.lower() in ['tif', 'tiff']:
        filename = f"images.{export_format}"
        save_path = os.path.join(save_dir, filename)
        imwrite(save_path, reordered_image)

    elif export_format.lower() == 'cellpose':
        for p in range(reordered_image.shape[0]) if p_range is None else p_range:
            for t in range(reordered_image.shape[1]) if t_range is None else t_range:
                filename = f"frame{(p+1)*(t+1)}-image.tif"
                save_path = os.path.join(save_dir, filename)
                imwrite(save_path, reordered_image[p,t])

    else:
        raise ValueError(f"Unsupported export format: {export_format}")

    return receipt

















































