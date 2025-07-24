from pycromanager import Dataset
import os
import json
import numpy as np
from tifffile import imread, memmap
import pandas as pd


def pycromanager_data_loader(receipt):
    data = {}

    local_path = receipt['meta_arguments'].get('local_location', None)
    if local_path and os.path.exists(local_path):
        ds = Dataset(local_path)

        images = ds.as_array(['position', 'time', 'channel', 'z'])
        pp, tt, cc, zz, yy, xx = images.shape
        data['images'] = images
        data['pp'], data['tt'], data['cc'], data['zz'], data['yy'], data['xx'] = pp, tt, cc, zz, yy, xx

        metadata = lambda p, t, z=0, c=0: ds.read_metadata(position=p, time=t, channel=c, z=z)
        data['metadata'] = metadata

        analysis_name = receipt['meta_arguments']['analysis_name']
        analysis_dir = os.path.join(local_path, analysis_name)
        os.makedirs(analysis_dir, exist_ok=True)
        data['analysis_dir'] = analysis_dir

        results_dir = os.path.join(analysis_dir, 'results')
        os.makedirs(analysis_dir, exist_ok=True)
        data['results_dir'] = results_dir

        status_dir = os.path.join(analysis_dir, 'status')
        os.makedirs(status_dir, exist_ok=True)
        data['status_dir'] = status_dir

        for file in os.listdir(results_dir):
            key, data = read_data(file)
            data[key] = data

    return data

def read_data(file_path):
    # extract file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    key = os.path.basename(file_path)
    data = None

    # read data
    if ext in ['.tif', '.tiff']:
        data = memmap(file_path)
    elif ext == '.json':
        data = json.load(open(file_path, 'r'))
    elif ext == '.csv':
        data = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return key, data









































