from pycromanager import Dataset
import os
import json
import numpy as np
from tifffile import imread, memmap
import pandas as pd


def pycromanager_data_loader(receipt):

    # These are the mandatory directories of this 'data structure'
    for k in receipt['dirs'].keys():
        os.makedirs(receipt['dirs'][k], exist_ok=True)

    # Check if local path and does a bunch with it if it does (if it doesnt exist theres not much to do)
    local_path = receipt['arguments'].get('local_location', None)
    analysis_name = receipt['arguments']['analysis_name']

    return load_pycromanager(local_path, analysis_name)


def load_pycromanager(location, analysis_name):
    data = {} # were gonna use this as basically a struct
    if location and os.path.exists(location):
        ds = Dataset(location)

        images = ds.as_array(['position', 'time', 'channel', 'z'])
        pp, tt, cc, zz, yy, xx = images.shape
        data['images'] = images
        data['pp'], data['tt'], data['cc'], data['zz'], data['yy'], data['xx'] = pp, tt, cc, zz, yy, xx

        metadata = lambda p, t, z=0, c=0: ds.read_metadata(position=p, time=t, channel=c, z=z)
        data['metadata'] = metadata

        for file in os.listdir(os.path.join(location, analysis_name, 'results')):
            key, returned_data = read_data(os.path.join(location, analysis_name, 'results', file))
            data[key] = returned_data

        for file in os.listdir(os.path.join(location, 'masks')):
            key, returned_data = read_data(os.path.join(os.path.join(location, 'masks'), file))
            data[key] = returned_data

    return data


def read_data(file_path):
    # extract file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    key = os.path.splitext(os.path.basename(file_path))[0]
    data = None

    # read data
    if ext in ['.tif', '.tiff']:
        data = memmap(file_path, mode='r+')
    elif ext == '.json':
        data = json.load(open(file_path, 'r'))
    elif ext == '.csv':
        data = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return key, data


def close_data(data):

    keys_to_delete = [key for key, value in data.items() if isinstance(value, np.memmap)]
    for key in keys_to_delete:
        data[key]._mmap.close()
        del data[key]
    data.clear()









































