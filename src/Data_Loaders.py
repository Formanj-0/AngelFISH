from pycromanager import Dataset
import os
import json
import numpy as np
from tifffile import imread, memmap
import pandas as pd


def pycromanager_data_loader(receipt):
    """
    loads pycromanager datasets (ndstorage, or ndtiff datasets) with a few extensions
    extensions are
    - masks dir 
    - analysis dirs


    """
    # Check if local path and does a bunch with it if it does (if it doesnt exist theres not much to do)
    local_path = receipt['arguments']['local_location']# the location must be local
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

        metadata = lambda p, t, c=0, z=0: ds.read_metadata(position=p, time=t, channel=c, z=z)
        data['metadata'] = metadata

        for file in os.listdir(os.path.join(location, analysis_name, 'results')):
            key, returned_data = read_data(os.path.join(location, analysis_name, 'results', file))
            data[key] = returned_data

        for file in os.listdir(os.path.join(location, 'masks')):
            key, returned_data = read_data(os.path.join(os.path.join(location, 'masks'), file))
            data[key] = returned_data

    return data


def concate_data(x, concate_function_str:str=None):
    first_type = type(x[0])
    assert np.all([first_type == type(d) for d in x]), 'data is not the same type'

    if concate_function_str:
        return eval(concate_function_str)
    elif issubclass(first_type, pd.DataFrame):
        return pd.concat(x, axis=0, ignore_index=True)
    elif issubclass(first_type, (np.ndarray, np.memmap)):
        return np.concatenate(x, axis=0)
    else:
        return x


def format_list_of_pyromanager_data(data, local_path, analysis_name, recursive_analysis_name, receipt):
    """
    This may need to be changed in the future, or use a different recursive_pycromanager_data_loader
    use ['arguments']['data_loader_settings'] or something to make this work better cause were hitting
    a large area of possiblities that will work with different functions
    """
    first_keys = list(data[0].keys())

    final_data = {}
    final_data['images'] = np.concatenate([d['images'] for d in data], axis=0)
    pp, tt, cc, zz, yy, xx = final_data['images'].shape
    final_data['pp'], final_data['tt'], final_data['cc'], final_data['zz'], final_data['yy'], final_data['xx'] = pp, tt, cc, zz, yy, xx
    first_keys.remove('images')

    # this is a map from new p values to the os.listdir(local_path) index and p in that original image
    lengths_of_images = [d['images'].shape for d in data]
    left_inclusive_right_exlusive_indexs = np.concatenate(([0], np.cumsum(lengths_of_images), [np.inf]))
    # map_p2np: Given a global position index p, find the subdirectory index n and local position index p_local
    def map_p2np(p):
        # Find the index n such that left_inclusive_right_exlusive_indexs[n] <= p < left_inclusive_right_exlusive_indexs[n+1]
        n = np.searchsorted(left_inclusive_right_exlusive_indexs, p, side='right') - 1
        p_local = p - left_inclusive_right_exlusive_indexs[n]
        return n, p_local
    
    final_data['metadata'] = lambda p, t, z=0, c=0: (
        lambda n_p: data[n_p[0]]['metadata'](p=n_p[1], t=t, c=c, z=z)
    )(map_p2np(p))
    first_keys.remove('metadata')

    first_dir = os.listdir(local_path)[0]
    for mask_name in os.listdir(os.path.join(first_dir, 'masks')):
        final_data[mask_name] = np.concatenate([d[mask_name] for d in data], axis=0)
        first_keys.remove(mask_name)

    completed_keys = ['pp', 'tt', 'cc', 'zz', 'yy', 'xx']
    for k in first_keys:
        final_data[k] = concate_data([d[k] for d in data])
        completed_keys.append(k)

    for k in completed_keys:
        first_keys.remove(k)
    assert len(first_keys) == 0, 'not all keys were concatenated' # this is kinda useless but whatever

    return final_data


def recursive_pycromanager_data_loader(receipt):
    """
    loads a dir of pycromanager datasets (ndstorage, or ndtiff datasets) with a few extensions
    extensions are
    - masks dirs
    - analysis dirs

    # the return_data and downlaod_data rely on singular values nas location and local locations. 
    # It is possible to make a data_loader that doesnt need a single value for these values,
    # but other steps may need to be fixed too. 
    """
    # Check if local path and does a bunch with it if it does (if it doesnt exist theres not much to do)
    local_path = receipt['arguments']['local_location'] # the location must be local
    analysis_name = receipt['arguments']['analysis_name']

    # get all the data we want
    recursive_analysis_name = receipt['arguments'].get('recursive_analysis_name', analysis_name)
    data = [load_pycromanager(path, recursive_analysis_name) for path in os.listdir(local_path)]
    keys = data[0].keys()
    assert np.all([keys == d.keys() for d in data]), 'all sub pycromanager dataset dont have the same keys'

    # format the data from the subdirectories so that it is usable
    subdirectories_data = format_list_of_pyromanager_data(data, local_path, analysis_name, recursive_analysis_name, receipt)

    # get data from the parent (combined dir)
    pardirectory_data = {}
    for file in os.listdir(os.path.join(local_path, analysis_name, 'results')):
        key, returned_data = read_data(os.path.join(local_path, analysis_name, 'results', file))
        pardirectory_data[key] = returned_data

    for file in os.listdir(os.path.join(local_path, 'masks')):
        key, returned_data = read_data(os.path.join(os.path.join(local_path, 'masks'), file))
        pardirectory_data[key] = returned_data

    shared_keys = set(data.keys()) & set(pardirectory_data.keys())
    assert len(shared_keys) == 0, f"Shared keys found between parent directory data and sub directory data: {shared_keys}"

    return {**subdirectories_data, **pardirectory_data}


def read_data(file_path):
    """
    general file loader
    """
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









































