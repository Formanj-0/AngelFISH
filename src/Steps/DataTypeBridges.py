import pathlib
import pycromanager as pycro
import shutil
import tifffile
import numpy as np
import os
import sys
import inspect
from datetime import datetime
from abc import ABC, abstractmethod
import dask_image.imread as dask_imread
from dask import array as da
from ndstorage import NDTiffDataset, NDTiffPyramidDataset
from ndtiff import Dataset

import h5py
import json


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# from src import IndependentStepClass, DataContainer
# from src.Parameters import Parameters
from src.NASConnection import NASConnection



#%% Useful Functions
def get_first_executing_folder():
    # Get the current stack
    stack = inspect.stack()
    
    # The first frame in the stack is the current function,
    # The second frame is the caller. However, we want the first script.
    # To find that, we look for the first frame that isn't from an internal call.
    for frame in stack:
        if frame.filename != __file__:
            # Extract the directory of the first non-internal call
            return os.path.dirname(os.path.abspath(frame.filename))

    return None

class DownloadData: # (IndependentStepClass):
    def main(self, initial_data_location, connection_config_location, 
            local_dataset_location: list[str] = None, 
            **kwargs):
        # save to a single location
        database_loc = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        database_loc = os.path.join(database_loc, 'dataBases')

        # if data is local
        if local_dataset_location is not None:
            if isinstance(local_dataset_location, str):
                local_dataset_location = [local_dataset_location]

        # if data is on originates from the nas
        else:
            local_dataset_location = []
            if type(initial_data_location) == str:
                initial_data_location = [initial_data_location]
            initial_data_location = [location.replace('\\', '/') for location in initial_data_location]

            for i, location in enumerate(initial_data_location):
                is_file = any(location.endswith(ext) for ext in ['.h5', '.tif', '.zip', '.log'])
                is_folder = not is_file
                name = os.path.basename(location)

                if is_folder: # if it is a folder (aka pyromanager datasets)
                    destination = os.path.join(database_loc, name)

                local_dataset_location.append(destination)

                # stops if h5 is already there
                if os.path.exists(os.path.join(database_loc, name)):
                    continue

                # download the data
                if is_folder:
                    self.download_folder_from_NAS(location, destination, connection_config_location)
                else:
                    self.download_file_from_NAS(location, destination, connection_config_location)
        return {'local_dataset_location': local_dataset_location}

    def download_folder_from_NAS(self, remote_path, local_folder_path, connection_config_location):
        # Downloads a folder from the NAS, confirms that the it has not already been downloaded
        if not os.path.exists(local_folder_path):
            os.makedirs(local_folder_path, exist_ok=True)
            nas = NASConnection(pathlib.Path(connection_config_location))
            nas.copy_folder(remote_folder_path=pathlib.Path(remote_path), 
                        local_folder_path=local_folder_path)
            
    def download_file_from_NAS(self, remote_path, local_folder_path, connection_config_location):
        # Downloads a folder from the NAS, confirms that the it has not already been downloaded
        if not os.path.exists(os.path.join(local_folder_path, os.path.basename(remote_path))):
            os.makedirs(local_folder_path, exist_ok=True)
            nas = NASConnection(pathlib.Path(connection_config_location))
            nas.download_file(remote_file_path=pathlib.Path(remote_path), 
                        local_folder_path=local_folder_path)


#%% Abstract Class
class DataTypeBridge_Dataset: # (IndependentStepClass):
    def main(self, load_in_mask, initial_data_location,
            independent_params, local_dataset_location: list[str] = None,
            **kwargs):
        for i, location in enumerate(local_dataset_location):
            local_dataset_location[i] = self.convert_data(location)

        # Load in H5 and build independent param
        images, masks, ip, position_indexs, mask_locations = self.load_in_dataset(local_dataset_location,
                                                                                    initial_data_location,
                                                                                        load_in_mask, 
                                                                                        independent_params)
        return {'images': images,
                'independent_params': ip, 'position_indexs': position_indexs,\
                'masks': masks, 'mask_locations': mask_locations, 'image_locations': local_dataset_location}

    @abstractmethod
    def convert_data(self, location):
        # For any standardized data type this will convert it to a h5 file and save that file in the folder
        # that the data originally came from
        ...

    def load_in_dataset(self, locations, nas_location, load_in_mask, independent_params):
        # given an h5 file this will load in the dat in a uniform manner
        position_indexs = []

        dss = [Dataset(location) for location in locations]
        # TODO: check if another file is already opening the h5 file

        images = [ds.as_array(['position', 'time', 'channel','z']) for ds in dss]
        metadata = [ds.read_metadata(position=0, time=0, z=0, channel=0) for ds in dss]
        exp_metadata = [m.get('experimental_metadata', None) for m in metadata]
        position_indexs = [img.shape[0] for img in images]
        images = da.concatenate(images, axis=0)
        images = images.rechunk((1, 1, -1, -1, -1, -1))
        masks = {}
        mask_locations = {}

        # Find tifs in the directory and add them to the dask dictionary
        for location in locations:
            mask_path = [f for f in os.listdir(location) if 'mask' in f]
            if len(mask_path) == 1:
                tifs = [f for f in os.listdir(os.path.join(location, mask_path[0]))]
                for tif in tifs:
                    tif_path = os.path.join(location, mask_path[0], tif)
                    if os.path.splitext(tif)[0] not in mask_locations.keys():
                        mask_locations[os.path.splitext(tif)[0]] = []
                    mask_locations[os.path.splitext(tif)[0]].append(tif_path)
                    tif_data = dask_imread.imread(tif_path)
                    if os.path.splitext(tif)[0] not in masks.keys():
                        masks[os.path.splitext(tif)[0]] = None
                    masks[os.path.splitext(tif)[0]] = tif_data

        for key in masks.keys():
            masks[key] = da.concatenate(masks[key], axis=0)

        num_chuncks = images.shape[0] * images.shape[1]

        position_indexs = np.cumsum(position_indexs)

        # we have 3 inputs for independent params list[dict], dict, dict w/ proper keys
        if isinstance(independent_params, dict) and set(independent_params.keys()) == set(np.arange(position_indexs[-1]).tolist()):
            # handles dict w/ proper keys
            ip = independent_params
        else:
            if not isinstance(independent_params, list):
                # handles dict w/o proper keys
                independent_params = [independent_params]

            # converts all to proper keys
            ip = {}
            for i, p in enumerate(position_indexs):
                if independent_params is not None and len(independent_params) > 1:
                    if exp_metadata[i] is not None:
                        for key, val in exp_metadata[i].items():
                            independent_params[i][key] = val
                    if nas_location is not None:
                        independent_params[i]['NAS_location'] = nas_location[i]
                    if i == 0:
                        temp = {p_idx: independent_params[i] for p_idx in range(p)}
                    else:
                        temp = {p_idx: independent_params[i] for p_idx in range(position_indexs[i-1], p)}
                    ip = {**ip, **temp}
                elif independent_params is not None and len(independent_params) == 1:
                    if nas_location is not None:
                        independent_params[0]['NAS_location'] = nas_location[i]
                    if i == 0:
                        temp = {p_idx: independent_params[0] for p_idx in range(p)}
                    else:
                        temp = {p_idx: independent_params[0] for p_idx in range(position_indexs[i-1], p)}
                    ip = {**ip, **temp}
                else:
                    print('Something is broken')

        return images, masks, ip, position_indexs, mask_locations
    
    def delete_folder(self, folder):
        shutil.rmtree(folder)


class DataTypeBridge_H5: # (IndependentStepClass):
    def main(self, load_in_mask, initial_data_location, mask_structure,
            independent_params, local_dataset_location: list[str] = None,
            **kwargs):
        for i, location in enumerate(local_dataset_location):
            local_dataset_location[i] = self.convert_data(location)

        # Load in H5 and build independent param
        images, masks, ip, position_indexs, mask_locations = self.load_in_dataset(local_dataset_location,
                                                                  initial_data_location,
                                                                    load_in_mask, 
                                                                    independent_params)
        return {'images': images,
                'independent_params': ip, 'position_indexs': position_indexs,\
                'masks': masks, 'mask_locations': mask_locations, 'image_locations': local_dataset_location}


    @abstractmethod
    def convert_data(self, location):
        # For any standardized data type this will convert it to a h5 file and save that file in the folder
        # that the data originally came from
        ...

    def load_in_dataset(self, locations, nas_location, load_in_mask, independent_params):
        # given an h5 file this will load in the dat in a uniform manner
        position_indexs = []
        H5_names = [n + '.h5' for n in nas_location]

        h5_files = [h5py.File(H5_location, 'r') for H5_location in locations]
        # TODO: check if another file is already opening the h5 file

        images = [da.from_array(h5['raw_images']) for h5 in h5_files]
        position_indexs = [img.shape[0] for img in images]
        images = da.concatenate(images, axis=0)
        images = images.rechunk((1, 1, -1, -1, -1, -1))
        masks = {}
        mask_locations = {}
        if load_in_mask:
            for key in h5_files[0].keys():
                if 'analysis' not in key and 'image' not in key and 'meta' not in key:
                    masks[key] = [da.from_array(h5[key]) for h5 in h5_files]
                    masks[key] = da.concatenate(masks[key], axis=0)
                    masks[key] = masks[key].rechunk((1, 1, -1, -1, -1, -1))
                    mask_locations[key] = locations


        num_chuncks = images.shape[0] * images.shape[1]

        position_indexs = np.cumsum(position_indexs)

        # we have 3 inputs for independent params list[dict], dict, dict w/ proper keys
        if isinstance(independent_params, dict) and set(independent_params.keys()) == set(np.arange(position_indexs[-1]).tolist()):
            # handles dict w/ proper keys
            ip = independent_params
        else:
            if not isinstance(independent_params, list):
                # handles dict w/o proper keys
                independent_params = [independent_params]

            # converts all to proper keys
            ip = {}
            for i, p in enumerate(position_indexs):
                if independent_params is not None and len(independent_params) > 1:
                    if nas_location is not None:
                        independent_params[i]['NAS_location'] = H5_names[i]
                    if i == 0:
                        temp = {p_idx: independent_params[i] for p_idx in range(p)}
                    else:
                        temp = {p_idx: independent_params[i] for p_idx in range(position_indexs[i-1], p)}
                    ip = {**ip, **temp}
                elif independent_params is not None and len(independent_params) == 1:
                    if nas_location is not None:
                        independent_params[0]['NAS_location'] = H5_names[i]
                    if i == 0:
                        temp = {p_idx: independent_params[0] for p_idx in range(p)}
                    else:
                        temp = {p_idx: independent_params[0] for p_idx in range(position_indexs[i-1], p)}
                    ip = {**ip, **temp}
                else:
                    print('Something is broken')

        for h5 in h5_files:
            h5.close()
        return images, masks, ip, position_indexs, mask_locations
    
    def delete_folder(self, folder):
        shutil.rmtree(folder)

#%% Data Bridges
class Pycro(DataTypeBridge_Dataset):
    def __init__(self):
        super().__init__()

    def convert_data(self, location):
        return location


class Pycromanager2H5(DataTypeBridge_H5):
    def __init__(self):
        super().__init__()

    def convert_data(self, location):
        if not location.endswith('.h5'):
            h5_name = location + '.h5'
        # check if h5 file already exists
        if os.path.exists(h5_name):
            return h5_name
        
        ds = Dataset(location)
        
        imgs = ds.as_array('position', 'time', 'channel', 'z', 'y', 'x')

        da.to_hdf5(location+'h5', '/raw_images', imgs)
        return location+'h5'


class FFF2H5(DataTypeBridge_H5):
    def __init__(self):
        super().__init__()

    def convert_data(self, location): 
        if not location.endswith('.h5'):
            h5_name = location + '.h5'
        else:
            h5_name = location
        # check if h5 file already exists
        if os.path.exists(h5_name):
            print(f'h5_name already exist {h5_name}')
            return h5_name
        
        files = os.listdir(location)
        tifs = [f for f in files if f.endswith('.tif')]
        logs = [f for f in files if f.endswith('.log')]
        mask_dirs = [f for f in files if f.startswith('masks')]

        already_made_masks = False

        if len(mask_dirs) > 0:
            zipped_mask_dir = [f for f in mask_dirs if f.endswith('.zip')]

            mask_tifs = [f for f in mask_dirs if f.endswith('.tif')]

            if len(zipped_mask_dir) == 1 and len(mask_tifs) == 0:
                shutil.unpack_archive(os.path.join(location, zipped_mask_dir[0]), location)
                already_made_masks = True
            
            mask_dirs = [f for f in files if f.startswith('masks')]
            mask_tifs = [f for f in mask_dirs if f.endswith('.tif')]

            mask_cells = [f for f in mask_tifs if 's_cyto_R' in f]
            mask_nuclei = [f for f in mask_tifs if 'masks_nuclei_R' in f]
            already_made_masks = True
    
        # create list of images
        list_images_names = [f for f in tifs if not f.startswith('masks')]
        list_channels = np.sort(list(set([f.split('_')[-1].split('.')[0] for f in list_images_names])))
        list_roi = np.sort(list(set([f.split('_')[0] for f in list_images_names])))
        timepoints = np.sort(list(set([f.split('_')[3] for f in list_images_names])))

        number_of_timepoints = len(set(timepoints))
        number_color_channels = len(set(list_channels))
        number_of_fov = len(set(list_roi))

        # os.makedirs(local_folder, exist_ok=True)
        imgs = None
        masks = {}
        count = 0
        img_metadata = {}
        for t in range(number_of_timepoints):
            tp = timepoints[t]
            for r in range(number_of_fov):
                fov = list_roi[r]

                for c in range(number_color_channels):
                    channel = list_channels[c]
                    search_params = [fov, channel, tp]
                    img_name = [f for f in list_images_names if all(v in f for v in search_params)][0]
                    img = tifffile.imread(os.path.join(location, img_name))
                    img = da.from_array(img)
                    # make all the image data floats
                    img = img.astype(np.float32)

                    search_params = [fov]
                    log_name = [f for f in logs if all(v in f for v in search_params)][0]
                    with open(os.path.join(location, log_name), 'r') as f:
                        log = f.readlines()

                    if fov not in img_metadata:
                        img_metadata[fov] = {}
                    img_metadata[fov][tp] = log

                    if imgs is None:
                        imgs = da.zeros((number_of_fov, number_of_timepoints, number_color_channels, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)

                    imgs[r, t, c, :, :, :] = img

                    if already_made_masks:
                        search_params = [fov, tp]
                        cell_mask_name = [f for f in mask_cells if all(v in f for v in search_params)][0] if len(mask_cells) > 0 else None
                        nuc_mask_name = [f for f in mask_nuclei if all(v in f for v in search_params)][0] if len(mask_nuclei) > 0 else None
                        if cell_mask_name is not None:
                            # print('cell ', cell_mask_name)
                            mask = tifffile.imread(os.path.join(location, cell_mask_name))
                            if 'cell' not in masks.keys():
                                masks['cell'] = da.zeros((number_of_fov, 1, 1, img.shape[1], img.shape[2]), dtype=np.float32)
                            masks['cell'][r, 0, 0, :, :] = da.from_array(mask)
                        if nuc_mask_name is not None:
                            if 'cell' not in masks.keys():
                                masks['nuc'] = da.zeros((number_of_fov, 1, 1, img.shape[1], img.shape[2]), dtype=np.float32)
                            mask = tifffile.imread(os.path.join(location, nuc_mask_name))
                            masks['nuc'][r, 0, 0, :, :] = da.from_array(mask)
                    count += 1

        imgs = imgs.rechunk((1, 1, -1, -1, -1, -1))
        for key in masks.keys():
            masks[key] = masks[key].rechunk((1, 1, -1, -1, -1, -1))
        
        da.to_hdf5(h5_name, {'/raw_images': imgs, **masks}, compression='gzip')

        metadata_str = json.dumps(img_metadata)
        with h5py.File(h5_name, 'a') as h5f:
            if '/metadata' in h5f:
                del h5f['/metadata']
            h5f.create_dataset('/metadata', data=metadata_str)

        del imgs
        del masks
        del metadata_str

        return h5_name


class SingleTIFF2H5(DataTypeBridge_H5):
    def __init__(self):
        super().__init__()

    def convert_data(self, location):
        if not location.endswith('.h5'):
            h5_name = location + '.h5'

        if os.path.exists(os.path.join(location)):
            return 'already exists'
        
        img = tifffile.imread(os.path.join(location))

        img = img[np.newaxis, :, np.newaxis, np.newaxis, :, :]
        img = da.from_array(img)
        img = img.rechunk((1, 1, -1, -1, -1, -1))

        masks = da.zeros(img.shape)

        masks = masks.rechunk((1, 1, -1, -1, -1, -1))
        da.to_hdf5(h5_name, {'/raw_images': img, '/masks': masks}, compression='gzip')

        return h5_name

class PycromanagerDataset(DataTypeBridge_Dataset):
    def convert_data(self, location):
        pass




