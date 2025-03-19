import pathlib
import numpy as np
import shutil
from fpdf import FPDF
import os
import pickle
import trackpy as tp
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import dask.array as da
from datetime import datetime
from abc import abstractmethod
import gc
from typing import List, Dict, Any, Union

from src.GeneralStep import FinalizingStepClass
from src.Parameters import Parameters, DataContainer, ScopeClass, Settings, Experiment
from src.NASConnection import NASConnection
import tiffile
import json

def close_h5_files():
    for obj in gc.get_objects():
        if isinstance(obj, h5py.File) and obj.id.valid:
            try:
                if 'temp' not in obj.filename:
                    print(f"Closing {obj.filename}")
                    obj.close()
            except OSError as e:
                print(f"Could not close {obj.filename}: {e}")

class Saving(FinalizingStepClass):
    @abstractmethod
    def main(self, **kwargs):
        pass

def handle_df(df):
    for col in df.columns:
        if df[col].dtype == 'O':  # Object type
            if df[col].map(type).nunique() == 1 and isinstance(df[col].iloc[0], str):
                df[col] = df[col].astype(str)  # Convert to string
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, if possible
    return df

def handle_dict(d):
    newd = {}
    for key in d.keys():
        if isinstance(d[key], str):
            newd[key] = d[key]
        elif isinstance(d[key], int):
            newd[key] = d[key]
        elif isinstance(d[key], float):
            newd[key] = d[key]
        elif isinstance(d[key], bool):
            newd[key] = d[key]
        elif isinstance(d[key], list):
            newd[key] = d[key]
        else:
            print(f'IDK what to do with {key}: {type(d[key])}')
    return newd


class Save_Outputs(Saving):
    def run(self, p: int = None, t:int = None, data_container = None, parameters = None): # TODO confirm - local_dataset_location is where the h5_file is in the correct dataBase
        data_container = DataContainer() if data_container is None else data_container
        parameters = Parameters() if parameters is None else parameters
        kwargs = self.load_in_parameters(p, t, parameters)
        results = self.main(data_container, **kwargs) 
        data_container.save_results(results, p, t, parameters)
        data_container.load_temp_data()
        return results

    def main(self, data_container, **kwargs):
        params = kwargs

        Analysis_name = params['name']
        local_dataset_location = params['local_dataset_location']
        independent_params = params['independent_params']
        position_indexs = params['position_indexs']
        if 'h5_files' in params.keys():
            for h5 in params['h5_files']:
                h5.flush()
                h5.close()

        # get todays date
        today = datetime.today()
        date = today.strftime("%Y-%m-%d")

        data = data_container.load_temp_data()

        self.save(data, local_dataset_location, f'Analysis_{Analysis_name}_{date}', position_indexs, independent_params)

    def save(self, attributes, locations, group_name: str, position_indexs: list[int], independent_params: dict[str, Any] ):
        # get all the attributes of the class

        def handle_df(df):
            for col in df.columns:
                if df[col].dtype == 'O':  # Object type
                    if df[col].map(type).nunique() == 1 and isinstance(df[col].iloc[0], str):
                        df[col] = df[col].astype(str)  # Convert to string
                    else:
                        df[col] = pd.to_numeric(df[col], errors='ignore')  # Convert to numeric, if possible
            
            # add the independent params to the dataframe
            if 'fov' in df.columns:
                if independent_params is not None:
                    for name in independent_params[0].keys():
                        if name in df.columns:
                            pass
                        else:
                            df[name] =[independent_params[fov][name] for fov in df['fov'].astype(int).tolist()]
            return df
        
        def split_df(df, lower, upper):
            if 'fov' not in df.columns:
                return df
            else:
                positions = df['fov'].unique()
                positions = positions[positions >= lower]
                positions = positions[positions <= upper]

                df = df[df['fov'].isin(positions)]
                df = df.sort_values(by='fov')
                df['fov'] = pd.Categorical(df['fov']).codes
            return df
        

        if locations[0].endswith('.h5'):
            is_h5 = True
            is_ds = False
        else:
            is_h5 = False
            is_ds = True


        if is_h5:
            close_h5_files()
            
            for i, location in enumerate(locations):
                for key, data in attributes.items():
                    if key not in ['_initialized', 'independent_params'] and not any(substring in key for substring in ['mask', 'image']):
                        if data is not None:
                            if isinstance(data, pd.DataFrame):
                                data = handle_df(data)
                                data = split_df(data, position_indexs[i-1] if i > 0 else 0, position_indexs[i]-1)
                                # print(f'saving {key}')
                                # data.to_hdf(location, key=f"{group_name}/{key}", mode='a',errors='ignore')
                                data = data.to_dict(orient='records')
                                data = json.dumps(data)
                            # else:
                            h5_file = h5py.File(location, 'a')

                            if group_name in h5_file:
                                group = h5_file[group_name]
                            else:
                                group = h5_file.create_group(group_name)

                            if key in group:
                                del group[key]
                            
                            print(f'saving {key}')
                            group[key] = data

                            h5_file.flush()
                            h5_file.close()
        else:
            for i, location in enumerate(locations):
                save_loc = os.path.join(location, group_name)
                os.makedirs(save_loc, exist_ok=True)
                for key, data in attributes.items():
                    if key not in ['_initialized', 'independent_params'] and not any(substring in key for substring in ['mask', 'image']):
                        if data is not None:
                            if isinstance(data, pd.DataFrame):
                                # data = handle_df(data)
                                data = split_df(data, position_indexs[i-1] if i > 0 else 0, position_indexs[i]-1)
                                print(f'Saving Key {key} to {location}')
                                csv_path = os.path.join(save_loc, f"{key}.csv")
                                data.to_csv(csv_path, index=False)
                            else:
                                json_path = os.path.join(save_loc, f"{key}.json")
                                with open(json_path, "w") as f:
                                    if isinstance(data, np.memmap):
                                        data = data.tolist()
                                    json.dump(data, f, indent=4)


class Save_Parameters(Saving):
    def run(self, p: int = None, t:int = None, data_container = None, parameters = None):
        data_container = DataContainer() if data_container is None else data_container
        parameters = Parameters() if parameters is None else parameters
        kwargs = self.load_in_parameters(p, t, parameters)
        results = self.main(parameters, **kwargs) 
        data_container.save_results(results, p, t, parameters)
        data_container.load_temp_data()
        return results
    
    def main(self, parameters, **kwargs):
        def recursively_save_dict_contents_to_group(h5file, path, dic):
            for key, item in dic.items():
                if isinstance(item, dict):
                    recursively_save_dict_contents_to_group(h5file, f"{path}/{key}", item)
                else:
                    h5file[f"{path}/{key}"] = item
        
        params = parameters.get_parameters()
        params_to_ignore = ['h5_file', 'local_dataset_location', 'images', 'masks', 'instances', 'state', 'temp']

        Analysis_name = kwargs['name']
        local_dataset_location = kwargs['local_dataset_location']


        # os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        # get todays date
        today = datetime.today()
        date = today.strftime("%Y-%m-%d")
        group_name = f'Analysis_{Analysis_name}_{date}'

        if local_dataset_location[0].endswith('.h5'):
            is_h5 = True
            is_ds = False
        else:
            is_h5 = False
            is_ds = True

        
        if is_h5:
            for h5 in kwargs['h5_files']:
                h5.close()
            close_h5_files()

            for i, locaction in enumerate(local_dataset_location):
                print(f'Saving Parameters to {locaction}')
                # save the parameters to the h5 file
                with h5py.File(locaction, 'a') as h5_file:
                # h5_file = h5py.File(locaction, 'r+')
                    if group_name in h5_file:
                        group = h5_file[group_name]
                    else:
                        group = h5_file.create_group(group_name)

                    # save the parameters to the h5 file
                    # remove params_to_ignore
                    for key in params_to_ignore:
                        if key in params:
                            del params[key]
                    
                    params = handle_dict(params)

                    # if dataset is already made, delete it
                    if 'parameters' in group:
                        del group['parameters']

                    recursively_save_dict_contents_to_group(group, 'parameters', params)

                    # save the nuc channel and cyto channel and fish channel at the top level
                    if 'nucChannel' in params:
                        group.attrs['nucChannel'] = params['nucChannel']
                    if 'cytoChannel' in params:
                        group.attrs['cytoChannel'] = params['cytoChannel']
                    if 'FISHChannel' in params:
                        group.attrs['FISHChannel'] = params['FISHChannel']
                
                    h5_file.flush()
                    h5_file.close()
        else:
            for i, location in enumerate(local_dataset_location):
                print(f'Saving Parameters to {location}')
                save_loc = os.path.join(location, group_name)
                os.makedirs(save_loc, exist_ok=True)
                filtered_params = {k: v for k, v in params.items() if k not in params_to_ignore}
                file_path = os.path.join(save_loc, "parameters.json")
                with open(file_path, "w") as f:
                    def convert_memmap_to_list(obj):
                        if isinstance(obj, np.memmap):
                            return obj.tolist()
                        elif isinstance(obj, dict):
                            return {k: convert_memmap_to_list(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_memmap_to_list(item) for item in obj]
                        elif isinstance(obj, tuple):
                            return tuple(convert_memmap_to_list(item) for item in obj)
                        return obj

                    filtered_params = convert_memmap_to_list(filtered_params)
                    json.dump(filtered_params, f, indent=4)


class Save_Images(Saving):
    def main(self, **kwargs):
        params = kwargs
        Analysis_name = params['name']
        local_dataset_location = params['local_dataset_location']
        images = params['images']
        position_indexs = params['position_indexs']

        if local_dataset_location[0].endswith('.h5'):
            is_h5 = True
            is_ds = False
        else:
            is_h5 = False
            is_ds = True

        today = datetime.today()
        date = today.strftime("%Y-%m-%d")
        group_name = f'Analysis_{Analysis_name}_{date}'
        # os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

        if is_h5:
            close_h5_files()

            # os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

            # save the images to the h5 file
            for i, locaction in enumerate(local_dataset_location):
                close_h5_files()
                with h5py.File(locaction, 'a') as h5:
                    if group_name in h5:
                        group = h5[group_name]
                    else:
                        group = h5.create_group(group_name)

                    if 'images' in group:
                        del group['images']

                    group.create_dataset('images', data=images[position_indexs[i-1] if i > 0 else 0:position_indexs[i]]) 
                    if group_name in h5:
                        group = h5[group_name]
                    else:
                        group = h5.create_group(group_name)

                    # if dataset is already made, delete it
                    if 'images' in group:
                        del group['images']

                    start_idx = position_indexs[i - 1] if i > 0 else 0
                    end_idx = position_indexs[i]
                    data_slice = images[start_idx:end_idx]

                    dset = group.create_dataset(
                        'images',
                        shape=data_slice.shape,
                        dtype=data_slice.dtype,
                        chunks=(1,) + data_slice.shape[1:], 
                        compression='gzip'
                    )
                    dset[:] = data_slice

                    h5.flush()
                    h5.close()
            
        else:
            for i, location in enumerate(local_dataset_location):
                save_loc = os.path.join(location, group_name)
                os.makedirs(save_loc, exist_ok=True)
                start_idx = position_indexs[i - 1] if i > 0 else 0
                end_idx = position_indexs[i]
                print(f'Saving Images {start_idx}-{end_idx} to {location}')
                filename = os.path.join(save_loc, f"images.tif")
                img = images[start_idx:end_idx]
                tiffile.imwrite(filename, img)


class Save_Masks(Saving):
    def main(self, **kwargs):
        params = kwargs

        local_dataset_location = params['local_dataset_location']
        masks = params['masks']
        if local_dataset_location[0].endswith('.h5'):
            is_h5 = True
            is_ds = False
        else:
            is_h5 = False
            is_ds = True

        position_indexs = params['position_indexs']
        # os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

        if is_h5:
            for h5 in params['h5_files']:
                h5.close()
            close_h5_files()

            for i, location in enumerate(local_dataset_location):
                # with h5py.File(location, 'a') as h5:
                for k in masks.keys():
                    print(f'Saving masks {k} to {location}')
                    chunk_size = (1,) + masks[k].shape[1:]  # Define chunk size
                    start_idx = position_indexs[i - 1] if i > 0 else 0
                    end_idx = position_indexs[i]
                    if isinstance(masks[k], da.Array):
                        mdata = masks[k][start_idx:end_idx].rechunk(chunk_size)
                    else:
                        mdata = da.from_array(masks[k][start_idx:end_idx], chunks=chunk_size)
                    da.to_hdf5(location, f'/{k}', mdata)

        else:
            for i, location in enumerate(local_dataset_location):
                for k, data in masks.items():
                    start_idx = position_indexs[i - 1] if i > 0 else 0
                    end_idx = position_indexs[i]
                    print(f'Saving masks {k} {start_idx}-{end_idx} to {location}')
                    save_loc = os.path.join(location, 'masks')
                    os.makedirs(save_loc, exist_ok=True)
                    filename = os.path.join(save_loc, f"{k}.tif")
                    mask = data[start_idx:end_idx]
                    tiffile.imwrite(filename, mask)







