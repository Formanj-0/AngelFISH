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

def close_h5_files():
    for obj in gc.get_objects():
        if isinstance(obj, h5py.File):
            # check if the file is open
            try:
                if 'temp' not in obj.filename:
                    print(f"Closing {obj.filename}")
                    obj.close()
            except:
                pass

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
    def main(self, **kwargs):
        params = kwargs

        h5_file = params['h5_file']
        Analysis_name = params['name']
        local_dataset_location = params['local_dataset_location']
        independent_params = params['independent_params']
        position_indexs = params['position_indexs']

        # get todays date
        today = datetime.today()
        date = today.strftime("%Y-%m-%d")

        data = DataContainer().load_temp_data()

        self.save(data, local_dataset_location, h5_file, f'Analysis_{Analysis_name}_{date}', position_indexs, independent_params)

    def save(self, attributes, locations, h5_file: str, group_name: str, position_indexs: list[int], independent_params: dict[str, Any] ):
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
        
        close_h5_files()
        
        for i, location in enumerate(locations):
            for key, data in attributes.items():
                if key not in ['_initialized', 'independent_params']:
                    if data is not None:
                        if isinstance(data, pd.DataFrame):
                            data = handle_df(data)
                            data = split_df(data, position_indexs[i-1] if i > 0 else 0, position_indexs[i]-1)
                            data.to_hdf(location, f'{group_name}/{key}', mode='a', format='table', data_columns=True)

                        else:
                            h5_file = h5py.File(location, 'a')

                            if group_name in h5_file:
                                group = h5_file[group_name]
                            else:
                                group = h5_file.create_group(group_name)

                            if key in group:
                                del group[key]
                            
                            group.create_dataset(key, data=data)

                            h5_file.close()


class Save_Parameters(Saving):
    def main(self, **kwargs):
        def recursively_save_dict_contents_to_group(h5file, path, dic):
            for key, item in dic.items():
                if isinstance(item, dict):
                    recursively_save_dict_contents_to_group(h5file, f"{path}/{key}", item)
                else:
                    h5file[f"{path}/{key}"] = item
        
        params = kwargs
        params_to_ignore = ['h5_file', 'local_dataset_location', 'images', 'masks', 'instances', 'state']

        h5_file = DataContainer().h5_file
        Analysis_name = params['name']
        local_dataset_location = params['local_dataset_location']

        close_h5_files()

        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        # get todays date
        today = datetime.today()
        date = today.strftime("%Y-%m-%d")

        for i, locaction in enumerate(local_dataset_location):
            # save the parameters to the h5 file
            h5_file = h5py.File(locaction, 'r+')
            group_name = f'Analysis_{Analysis_name}_{date}'
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
        
            h5_file.close()


class Save_Images(Saving):
    def main(self, **kwargs):
        params = kwargs

        h5_file = params['h5_file']
        Analysis_name = params['name']
        local_dataset_location = params['local_dataset_location']
        images = params['images']
        position_indexs = params['position_indexs']

        close_h5_files()

        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        # get todays date
        today = datetime.today()
        date = today.strftime("%Y-%m-%d")

        # save the images to the h5 file
        for i, locaction in enumerate(local_dataset_location):
            if h5_file[i]:
                h5_file[i].close()
            h5 = h5py.File(locaction, 'r+')
            group_name = f'Analysis_{Analysis_name}_{date}'
            if group_name in h5:
                group = h5[group_name]
            else:
                group = h5.create_group(group_name)

            # if dataset is already made, delete it
            if 'images' in group:
                del group['images']

            # save the images to the h5 file
            group.create_dataset('images', data=images[position_indexs[i-1] if i > 0 else 0:position_indexs[i]])

            h5.close()


class Save_Masks(Saving):
    def main(self, **kwargs):
        params = kwargs

        local_dataset_location = params['local_dataset_location']
        # masks = params['masks']
        h5_file = params['h5_file']
        position_indexs = params['position_indexs']
        mask_structure = params['mask_structure']

        close_h5_files()

        # os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

        for k, (s, c, p) in mask_structure.items():
            for i, locaction in enumerate(local_dataset_location):
                if p is None:
                    with h5py.File(locaction, 'r+') as h5:
                        masks = params[k].compute()
                        
                        # check if the dataset is already made
                        if f'/{k}' in h5:
                            del h5[f'/{k}']

                        chunk_size = (1,) + masks.shape[1:]  # Define chunk size
                        h5.create_dataset('/masks', data=masks[position_indexs[i-1] if i > 0 else 0:position_indexs[i]], chunks=chunk_size, compression="gzip")
                        







