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
import json

from src.GeneralStep import FinalizingStepClass
from src.Parameters import Parameters, DataContainer, ScopeClass, Settings, Experiment
from src.NASConnection import NASConnection
from PIL import Image

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

        h5_file = params['h5_file']
        Analysis_name = params['name']
        local_dataset_location = params['local_dataset_location']
        independent_params = params['independent_params']
        position_indexs = params['position_indexs']

        # get todays date
        today = datetime.today()
        date = today.strftime("%Y-%m-%d")

        data = data_container.load_temp_data()

        self.save(data, local_dataset_location, h5_file, f'Analysis_{Analysis_name}_{date}', position_indexs, independent_params)

    def save(self, attributes, locations, h5_file: str, group_name: str, position_indexs: list[int], independent_params: dict[str, Any] ):
        # get all the attributes of the class

        # def handle_df(df):
        #     for col in df.columns:
        #         if df[col].dtype == 'O':  # Object type
        #             if df[col].map(type).nunique() == 1 and isinstance(df[col].iloc[0], str):
        #                 df[col] = df[col].astype(str)  # Convert to string
        #             else:
        #                 df[col] = pd.to_numeric(df[col], errors='ignore')  # Convert to numeric, if possible
            
        #     # add the independent params to the dataframe
        #     if 'fov' in df.columns:
        #         if independent_params is not None:
        #             for name in independent_params[0].keys():
        #                 if name in df.columns:
        #                     pass
        #                 else:
        #                     df[name] =[independent_params[fov][name] for fov in df['fov'].astype(int).tolist()]
        #     return df
        


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
            if not os.path.exists(os.path.join(location, group_name)):
                os.makedirs(os.path.join(location, group_name))

            for key, data in attributes.items():
                if key not in ['_initialized', 'independent_params'] and not any(substring in key for substring in ['mask', 'image']):

                    if data is not None:
                        if isinstance(data, pd.DataFrame):
                            # data = handle_df(data)
                            data = split_df(data, position_indexs[i-1] if i > 0 else 0, position_indexs[i]-1)
                            print(f'saving {key}')
                            data.to_csv(os.path.join(location, group_name, f'{key}.csv'))
                            # data.to_hdf(location, f'{group_name}/{key}', mode='a', format='table', data_columns=True)

                        else:
                            # h5_file = h5py.File(location, 'a')

                            # if group_name in h5_file:
                            #     group = h5_file[group_name]
                            # else:
                            #     group = h5_file.create_group(group_name)

                            # if key in group:
                            #     del group[key]
                            
                            print(f'saving {key}')
                            with open(os.path.join(location, group_name, f'{key}.json'), 'w') as json_file:
                                json.dump(data, json_file)

                            # group.create_dataset(key, data=data)

                            # h5_file.flush()
                            # h5_file.close()


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
        params_to_ignore = ['h5_file', 'local_dataset_location', 'images', 'masks', 'instances', 'state']

        h5_file = kwargs['h5_file']
        Analysis_name = kwargs['name']
        local_dataset_location = kwargs['local_dataset_location']

        close_h5_files()

        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        # get todays date
        today = datetime.today()
        date = today.strftime("%Y-%m-%d")

        for i, locaction in enumerate(local_dataset_location):
            with open(os.path.join(locaction, f'parameters_{Analysis_name}_{date}.json'), 'w') as json_file:
                json.dump(params, json_file)
            # save the parameters to the h5 file
            # with h5py.File(locaction, 'r+') as h5_file:
            # # h5_file = h5py.File(locaction, 'r+')
            #     group_name = f'Analysis_{Analysis_name}_{date}'
            #     if group_name in h5_file:
            #         group = h5_file[group_name]
            #     else:
            #         group = h5_file.create_group(group_name)

            #     # save the parameters to the h5 file
            #     # remove params_to_ignore
            #     for key in params_to_ignore:
            #         if key in params:
            #             del params[key]
                
            #     params = handle_dict(params)

            #     # if dataset is already made, delete it
            #     if 'parameters' in group:
            #         del group['parameters']

            #     recursively_save_dict_contents_to_group(group, 'parameters', params)

            #     # save the nuc channel and cyto channel and fish channel at the top level
            #     if 'nucChannel' in params:
            #         group.attrs['nucChannel'] = params['nucChannel']
            #     if 'cytoChannel' in params:
            #         group.attrs['cytoChannel'] = params['cytoChannel']
            #     if 'FISHChannel' in params:
            #         group.attrs['FISHChannel'] = params['FISHChannel']
            
            #     h5_file.flush()
            #     h5_file.close()


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
            group_name = f'Analysis_{Analysis_name}_{date}'
            img_path = os.path.join(locaction, f'{group_name}', 'images.tiff')
            imgs_to_save = images[position_indexs[i-1] if i > 0 else 0:position_indexs[i]]
            imgs_to_save = [Image.fromarray(img) for img in imgs_to_save]
            imgs_to_save[0].save(img_path, save_all=True, append_images=imgs_to_save[1:])


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
                        
                        # Create masks folder if it doesn't exist
                        masks_folder = os.path.join(locaction, 'masks')
                        if not os.path.exists(masks_folder):
                            os.makedirs(masks_folder)
                        
                        # Save each mask as a separate file
                        combined_mask = np.concatenate(masks[position_indexs[i-1] if i > 0 else 0:position_indexs[i]], axis=0)
                        mask_path = os.path.join(masks_folder, f'{k}.tiff')
                        Image.fromarray(combined_mask).save(mask_path)







