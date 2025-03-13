from dataclasses import dataclass, fields, field
from typing import Union
from abc import ABC, abstractmethod
import pathlib
import numpy as np
import os
from pycromanager import Dataset
import dask.array as da
import dask.dataframe as dd
import dask.bag as db
from dask_image import imread
import h5py
from dataclasses import asdict
import gc
import pandas as pd
from skimage.io import imsave
import tempfile
import json
import dask_image.imread as dask_imread


class Parameters(ABC):
    """
    This class is used to store the parameters of the pipeline.
    Alls children class will be singletons and will be stored in the _instances list.
    """
    _instances = []

    def __new__(cls, *args, **kwargs):
        for instance in cls._instances:
            if instance.__class__ == cls:
                return instance
            
        instance = super().__new__(cls)
        instance.__init__(*args, **kwargs)
        cls._instances.append(instance)
        return instance

    def __init__(self):
        super().__init__()
        self.state = 'global'
        self.instances = []
    
    @classmethod
    def get_all_instances(cls):
        # Class method to return all instances of the parent class
        return cls._instances

    @classmethod
    def clear_instances(cls):
        # Class method to clear all instances of the parent class

        # deletes all instances of the class
        # for instance in cls._instances:
        #     del instance
        cls._instances = []

    @classmethod
    def validate(cls):
        # make sure settings, scope, experiemnt, and datacontainer are all initialized
        # if len(cls._instances) < 3 or len(cls._instances) > 5:
        #     raise ValueError(f"Settings, ScopeClass, and Experiment must all be initialized")
        # makes sure ScopeClass is in _instances
        # if not any(isinstance(instance, ScopeClass) for instance in cls._instances):
        #     raise ValueError(f"ScopeClass must be initialized")
        # # makes sure Experiment is in _instances
        # if not any(isinstance(instance, Experiment) for instance in cls._instances):
        #     raise ValueError(f"Experiment must be initialized")
        # # makes sure DataContainer is in _instances
        # if not any(isinstance(instance, DataContainer) for instance in cls._instances):
        #     print(f"DataContainer must be initialized")
        # # makes sure Settings is in _instances
        # if not any(isinstance(instance, Settings) for instance in cls._instances):
        #     raise ValueError(f"Settings must be initialized")

        # Class method to validate all instances of the parent class
        for instance in cls._instances:
            instance.validate_parameters()

    @classmethod
    def initialize_parameters_instances(cls):
        ScopeClass()
        Experiment()
        DataContainer()
        Settings()

    def update_parameters(self, kwargs: dict):
        # Class method to update all instances of the parent class
        if kwargs is None:
            return None
        used_keys = []
        instances = Parameters()._instances if self.state == 'global' else self.instances
        for instance in instances:
            for key, value in kwargs.items():
                # check if the key exists in the instance
                if hasattr(instance, key):
                    setattr(instance, key, value)
                    print(f'Overwriting {key} in {instance.__class__.__name__}')
                    used_keys.append(key)

        # if there are any kwargs left, add them to the settings class
        # remove the used keys
        for key in used_keys:
            if key in kwargs:
                kwargs.pop(key)
        
        if kwargs:
            print(f'Adding leftover kwargs to Settings')
            # find the settings instance
            for instance in instances:
                if instance.__class__.__name__ == 'Settings':
                    for key, value in kwargs.items():
                        setattr(instance, key, value)
                        print(f'Adding {key} to {instance.__class__.__name__}')

    def validate_parameters(self):
        pass

    def todict(self):
        # Convert all attributes of the instance to a dictionary
        # return {field.name: getattr(self, field.name) for field in fields(self)}
        # return {**{field.name: getattr(self, field.name) for field in fields(self)}, **vars(self)}
        return vars(self)
    
    def reset(self):
        for field in fields(self):
            setattr(self, field.name, field.default)

    def __str__(self):
        string = f'{self.__class__.__name__}:\n'
        for key, value in self.todict().items():
            # make one large string of all the parameters
            string += f'{key}: {value} \n'
        return string

    def get_parameters(self) -> dict:
        # Get all the parameters of all instances of the class
        params = {}
        acceptableDuplicateKeys = ['kwargs', 'state', 'instances', 'init', 'state']
        keys2remove = ['_instances', 'instances']


        # Ensure all instances have the 'state' attribute
        for instance in Parameters._instances:
            if not hasattr(instance, 'state'):
                instance.state = 'global'  # Assign a default value if missing

        if self.state == 'global':
            for instance in Parameters()._instances:
                instance._update()
                # check for duplicates and raise an error if found
                keys2add = set(instance.todict().keys())
                duplicate_keys = list(set(params.keys()).intersection(keys2add))
                duplicate_keys = [key for key in duplicate_keys if key not in {'kwargs', 'state', 'instances', 'init'}]
                if duplicate_keys:
                    raise ValueError(f"Duplicate parameter found: {duplicate_keys}")
                params.update(instance.todict())
        else:
            for instance in self.instances:
                instance._update()
                # check for duplicates and raise an error if found
                keys2add = set(instance.todict().keys())
                data2add = instance.todict()
                duplicate_keys = list(set(params.keys()).intersection(keys2add))
                if duplicate_keys:
                    for key in acceptableDuplicateKeys:
                        if key in duplicate_keys:
                            duplicate_keys.remove(key)
                            data2add.pop(key)
                    if duplicate_keys:
                        raise ValueError(f"Duplicate parameter found: {duplicate_keys}")
                keys_to_remove = [key for key in data2add.keys() if key in keys2remove]
                for key in keys_to_remove:
                    data2add.pop(key)
    
                params.update(data2add)
        return params

    def isolate(self):
        self.state = 'local'
        self.instances = self.get_all_instances()
        for i in self.instances:
            i.state = self.state
        self.clear_instances()

    def _update(self):
        pass

class ScopeClass(Parameters):
    """
    Class to store the parameters of the microscope.
    Attributes:
    voxel_size_yx: int
    spot_z: int
    spot_yx: int

    Default values will be for Terminator Scope
    
    """


    def __init__(self,    
                 voxel_size_yx: int = 130,
                 spot_z: int = 500,
                 spot_yx: int = 360,
                **kwargs):
        super().__init__()
        if not hasattr(self, 'init'):
            self.state = 'global'
            self.voxel_size_yx = voxel_size_yx
            self.spot_z = spot_z
            self.spot_yx = spot_yx
            self.init = True

        if kwargs is not None: # TODO this needs to be moved up but it has issues being in parmaeters class
            for key, value in kwargs.items():
                setattr(self, key, value)


class Experiment(Parameters):
    """
    
    This class is used to store the information of the experiment that will be processed.
    It is expected that the data located in local_img_location and local_mask_location will be processed
    and will be in the format of : [Z, Y, X, C]

    The data must already be downloaded at this point and are in tiff for each image
    """

    def __init__(self, 
                    initial_data_location: Union[str, list[str]] = None,
                    index_dict: dict = None,
                    nucChannel: int = None,
                    cytoChannel: int = None,
                    FISHChannel: Union[list[int], int] = None,
                    voxel_size_z: int = 300,
                    independent_params: Union[dict, list[dict]] = [{}],
                    timestep_s: float = None,
                    **kwargs):
        super().__init__()
        self.state = 'global'
        if not hasattr(self, 'init'):
            self.initial_data_location = initial_data_location
            self.index_dict = index_dict
            self.nucChannel = nucChannel
            self.cytoChannel = cytoChannel
            self.FISHChannel = FISHChannel
            self.voxel_size_z = voxel_size_z
            self.independent_params = independent_params
            self.timestep_s = timestep_s
            self.init = True
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def validate_parameters(self):
        if self.initial_data_location is None and DataContainer().local_dataset_location is None:
            raise ValueError("initial_data_location or local_dataset_location must be set")
        
        if self.nucChannel is None:
            print("nucChannel not set")

        if self.cytoChannel is None:
            print("cytoChannel not set")

        if self.FISHChannel is None:
            print("FISHChannel not set")

        if type(self.FISHChannel) is int:
            self.FISHChannel = [self.FISHChannel]

        if type(self.initial_data_location) is str:
            self.initial_data_location = [self.initial_data_location]

        # make sure each independent parameter has the same keys
        if self.independent_params is not None:
            if type(self.independent_params) is list:
                for i, params in enumerate(self.independent_params):
                    if i == 0:
                        keys = set(params.keys())
                    else:
                        if keys != set(params.keys()):
                            raise ValueError(f"Independent parameters must have the same keys")
            else:
                self.independent_params = [self.independent_params]
        else:
            self.independent_params = [{}]



class DataContainer(Parameters):
    def __init__(self, 
                local_dataset_location: Union[list[pathlib.Path], pathlib.Path] = None,
                total_num_chunks: int = None,
                images: da = None,
                masks: da = None,
                temp: tempfile.TemporaryDirectory = None,
                clear_after_error: bool = True,
                **kwargs):
        super().__init__()
        if not hasattr(self, 'init'):
            self.local_dataset_location = local_dataset_location
            self.total_num_chunks = total_num_chunks
            self.images = images
            self.masks = masks
            self.temp = temp
            self.clear_after_error = clear_after_error
            self.init = True
            self.mask_locations = None
            self.image_locations = None

        self.state = 'global'
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def __setattr__(self, name, value):
        if name == 'total_num_chunks':
            for instance in Parameters._instances:
                if isinstance(instance, Settings):
                    l = [instance.num_chunks_to_run, value]
                    instance.num_chunks_to_run = min(i for i in l if i is not None)

        if name == 'images' and value is not None:
            (self.PP, self.TT, self.CC, self.ZZ, self.YY, self.XX) = value.shape
            
        super().__setattr__(name, value)
    
    def validate_parameters(self):
        if type(self.local_dataset_location) is str:
            self.local_dataset_location = [self.local_dataset_location]

    def save_results(self, kwargs: dict, p:int = None, t:int = None, parameters = None):
        if self.temp is None:
            self.temp = tempfile.TemporaryDirectory(dir=os.getcwd(), ignore_cleanup_errors=True)
        parameters = Parameters() if parameters is None else parameters
        if kwargs is not None:
            for name, value in kwargs.items():
                # check if it a mask
                if 'mask' in name.lower():
                    if 'location' not in name.lower():
                        # if I know where that masks exists and it doesnt need to be changed
                        if self.mask_locations is None:
                            pass
                        else:
                            if isinstance(value, dict):
                                for key in value.keys():
                                    key = name.split('_')[0]
                                    if key in self.masks:
                                        if p is not None and t is not None:
                                            if self.masks[key][p, t] != value[key]:
                                                self.save_masks(key, value[key], p, t, parameters)
                                                if not isinstance(self.mask_locations, list):
                                                    self.mask_locations = []
                                                self.mask_locations.append(key)
                                        else:
                                            if self.masks[key] != value[key]:
                                                self.save_masks(key, value[key], p, t, parameters)
                                                if not isinstance(self.mask_locations, list):
                                                    self.mask_locations = []
                                                self.mask_locations.append(key)
                                    else:
                                        self.save_masks(key, value[key], p, t, parameters)
                                        if not isinstance(self.mask_locations, list):
                                            self.mask_locations = []
                                        self.mask_locations.append(key)
                            else:
                                key = name.split('_')[0]
                                self.save_masks(key, value, p, t, parameters)
                                if not isinstance(self.mask_locations, list):
                                    self.mask_locations = []
                                self.mask_locations.append(key)
                            
                    elif 'location' in name.lower():
                        self.mask_locations = value
                # check if it a image
                elif 'image' in name.lower():
                        if 'location' not in name.lower():
                            if p is None and t is None:
                                if self.images and self.images[key] != value:
                                    self.save_images(name, value, p, t)
                                    self.image_locations = 'temp'
                            else:
                                if self.images and self.images[p, t] != value:
                                    self.save_images(name, value, p, t)
                                    self.image_locations = 'temp'

                        elif 'location' in name.lower():
                            self.image_locations = value
                elif isinstance(value, pd.DataFrame):
                    # check the type:
                    # if pd.df save as csv
                    self.save_df(name, value)
                elif isinstance(value, np.ndarray):
                    self.save_np(name, value)
                elif name in parameters.get_parameters().keys():
                    parameters.update_parameters({name: value})
                else:
                    # else save as json
                    self.save_extra(name, value)

    def save_masks(self, name, mask, p:int = None, t:int = None, parameters = None):
        parameters = Parameters() if parameters is None else parameters
        # params = parameters.get_parameters()
        if not isinstance(self.mask_locations, list):
            self.mask_locations[name] = os.path.join(self.temp.name, name)
        nump, numt, numc, numz, numy,numx = self.images.shape
        if mask is not None: 
            if hasattr(self, "masks"):
                if isinstance(self.masks, dict):
                    for key in list(self.masks.keys()):
                        del self.masks[key]
                del self.masks
            if p is None and t is None:
                masks = da.rechunk(da.asarray(mask, dtype=np.int8),[1, 1, -1, -1, -1])
                save_path = os.path.join(self.temp.name, name)
                da.to_npy_stack(save_path, masks, axis=0)
            else:
                save_path = os.path.join(self.temp.name, name)
                if not os.path.exists(save_path):
                    # Create a big array of zeros if the directory does not exist
                    zero_array = da.rechunk(da.zeros(shape=[nump, numt, numz, numy, numx], dtype=np.int8), [1,1,-1,-1,-1])
                    da.to_npy_stack(save_path, zero_array, axis=0)
                    del zero_array
                    gc.collect()
                # Load the existing npy stack
                # existing_stack = da.from_npy_stack(save_path)
                # Replace the specific slice with the new mask
                previous_data = np.load(os.path.join(save_path, f'{p}.npy'), mmap_mode='r+')
                if numz != mask.shape[0]:
                    mask = np.tile(mask, (numz, 1, 1))
                previous_data[0, t] = mask
                del previous_data
                # mask = mask[np.newaxis, np.newaxis,:]
                # existing_stack[p, t] = mask
                # da.to_npy_stack(save_path, existing_stack, axis=0)
                # refs = gc.get_referrers(previous_data)
                # gc.collect()
                # previous_data = np.memmap(os.path.join(save_path, f'{p}.npy'), mode='r+')
                # with open(os.path.join(save_path, f'{p}.npy'), 'wb') as f:
                #     np.save(f, previous_data)
        else:
            print('returned empty mask')

    def save_images(self, name, image, p:int = None, t:int = None):
        if p is None and t is None:
            images = da.rechunk(da.asarray(image), (1, -1, -1, -1,-1,-1))
            da.to_npy_stack(os.path.join(self.temp.name, 'images'), images) 
        
        elif p is not None and t is not None:
            np.save(os.path.join(self.temp.name, 'images', f'{p}.npy'), image)

    def save_df(self, name, value):
        folder_path = os.path.join(self.temp.name, name)
        os.makedirs(folder_path, exist_ok=True)
        existing_files = [f for f in os.listdir(folder_path)]
        file_index = len(existing_files) + 1
        value.to_csv(os.path.join(folder_path, f'{name}_{file_index}.csv'), index=False)

    def save_extra(self, name, value):
        folder_path = os.path.join(self.temp.name, name)
        os.makedirs(folder_path, exist_ok=True)
        existing_files = [f for f in os.listdir(folder_path)]
        file_index = len(existing_files) + 1
        with open(os.path.join(folder_path, f'{name}_{file_index}.json'), 'w') as f:
            json.dump(value, f)

    def save_np(self, name, value):
        folder_path = os.path.join(self.temp.name, name)
        os.makedirs(folder_path, exist_ok=True)
        existing_files = [f for f in os.listdir(folder_path) if f.startswith(name) and f.endswith('.csv')]
        file_index = len(existing_files) + 1
        np.save(os.path.join(folder_path, f'{name}_{file_index}.npy'), value)

    def load_temp_data(self):
        if self.temp is not None and os.path.exists(self.temp.name):
            # self.temp = tempfile.TemporaryDirectory(dir=os.getcwd(), ignore_cleanup_errors=True)

            # Load masks and images
            if hasattr(self, 'images') and self.images is not None:
                del self.images
            if hasattr(self, 'masks') and self.masks is not None:
                del self.masks
            gc.collect()

            # Load everything else:
            # go through all remaining folders in temp
            data = {}

            if self.image_locations is not None and self.image_locations[0] != 'temp':
                if self.image_locations[0].endswith('.h5'):
                    h5_files = [h5py.File(H5_location, 'r') for H5_location in self.image_locations]
                    images = [da.from_array(h5['raw_images']) for h5 in h5_files]
                    images = da.concatenate(images, axis=0)
                    images = images.rechunk((1, 1, -1, -1, -1, -1))
                    data['images'] = images
                    setattr(self, 'images', images)

                else:
                    dss = [Dataset(l) for l in self.image_locations]
                    images = [ds.as_array(['position', 'time', 'channel','z']) for ds in dss]
                    images = da.concatenate(images, axis=0)
                    data['images'] = images
                    setattr(self, 'images', images)

            if self.mask_locations is not None and not isinstance(self.mask_locations, list):
                masks = {}
                # Find tifs in the directory and add them to the dask dictionary
                if self.mask_locations is not None and self.mask_locations:
                    if list(self.mask_locations.values())[0][0].endswith('.h5'):
                        for key in self.mask_locations.keys():
                            h5_files = [h5py.File(H5_location, 'r') for H5_location in self.mask_locations[key]]
                            masks[key] = [da.from_array(h5[key]) for h5 in h5_files]
                            masks[key] = da.concatenate(masks[key], axis=0)
                            data['masks'][key] = masks[key].rechunk((1, 1, -1, -1, -1))
                            for h5 in h5_files:
                                h5.close()
                        setattr(self, 'masks', masks)
                            
                    else:
                        for key in self.mask_locations.keys():
                            tif_data = [dask_imread.imread(l) for l in self.mask_locations[key]]
                            masks[key] = da.concatenate(tif_data, axis=0)
                            data['masks'][key] = masks[key].rechunk((1, 1, -1, -1, -1))
                        setattr(self, 'masks', masks)


            masks = {}
            for folder in os.listdir(self.temp.name):
                if isinstance(self.mask_locations, list) and folder in self.mask_locations:
                    is_mask = True
                else:
                    is_mask = False
                folder_path = os.path.join(self.temp.name, folder)
                files = os.listdir(folder_path)
                csv_files = [f for f in files if f.endswith('.csv')]
                json_files = [f for f in files if f.endswith('.json')]
                npy_files = [f for f in files if f.endswith('.npy')]

                if len(csv_files) > 0 and len(json_files) == 0 and len(npy_files) == 0:
                    if len(csv_files) == 1:
                        df = pd.read_csv(os.path.join(folder_path, csv_files[0]))
                    else:
                        df = []
                        for c in csv_files:
                            df.append(pd.read_csv(os.path.join(folder_path, c)))
                        df = pd.concat(df, axis=0)
                    setattr(self, folder, df)
                    data[folder] = df

                elif len(csv_files) == 0 and len(json_files) > 0 and len(npy_files) == 0:
                    if len(json_files) == 1:
                        with open(os.path.join(folder_path, json_files[0]), 'r') as f:
                            json_data = json.load(f)
                    else:
                        json_data = []
                        for j in json_files:
                            with open(os.path.join(folder_path, j), 'r') as f:
                                json_data.append(json.load(f))
                    setattr(self, folder, json_data)
                    data[folder] = json_data

                elif len(csv_files) == 0 and len(json_files) == 0 and len(npy_files) > 0:
                    try:
                        npy_data = da.from_npy_stack(folder_path, mmap_mode='r')
                    except:
                        if len(npy_files) == 1:
                            npy_data = np.load(os.path.join(folder_path, npy_files[0]), mmap_mode='r')
                        else:
                            npy_data = []
                            for n in npy_files:
                                npy_data.append(np.load(os.path.join(folder_path, n)), mmap_mode='r')
                            npy_data = np.concatenate(npy_data, axis=0)
                    if is_mask:
                        masks[folder] = npy_data.rechunk((1,1,-1,-1,-1))
                    else:
                        setattr(self, folder, npy_data)
                        data[folder] = npy_data

                else:
                    raise ValueError('All temp files must either be csv, json, or npy and not a mix')
            data['masks'] = masks
            setattr(self, 'masks', masks)
            # self.update_parameters(data)
            return data

    def delete_temp(self):
        if self.temp is not None:
            self.temp.cleanup()

    def _update(self):
        self.load_temp_data()

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
class Settings(Parameters):
    def __init__(self,     
                    name: str = None,
                    return_data_to_NAS: bool = True,
                    NUMBER_OF_CORES: int = 4,
                    save_files: bool = True,
                    num_chunks_to_run: int = 100_000,
                    connection_config_location: str = str(os.path.join(repo_path, 'config_nas.yml')),
                    display_plots: bool = True,
                    load_in_mask: bool = False,
                    mask_structure: dict = None,
                    order: str = 'pt',
                    share_name: str = 'share',
                    **kwargs):
        super().__init__()
        if not hasattr(self, 'init'):
            self.name = name
            self.return_data_to_NAS = return_data_to_NAS
            self.NUMBER_OF_CORES = NUMBER_OF_CORES
            self.save_files = save_files
            self.num_chunks_to_run = num_chunks_to_run
            self.connection_config_location = connection_config_location
            self.display_plots = display_plots
            self.load_in_mask = load_in_mask
            self.mask_structure = mask_structure
            self.order = order
            self.state = 'global'
            self.share_name = share_name
            self.init = True

        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)
        if self.connection_config_location is None:
            self.connection_config_location = str(os.path.join(repo_path, 'config_nas.yml'))
        if self.mask_structure is None:
            self.mask_structure = {'masks': ('ptczyx', None, None), 
                                   'cell_mask': ('zyx', 'cytoChannel', 'masks'), 
                                   'nuc_mask': ('zyx', 'nucChannel', 'masks')}
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def validate_parameters(self):
        if self.name is None:
            raise ValueError("Name must be set")


#%% Required Params
class required_params(ABC):
    @abstractmethod
    def validate_parameter(self):
        ...
