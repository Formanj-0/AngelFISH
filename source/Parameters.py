from tempfile import TemporaryDirectory
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
# import h5py
import dask.array as da
import os
from ndtiff import Dataset
import pandas as pd


class Parameters:
    def __init__(self):
        self.voxel_size_yx = 130
        self.voxel_size_z = 500
        self.spot_z = 500
        self.spot_yx = 360
        self.local_dataset_location = None
        self.clear_after_error = True
        self.analysis_name = 'DEFAULT_NAME'
        self.number_of_cores = 4
        self.num_chunks_to_run = 100_000
        self.connection_config_location = ''
        self.display_plots = True
        self.load_in_mask = True
        self.order = 'pt'
        self.state = 'global'
        self.share_name = 'share'
        self.log_location = r'Users\Jack\All_Analysis'
        self.initial_data_location = None
        self.nucChannel = None
        self.cytoChannel = None
        self.FISHChannel = None
        self.experimental_params = [{}]
        self.timestep_s = None

    def to_json(self):
        return json.dumps(self.load(), default=str)

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        instance = cls()
        for key, value in data.items():
            setattr(instance, key, value)
        return instance



from pathlib import Path
import zarr
from numcodecs import JSON
import dask.dataframe as dd

class Data:
    def __init__(self, zarr_path):
        self._zarr_path = Path(zarr_path)
        self._ds = None
        self._loaded = False

        self._ds = self.dataset

    @property
    def zarr_path(self):
        return self._zarr_path

    @zarr_path.setter
    def zarr_path(self, new_path):
        self._zarr_path = Path(new_path)
        self._loaded = False  # force reload on next access

    @property
    def dataset(self):
        """Lazily load the dataset if not already loaded."""
        if not self._loaded:
            self._ds = zarr.open(self._zarr_path, mode='a')  # Open in append mode for read/write access
            self._loaded = True
        return self._ds


    def __getattr__(self, name):
        if name in ['_zarr_path', '_ds', '_loaded']:
            return self.__dict__.get(name, None)

        if '_zarr_path' in self.__dict__ and self._zarr_path is not None and not self._zarr_path.exists():
            self._zarr_path.mkdir(parents=True, exist_ok=True)
            zarr.open(self._zarr_path, mode='w')

        result = self.dataset[name]

        if isinstance(result, zarr.Array):
            return da.from_array(result)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        if name in ['_zarr_path', '_ds', '_loaded']:
            super().__setattr__(name, value)
            return

        if isinstance(value, da.Array):
            value = value.compute()
            self.dataset[name] = zarr.array(value)
        elif isinstance(value, np.ndarray):
            self.dataset[name] = zarr.array(value)
        elif isinstance(value, pd.DataFrame):
                ddf = dd.from_pandas(value, npartitions=1)
                ddf.to_zarr(self._zarr_path / name)
        else:
            self.dataset[name] = json.dumps(value)

    def append(self, newValues: dict):
        for k, v in newValues.items():
            setattr(self, k, v)

    def __str__(self):
        return f"Data(zarr_path={self._zarr_path}, loaded={self._loaded}, dataset_keys={list(self._ds.keys()) if self._ds else []})"



if __name__  == '__main__':
    params = Parameters()
    data = Data()

    params.analysis_name = 'analysis_name'

    print(params)
    print(data)

    print(params.__dict__)
    print(data.__dict__)


























