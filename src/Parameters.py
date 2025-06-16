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
        self.connection_config_location = os.path.join(os.path.dirname(__file__), '..', 'config_nas.yml')
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
        non_default_values = {
            key: value for key, value in self.__dict__.items()
            if getattr(Parameters(), key, None) != value
        }
        return json.dumps(non_default_values, default=str)

    def from_json(self, json_str):
        import ast
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Try to parse as Python dict string (with single quotes)
            data = ast.literal_eval(json_str)
        for key, value in data.items():
            setattr(self, key, value)



from pathlib import Path
import zarr
from numcodecs import JSON
import dask.dataframe as dd
import pyarrow as pa
import pyarrow.parquet as pq

class Data:
    def __init__(self, zarr_path):
        zarr_path = Path(zarr_path)
        base = zarr_path
        count = 1
        while zarr_path.exists():
            zarr_path = base.with_name(f"{base.stem}_{count}{base.suffix}")
            count += 1
        self._zarr_path = zarr_path
        
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

        for key in self._ds.keys():
            super().__setattr__(key, self._ds[key])

        return self._ds

    def __getattr__(self, name):
        if name in ['_zarr_path', '_ds', '_loaded']:
            return self.__dict__.get(name, None)

        if self._zarr_path is not None and not self._zarr_path.exists():
            self._zarr_path.mkdir(parents=True, exist_ok=True)
            zarr.open(self._zarr_path, mode='w')

        if name in self.dataset.attrs and self.dataset.attrs[name] == 'parquet':
            parquet_path = self._zarr_path / f"{name}.parquet"
            if parquet_path.exists():
                return pd.read_parquet(parquet_path)

        if name in self.dataset.attrs:
            return self.dataset.attrs[name]

        if name in self.dataset.keys():
            result = self.dataset[name]
            if isinstance(result, (zarr.core.Array, np.ndarray)):
                result = result[...].copy()
            return result

        else:
            return None
    
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ['_zarr_path', '_ds', '_loaded']:
            return

        if isinstance(value, da.Array):
            value = value.compute()
            self.dataset[name] = value
        elif isinstance(value, (np.ndarray, zarr.Array)):
            self.dataset[name] = value
        elif isinstance(value, pd.DataFrame):
            # Save as Parquet inside the Zarr directory
            parquet_path = self._zarr_path / f"{name}.parquet"
            table = pa.Table.from_pandas(value)
            pq.write_table(table, parquet_path)
            self.dataset.attrs[name] = 'parquet'
        else:
            # Use Zarr attributes for JSON-serializable values
            self.dataset.attrs[name] = value


    def __setitem__(self, key, value):
        """Handle item assignment for array-like data."""
        # Lazy-load the dataset if not already loaded
        self.dataset  # Ensure the dataset is loaded

        # Check if the key is valid (i.e., a tuple of indices)
        if isinstance(key, tuple):
            # If the key corresponds to a zarr array (like data.array[p, t])
            def __setitem__(self, key, value):
                # Ensure the dataset is loaded
                self.dataset

                if isinstance(key, tuple) and len(key) > 1:
                    arr_name = key[0]
                    arr_index = key[1:]
                    if arr_name not in self.dataset:
                        # Optionally create an empty array if needed
                        pass
                    self.dataset[arr_name][arr_index] = value
                else:
                    raise TypeError(f"Invalid key: {key}. Expected something like data['nuc_masks', p, t].")
        else:
            raise TypeError(f"Invalid key type: {type(key)}. Expected tuple of indices.")

    def append(self, newValues: dict):
        for k, v in newValues.items():
            if v is not None:
                if isinstance(v, pd.DataFrame):
                    # Save as Parquet inside the Zarr directory
                    parquet_path = self._zarr_path / f"{k}.parquet"
                    if parquet_path.exists():
                        # Read the existing DataFrame and concatenate
                        existing_df = getattr(self, k)
                        if existing_df is not None:
                            v = pd.concat([existing_df, v], ignore_index=True)

                setattr(self, k, v)

    def __str__(self):
        return f"Data(zarr_path={self._zarr_path}, loaded={self._loaded}, dataset_keys={list(self.__dict__.keys()) if self._ds else []})"

    def __delattr__(self, name):
        """Delete an attribute or dataset."""
        if name in ['_zarr_path', '_ds', '_loaded']:
            raise AttributeError(f"Cannot delete protected attribute: {name}")

        if name in self.dataset.attrs and self.dataset.attrs[name] == 'parquet':
            # Delete Parquet file if it exists
            parquet_path = self._zarr_path / f"{name}.parquet"
            if parquet_path.exists():
                parquet_path.unlink()
            del self.dataset.attrs[name]
        elif name in self.dataset:
            # Delete Zarr dataset
            del self.dataset[name]
        else:
            raise AttributeError(f"Attribute or dataset '{name}' does not exist.")




if __name__  == '__main__':
    params = Parameters()
    data = Data()

    params.analysis_name = 'analysis_name'

    print(params)
    print(data)

    print(params.__dict__)
    print(data.to_dict())






