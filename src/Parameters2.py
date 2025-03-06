from pydantic import BaseModel, validator, field_validator
from typing import Any, Dict, List, Optional, Union
from numpydantic import NDArray
import dask.array as da
from pycromanager import Dataset

class Parameters(BaseModel):
    voxel_size_yx: float = 130
    voxel_size_z: float = 500
    spot_z: float = 100
    spot_yx: float = 360
    index_dict: Optional[Dict[str, Any]] = None
    nucChannel: Optional[int] = None
    cytoChannel: Optional[int] = None
    FISHChannel: Optional[int] = None
    independent_params: List[Dict[str, Any]] = [{}]
    timestep_s: Optional[float] = None
    local_dataset_location: Optional[Union[str, List]] = None
    nas_location: Optional[Union[str, List]] = None
    num_images: Optional[int] = None
    images: Optional[NDArray] = da.empty((0, 0, 0))
    masks: Optional[NDArray] = da.empty((0, 0, 0))
    clear_after_error: bool = True
    name: Optional[str] = None
    NUMBER_OF_CORES: int = 4
    save_files: bool = True
    num_chunks_to_run: int = 100000
    connection_config_location: str = 'c:\\Users\\formanj\\GitHub\\AngelFISH\\config_nas.yml'
    display_plots: bool = True
    load_in_mask: bool = False
    mask_structure: Optional[Any] = None
    order: str = 'pt'
    share_name: str = 'share'

    @field_validator('images', 'masks', mode='before')
    def load_array_from_location(cls, v, values):
        if v is None:
            return v
        # Check if the input is a path (str or list of paths)
        if isinstance(v, (str, list)):
            # Assuming that we can use Dataset to load data via the `as_array` method
            if isinstance(v, str):
                dataset = Dataset(v)
                array = dataset.as_array()
            else:
                # Handle the case for a list of paths
                arrays = []
                for path in v:
                    dataset = Dataset(path)
                    arrays.append(dataset.as_array())
                array = da.concatenate(arrays, axis=0)
            # Convert the array to a Dask array if needed (depending on your data handling logic)
            return array  # or da.asarray(array) if array is already numpy
        return v  # If it's already an NDArray, return as is

    class Config:
        extra = 'allow'


