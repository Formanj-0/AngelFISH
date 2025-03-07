from pydantic import BaseModel, validator, field_validator
from typing import Any, Dict, List, Optional, Union
from numpydantic import NDArray
import dask.array as da
from pycromanager import Dataset
import numpy as np
import luigi
from abc import ABC, abstractmethod
import os
import json

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
    image_path: Optional[str] = None
    mask_path: Optional[str] = None
    clear_after_error: bool = True
    name: Optional[str] = None
    NUMBER_OF_CORES: int = 4
    num_images_to_run: int = np.inf
    connection_config_location: str = 'c:\\Users\\formanj\\GitHub\\AngelFISH\\config_nas.yml'
    display_plots: bool = True
    load_in_mask: bool = False
    mask_structure: Optional[Any] = None
    order: str = 'pt'

    share_name: str = 'share'

    # @field_validator('images', 'masks', mode='before')
    # def load_array_from_location(cls, v, values):
    #     if v is None:
    #         return v
    #     # Check if the input is a path (str or list of paths)
    #     if isinstance(v, (str, list)):
    #         # Assuming that we can use Dataset to load data via the `as_array` method
    #         if isinstance(v, str):
    #             dataset = Dataset(v)
    #             array = dataset.as_array()
    #         else:
    #             # Handle the case for a list of paths
    #             arrays = []
    #             for path in v:
    #                 dataset = Dataset(path)
    #                 arrays.append(dataset.as_array())
    #             array = da.concatenate(arrays, axis=0)
    #         # Convert the array to a Dask array if needed (depending on your data handling logic)
    #         return array  # or da.asarray(array) if array is already numpy
    #     return v  # If it's already an NDArray, return as is

    class Config:
        extra = 'allow'
        use_enum_values = True

class DataSingle(BaseModel):
    p: int
    t: int
    nas_location: str
    history: Optional[List[str]]
    image_path: Optional[str] = None
    mask_path: Optional[str] = None
    images: Optional[NDArray] = da.empty((0, 0, 0))
    masks: Optional[NDArray] = da.empty((0, 0, 0))
    image: Optional[NDArray] = da.empty((0, 0, 0))
    mask: Optional[NDArray] = da.empty((0, 0, 0))
    independent_params: List[Dict[str, Any]] = [{}]
    history: Optional[List[str]]

    class Config:
        extra = 'allow'

class DataBulk(BaseModel):
    images: Optional[NDArray] = da.empty((0, 0, 0))
    masks: Optional[NDArray] = da.empty((0, 0, 0))
    independent_params: List[Dict[str, Any]] = [{}]
    history: Optional[List[str]]

    class Config:
        extra = 'allow'


class ImageProcessor(luigi.Task, ABC):
    input_data: Any
    output_data: Any
    step_name: str = "base"

    def requires(self):
        """Allow for the chaining of tasks. Returns dependencies if needed."""
        return []  # Return other tasks that need to run first

    def output(self):
        """Define the output of this task."""
        return luigi.LocalTarget(f"{self.step_name}_output.json")

    @abstractmethod
    def eval(self, **kwargs):
        """Define the actual processing logic."""
        pass

    def run(self):
        """Run the processing task."""
        results = self.eval(self.input_data)
        with self.output().open('w') as f:
            json.dump(results, f)


class SingleImageProcessingTask(ImageProcessor):
    param_path: str
    in_path: str
    modify_images = False
    modify_masks = False

    def requires(self):
        return 

    def output(self):
        stepName = self.__class__.__name__
        p = self.data.p
        t = self.data.t
        dataName = os.path.basename(self.data.nas_location)
        cache_key = f'{stepName}-p_{p}-t_{t}-{dataName}.json'
        return luigi.LocalTarget(cache_key)

    def run(self):
        if self.output().exists():
            raise 'Result already exist'
        
        print(f'Processing {os.path.basename(self.data.nas_location)}, {self.data.p}, {self.data.t} with {self.__class__.__name__}')

        results = self.eval(**self.params)

        with self.output().open('w') as f:
            if not self.modify_images and not self.modify_masks:
                json.dump(results.model_dump_json(round_trip=True, excule=['image', 'images', 'masks', 'mask']), f)
            elif not self.modify_images and self.modify_masks :
                json.dump(results.model_dump_json(round_trip=True, excule=['masks', 'mask']), f)
            elif self.modify_images and not self.modify_masks :
                json.dump(results.model_dump_json(round_trip=True, excule=['image', 'images']), f)
            else:
                json.dump(results.model_dump_json(round_trip=True), f)

    @abstractmethod
    def eval(self):
        raise NotImplementedError


class BulkImageProcessingTask(ImageProcessor):
    def requires(self):
        pass

    def output(self):
        pass

    def run(self):
        pass

    @abstractmethod
    def eval(self):
        pass


class ImageProcessingPipeline(luigi.Task):
    params: Parameters
    steps: List[str]

    def requires(self):
        """Dynamically create a list of tasks based on the steps."""
        previous_task = None
        
        for step in self.steps:
            # Dynamically look up the class for the step
            step_class = globals().get(f"{step.capitalize()}Step")
            if step_class:
                # If this is the first step, pass the input data, otherwise, pass the output of the previous task
                task = step_class(input_data=previous_task.output() if previous_task else self.input_data)
                previous_task = task  # The current task will be the previous task for the next iteration
            else:
                raise ValueError(f"Unknown step: {step}")
        
        return previous_task  # Return the last task in the sequence, it will be the final step

    def output(self):
        """The final output of the pipeline."""
        return luigi.LocalTarget("final_output.json")

    def run(self):
        """This is where the final result is collected and written to disk."""
        with self.output().open('w') as f:
            f.write("Final result of configurable pipeline")


class Splitter(luigi.task):
    def requires(self):
        pass

    def output(self):
        pass

    def run(self):
        pass

    @abstractmethod
    def eval(self):
        pass

class Merger(luigi.task):
    def requires(self):
        pass

    def output(self):
        pass

    def run(self):
        pass

    @abstractmethod
    def eval(self):
        pass

























