from pydantic import BaseModel, field_validator
from typing import Any, Dict, List, Optional, Union
from numpydantic import NDArray
import dask.array as da
from pycromanager import Dataset
import numpy as np
import luigi
from abc import ABC, abstractmethod
import os
import json

def convert_to_pydantic_safe(value):
    if type(value) is np.int32:
        return int(value)


#%% Parameters
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
    images: Optional[NDArray] = da.ones((0, 0, 0))
    masks: Optional[NDArray] = da.ones((0, 0, 0))
    image_path: Optional[str] = None
    mask_path: Optional[str] = None
    clear_after_error: bool = True
    name: Optional[str] = None
    NUMBER_OF_CORES: int = 4
    num_images_to_run: int = 100000000
    connection_config_location: str = 'c:\\Users\\formanj\\GitHub\\AngelFISH\\config_nas.yml'
    display_plots: bool = True
    load_in_mask: bool = False
    mask_structure: Optional[Any] = None
    order: str = 'pt'

    share_name: str = 'share'

    class Config:
        extra = 'allow'
        use_enum_values = True


class Data(BaseModel, ABC):
    nas_location: str
    history: Optional[List[str]]
    image_path: Optional[str] = None
    mask_path: Optional[str] = None
    images: Optional[NDArray] = da.empty((0, 0, 0))
    masks: Optional[NDArray] = da.empty((0, 0, 0))
    independent_params: List[Dict[str, Any]] = [{}]

    class Config:
        extra = 'allow'
        use_enum_values = True

    def append_dict(self, data: dict):
        for k, v in data.items():
            setattr(self, k, convert_to_pydantic_safe(v))


class DataSingle(Data):
    p: int
    t: int
    image: Optional[NDArray] = da.empty((0, 0, 0))
    mask: Optional[NDArray] = da.empty((0, 0, 0))
    independent_params: List[Dict[str, Any]] = [{}]


class DataBulk(Data):
    images: Optional[NDArray] = da.empty((0, 0, 0))
    masks: Optional[NDArray] = da.empty((0, 0, 0))
    
    def split(p, t) -> DataSingle:
        pass

    def merge(data: list):
        pass






#%% Image Processing Tasks
class ImageProcessor(luigi.Task, ABC):
    input_path: Any
    param_path: Any
    step_name: str = "base"
    modify_images = False
    modify_masks = False
    previous_task: luigi.Task
    nas_location: str

    def requires(self):
        """Allow for the chaining of tasks. Returns dependencies if needed."""
        return self.previous_task() if self.previous_task is not None else None# Return other tasks that need to run first

    def output(self):
        """Define the output of this task."""
        return luigi.LocalTarget(f"{self.step_name}_output.json")

    @abstractmethod
    def eval(self, **kwargs):
        """Define the actual processing logic."""
        pass

    def run(self):
        """Run the processing task."""
        print(f'Processing {os.path.basename(self.nas_location)} with {self.step_name}')

        if self.output().exists():
            print('Results already exist')
    
        with open(self.param_path, 'r') as json_file: # load in params
            params = Parameters.model_validate_json(json.load(json_file))

        with open(self.input_path, 'r') as json_file: # load in data
            data = Data.model_validate_json(json.load(json_file))
        kwargs = {**params.model_dump(), **data.model_dump()}

        results = self.eval(**kwargs)

        data.append_dict(results)
    
        with self.output().open('w') as f:
            json.dump(data.model_dump_json(round_trip=True), f)

class SingleImageProcessingTask(ImageProcessor):
    p: int
    t: int

    def output(self):
        dataName = os.path.basename(self.nas_location)
        cache_key = f'{self.step_name}-p_{self.p}-t_{self.t}-{dataName}.json'
        return luigi.LocalTarget(cache_key)

    def run(self):
        print(f'Processing {os.path.basename(self.nas_location)}, {self.p}, {self.t} with {self.step_name}')

        if self.output().exists():
            raise 'Result already exist'

        if self.output().exists():
            print('Results already exist')
    
        with open(self.param_path, 'r') as json_file: # load in params
            params = Parameters.model_validate_json(json.load(json_file))

        with open(self.input_path, 'r') as json_file: # load in data
            data = DataSingle.model_validate_json(json.load(json_file))
    
        kwargs = {**params.model_dump(), **data.model_dump()}

        results = self.eval(**kwargs)

        data.append_dict(results)

        with self.output().open('w') as f:
            if not self.modify_images and not self.modify_masks:
                json.dump(data.model_dump_json(round_trip=True, exclude=['image', 'images', 'masks', 'mask']), f)
            elif not self.modify_images and self.modify_masks :
                json.dump(data.model_dump_json(round_trip=True, exclude=['masks', 'mask']), f)
            elif self.modify_images and not self.modify_masks :
                json.dump(data.model_dump_json(round_trip=True, exclude=['image', 'images']), f)
            else:
                json.dump(data.model_dump_json(round_trip=True), f)

    @abstractmethod
    def eval(self):
        raise NotImplementedError


class BulkImageProcessingTask(ImageProcessor):
    def output(self):
        dataName = os.path.basename(self.nas_location)
        cache_key = f'{self.step_name}-{dataName}.json'
        return luigi.LocalTarget(cache_key)

    def run(self):
        print(f'Processing {os.path.basename(self.nas_location)} with {self.step_name}')

        if self.output().exists():
            print('Results already exist')
    
        with open(self.param_path, 'r') as json_file: # load in params
            params = Parameters.model_validate_json(json.load(json_file))

        with open(self.input_path, 'r') as json_file: # load in data
            data = DataBulk.model_validate_json(json.load(json_file))
    
        kwargs = {**params.model_dump(), **data.model_dump()}

        results = self.eval(**kwargs)

        data.append_dict(results)
    
        with self.output().open('w') as f:
            if not self.modify_images and not self.modify_masks:
                json.dump(data.model_dump_json(round_trip=True, exclude=['image', 'images', 'masks', 'mask']), f)
            elif not self.modify_images and self.modify_masks :
                json.dump(data.model_dump_json(round_trip=True, exclude=['masks', 'mask']), f)
            elif self.modify_images and not self.modify_masks :
                json.dump(data.model_dump_json(round_trip=True, exclude=['image', 'images']), f)
            else:
                json.dump(data.model_dump_json(round_trip=True), f)

    @abstractmethod
    def eval(self):
        pass



#%% Logical tasks
class Splitter(luigi.Task):
    input_path: str
    nas_location: str
    previous_task: Any
    positions: List[int]
    time_points: List[int]

    def requires(self):
        return self.previous_task()
    
    def output(self):
        dataName = os.path.basename(self.nas_location)
        return {p: {t: luigi.LocalTarget(f'Split-p_{p}-t_{t}-{dataName}.json') for t in self.time_points} for p in self.positions}

    def run(self): # convert DataBulk -> {DataSingles}
        with open(self.input().path, 'r') as json_file: # load in params
            bulkData = DataBulk.model_validate_json(json.load(json_file))
        
        for p in self.positions:
            for t in self.time_points:
                singleData = bulkData.split(p=p, t=t)
                with self.output()[p][t].open('w') as f:
                    json.dump(singleData.model_dump_json(round_trip=True), f)

class Merger(luigi.Task):
    nas_location: str
    previous_task: Any
    positions: List[int]
    time_points: List[int]
    
    def requires(self):
        return {p: {t: self.previous_task(p=p, t=t) for t in self.time_points} for p in self.positions}

    def output(self):
        """Define the output for the merged result."""
        data_name = os.path.basename(self.nas_location)
        return luigi.LocalTarget(f'Merged-{data_name}.json')

    def run(self):
        """Merge all the split data."""
        merged_data = []
        for p in self.positions:
            for t in self.time_points:
                with self.input()[p][t].open('r') as f:
                    single_data = DataSingle.model_validate_json(json.load(f))
                    merged_data.append(single_data)
        
        bulk_data = DataBulk.merge(merged_data)
        
        with self.output().open('w') as f:
            json.dump(bulk_data.model_dump_json(round_trip=True), f)

#%% Workflows
class SingleWorkflow(luigi.WrapperTask):
    steps: List[Any]
    positions: List[int]
    time_points: List[int]
    
    def requires(self):
        """For each position and time point, run the processing steps."""
        tasks = []
        for step in self.steps:
            for p in self.positions:
                for t in self.time_points:
                    # Each time-point/position combination will require specific tasks
                    task = step(p=p, t=t, previous_task=Splitter, nas_location=self.nas_location)
                    tasks.append(task)
        return tasks

class BulkWorkflow(luigi.WrapperTask):
    steps: List[Any]
    nas_location: str
    input_path: str

    def requires(self):
        tasks = []
        for step in self.steps:
            task = step(input_path=self.input_path, param_path=self.param_path, nas_location=self.nas_location, previous_task=Merger)
            tasks.append(task)
        return tasks

class Workflow(luigi.WrapperTask):
    params: Parameters
    steps: List[Any]
    positions: List[int]
    time_points: List[int]
    nas_location: str
    input_path: str
    param_path: str
    
    def requires(self):
        return [
            SingleWorkflow(steps=self.steps, positions=self.positions, time_points=self.time_points, nas_location=self.nas_location),
            BulkWorkflow(steps=self.steps, nas_location=self.nas_location, input_path=self.input_path, param_path=self.param_path)
        ]

    def output(self):
        """The final output of the pipeline."""
        return luigi.LocalTarget("final_output.json")

    def run(self):
        """This is where the final result is collected and written to disk."""
        with self.output().open('w') as f:
            f.write("Final result of configurable pipeline")