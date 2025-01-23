import os
import inspect
import pickle
from typing import Union
from abc import ABC, abstractmethod
from itertools import cycle, islice
import json
import copy

from . import StepClass
from .Parameters import Parameters, Experiment, Settings, ScopeClass, DataContainer




class Pipeline:
    def __init__(self,
                    experiment_location: Union[str, list[str]] = None,
                    parameters = None,
                    steps: Union[list[str]] = None,
                    data_container = None,
                 ) -> None:
        self.experiment_location = experiment_location 
        self.steps = steps
        self.state = 'global'
        Parameters.initialize_parameters_instances()
        self.data_container = DataContainer() if data_container is None else data_container

        if parameters is None:
            self.parameters = Parameters() 
        elif isinstance(parameters, dict):
            self.parameters = Parameters()
            self.parameters.update_parameters(parameters)
        elif isinstance(parameters, Parameters):
            self.parameters = parameters
        else:
            raise Exception('Who knows what you did')
        
        self.get_independent_steps()
        self.get_sequential_steps()
        self.get_finalization_steps()

    def check_requirements(self):
        self.get_step_parameters()

        self.parameters.validate()

        # check if all required parameters are present
        params = self.get_parameters()

        params_to_ignore = ['nuc_mask', 'cell_mask', 'masks', 'images', 'image', 'fov', 'timepoint', 'data_container', 'parameters', 'kwargs']

        # check if all no default parameters are present
        for param in self.no_default_params:
            if param not in params and param not in params_to_ignore:
                raise ValueError(f'{param} is required to run the pipeline')

    def display_all_params(self):
        required_params, all_params = StepClass.get_step_parameters()
        print('Required Parameters: ')
        for param in required_params:
            print(param)
        print('All Parameters: ')
        for param in all_params:
            print(param)

    def execute_independent_steps(self):
        '''
        This method will run through all given prePipeline steps. PrePipeline steps are defined as steps that are ran on a subset of images.
        They can modify the pipelineData, they may also create new properties in pipelineData. they will also have the option to freeze the pipelineData in place

        '''
        from src.GeneralStep import IndependentStepClass
        IndependentStepClass.execute(self.data_container, self.parameters, self.independent_steps)

    def execute_sequential_steps(self):
        from src.GeneralStep import SequentialStepsClass
        SequentialStepsClass.execute(self.data_container, self.parameters, self.sequential_steps)

    def execute_finalization_steps(self):
        from src.GeneralStep import FinalizingStepClass
        FinalizingStepClass.execute(self.data_container, self.parameters, self.finalization_steps)

    def __post_init__(self):
        self.check_requirements()
        self.save_location = self.DataContainer.save_location()
        
        self.parameters.pipeline_init()

    def modify_kwargs(self, modify_kwargs: dict):
        self.parameters.update_parameters(modify_kwargs)

    def get_step_parameters(self):
        self.no_default_params, self.all_params = StepClass.get_step_parameters()

    def get_independent_steps(self):
        if self.state == 'global':
            from src.GeneralStep import IndependentStepClass
            self.independent_steps = copy.copy(IndependentStepClass._instances)
        return self.independent_steps

    def get_sequential_steps(self):
        if self.state == 'global':
            from src.GeneralStep import SequentialStepsClass
            self.sequential_steps = copy.copy(SequentialStepsClass._instances)
        return self.sequential_steps
    
    def get_finalization_steps(self):
        if self.state == 'global':
            from src.GeneralStep import FinalizingStepClass
            self.finalization_steps = copy.copy(FinalizingStepClass._instances)
        return self.finalization_steps

    def get_parameters(self):
        return self.parameters.get_parameters()

    def run(self):
        params = self.get_parameters()
        if self.experiment_location is None:
            self.experiment_location = params['initial_data_location']
        if self.steps is None:
            self.steps = [*[i.__class__.__name__ for i in self.get_independent_steps()._instances], 
                          *[i.__class__.__name__ for i in self.get_sequential_steps()._instances],
                          *[i.__class__.__name__ for i in self.get_finalization_steps()._instances]]
        

        # self.modify_kwargs(params)
        self._run(self.experiment_location , self.steps)

    def run_on_cluster(self, remote_folder, name: str = None):
        self.save_pipeline(name=Settings().name if name is None else name)
        self.send_pipeline_to_cluster(remote_folder)

    def _run(self, locations, steps):
        # save locations and steps
        if steps is not None:
            self.independent_steps, self.sequential_steps, self.finalization_steps = copy.copy(StepClass.initalize_steps_from_list(steps))
            print(self.finalization_steps)
            print(self.sequential_steps)
            print(self.independent_steps)
        if locations is not None:
            self.modify_kwargs({'initial_data_location': locations})
        self.parameters.validate()
        # method to to execute the steps in order
        self.check_requirements()
        print('Running Independent Steps')
        self.execute_independent_steps()
        print('Running Sequential Steps')
        self.execute_sequential_steps()
        print('Running Finalization Steps')
        self.execute_finalization_steps()
        self.clear_pipeline()

    def save_pipeline(self, name: str):
        # save params as a dictionary
        params = self.get_parameters()

        # Remove specified keys from params
        keys_to_remove = ['temp']  # specify the keys to remove
        for key in keys_to_remove:
            params.pop(key, None)

        # save save steps as a dictionary
        if self.steps is None:
            steps = [*[i.__class__.__name__ for i in self.get_independent_steps()], *[i.__class__.__name__ for i in self.get_sequential_steps()],
                    *[i.__class__.__name__ for i in self.get_finalization_steps()]]
        
        # save these as a dictionary
        pipeline = {'params': params, 'steps': steps}

        # save the pipeline as txt file
        file_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        parent_dir = os.path.dirname(file_path)

        pipeline_dir = os.path.join(parent_dir, 'Pipelines')
        
        if not os.path.exists(pipeline_dir):
            os.makedirs(pipeline_dir)
        
        self.pipeline_dictionary_location = os.path.join(pipeline_dir, f'{name}.txt')
        with open(self.pipeline_dictionary_location, 'w') as f:
            json.dump(pipeline, f)

    def send_pipeline_to_cluster(self, remote_folder):
        from .Send_To_Cluster import run_on_cluster
        run_on_cluster(remote_folder , self.pipeline_dictionary_location) 

    def clear_pipeline(self):
        pass
        # Parameters.clear_instances()
        # StepClass.clear_instances()

    def isolate(self):
        self.state = 'local'
        self.data_container = DataContainer()
        self.parameters = Parameters()

        self.parameters.isolate()

        self.get_independent_steps()
        self.get_sequential_steps()
        self.get_finalization_steps()

        Parameters.clear_instances()
        StepClass.clear_instances()
        from src.GeneralStep import FinalizingStepClass
        FinalizingStepClass().clear_instances()
        from src.GeneralStep import SequentialStepsClass
        SequentialStepsClass().clear_instances()
        from src.GeneralStep import IndependentStepClass
        IndependentStepClass().clear_instances()


        
if __name__ == '__main__':
    # get the current file path
    file_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    print (file_path)