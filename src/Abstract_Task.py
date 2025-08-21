from abc import ABC, abstractmethod
import sciluigi as sl
import os 
import time
import concurrent.futures
import napari
import numpy as np
from magicgui import magicgui
from qtpy.QtCore import QObject

class abstract_task:
    """
    The purpose of this class is to allow for create a general outline of 
    a single step. In doing so this class give a general outline and a few 
    methods that hopefully will allow you to quickly implement new steps

    YOU DO NOT NEED TO FOLLOW THIS EXACTLY ABSTRACT TASK
    YOU DO NEED A FEW THINGS TO OCCUR THO:
    1) every step needs to update the receipt
        - parameters under [steps][step name] (include task name)
        - order as append to [step order]
    2) every step must write results to disk in a interprable way, 
    laid out by the dataloader
    3) it must write a status file, for luigi so that it knows that this task is done

    Other then these 3 limitation all others can be changed and should work downstream

    If you have a weird data structure you need to implement a data loader for this to work.
    For instance if you have a single tif file, I would recommend splitting all the tiffs into their own 
    dir. This must be done for many reasons but mainly, that the goal of this repo is to allow multiple round
    of image processing to be preformed on the same images. So it only makes sense to isolate the tif so that
    they are alone, or treat them as one dataset all together. Make this destinction in the dateloader.

    There are some default directories created by the image processor, these can be found in receipts class.
    This implies that the local location is a directory and not an single image. This goes back to the belief that we 
    want multiple rounds of processing on the same data. 

    Finally, if you want pipelines that depend on other pipelines there are ways to do this. They will be complicated.
    Luigi allows for the creation of tree like dependencies. Additionally, you could build dataloader that take in 
    a directory of datasets. You could also make steps with parameters that specify results from other datasets.
    """
    def __init__(self, receipt, step_name):
        self.receipt = receipt
        self.step_name = step_name
        self.output_path = os.path.join(receipt['dirs']['status_dir'], f'step_{self.step_name}.txt')

        # These steps wont do anthing if the receipt already has the step
        # adds the step name to step order
        if self.step_name not in self.receipt['step_order']:
            self.receipt['step_order'].append(self.step_name)
        
        # makes sure the is a place for params
        if self.step_name not in self.receipt['steps'].keys():
            self.receipt['steps'][self.step_name] = {}

        # makes sure that the task_name is save (you can have multiple tasks of the same task)
        self.receipt['steps'][self.step_name]['task_name'] = self.task_name()

    @classmethod
    @abstractmethod
    def task_name(cls):
        pass

    @property
    @abstractmethod
    def required_keys(self):
        pass

    def gui(self):
        start_time = time.time() 

        # These steps wont do anthing if the receipt already has the step
        # adds the step name to step order
        if self.step_name not in self.receipt['step_order']:
            self.receipt['step_order'].append(self.step_name)
        
        # makes sure the is a place for params
        if self.step_name not in self.receipt['steps'].keys():
            self.receipt['steps'][self.step_name] = {}

        # makes sure that the task_name is save (you can have multiple tasks of the same task)
        self.receipt['steps'][self.step_name]['task_name'] = self.task_name()

        # loads data associated with receipt using data_loader
        self.data = load_data(self.receipt)

        self.viewer = napari.Viewer()
        self.voxel_size_yx = self.data['metadata'](0, 0)['PixelSizeUm']
        self.voxel_size_z = np.abs(self.data['metadata'](0, 0, z=1)['ZPosition_um_Intended'] - self.data['metadata'](0, 0, z=0)['ZPosition_um_Intended'])
        self.viewer.add_image(self.data['images'], name="images", axis_labels=('p', 't', 'c', 'z', 'y', 'x'),
                              scale=[1, 1, 1, self.voxel_size_z/self.voxel_size_yx, 1, 1])

        self.viewer.window.add_dock_widget(self.interface, area='right')

        def on_destroyed(obj=None):
            print('cleaning up')
            self.compress_and_release_memory()
            with open(self.output_path, "a") as f:
                f.write(f"{self.step_name} completed in {time.time() - start_time:.2f} seconds\n")

        self.viewer.window._qt_window.destroyed.connect(on_destroyed)

        return self.receipt

    def write_args_to_receipt(self):
        pass

    def run_process(self):
        pass

    @magicgui(
            call_button='Run'
    )
    def interface(self):
        self.write_args_to_receipt()
        self.preallocate_memmory()
        self.run_process()

    def process(self, 
                new_params:dict = None, 
                p_range = None, 
                t_range = None,
                run_in_parallel:bool = False):

        start_time = time.time() 

        # loads data associated with receipt using data_loader
        self.data = load_data(self.receipt)

        # change parameters at run time
        if new_params:
            for k, v in new_params.items():
                self.receipt['steps'][self.step_name][k] = v

        if self.handle_previous_run():
            # check that required params have been check
            self.check_required_arguments()

            # preallocate need memory in parallel manner
            self.preallocate_memmory()

            # run image processing
            self.iterate_over_data(p_range, t_range, run_in_parallel)

            # compress and release memory
            self.compress_and_release_memory()

        # records completion. This will mark completion for luigi
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "a") as f:
            f.write(f"{self.step_name} completed in {time.time() - start_time:.2f} seconds\n")

        return self.receipt
    
    def iterate_over_data(self, p_range, t_range, run_in_parallel):
        p_values = p_range if p_range is not None else range(self.data['pp'])
        t_values = t_range if t_range is not None else range(self.data['tt'])

        if run_in_parallel:
            def process_single(args):
                p, t = args
                extracted_args = self.extract_args(p, t)
                results = self.image_processing_task(**extracted_args)
                self.write_results(results, p, t)

            with concurrent.futures.ProcessPoolExecutor() as executor:
                executor.map(process_single, [(p, t) for p in p_values for t in t_values])

        else:
            # runs either the specified ranges or all
            for p in p_values:
                for t in t_values:
                    extracted_args = self.extract_args(p,t)
                    results = self.image_processing_function(**extracted_args) # runs image processing
                    self.write_results(results, p, t)

    @staticmethod
    @abstractmethod
    def image_processing_function():
        pass

    def extract_args(self, p, t):
        pass

    def check_required_arguments(self, receipt=None):
        if receipt is None:
            receipt = self.receipt

        for rk in self.required_keys:
            if rk not in receipt['steps'][self.step_name].keys():
                raise KeyError(f'required key {rk} not found for step {self.step_name}')

    # Implement if needed 
    def write_results(self, results, p, t):
        pass

    # Implement if needed 
    def preallocate_memmory(self):
        pass

    # Implement if needed 
    def compress_and_release_memory(self):
        pass

    # Implement if needed 
    def handle_previous_run(self):
        return True


def get_data_loader(name):
    if name == 'pycromanager_data_loader':
        from .Data_Loaders import pycromanager_data_loader
        return pycromanager_data_loader
    elif name == 'recursive_pycromanager_data_loader':
        from .Data_Loaders import recursive_pycromanager_data_loader
        return recursive_pycromanager_data_loader
    raise NotImplementedError(f"Data loader '{name}' is not implemented.")


def load_data(receipt):
    """
    This loads data associated with the receipt 
    """
    data_loader = get_data_loader(receipt['arguments']['data_loader'])
    data = data_loader(receipt)
    return data
















