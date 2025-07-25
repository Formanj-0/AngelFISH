from abc import ABC, abstractmethod
import sciluigi as sl
import os 
import time

class abstract_luigi_task(sl.Task):
    in_upstream = sl.Parameter()
    out_downstream = sl.Parameter()

    def out(self):
        return(sl.TargetInfo(self, self.out_downstream))

    def run(self):
        self.receipt = self.process(self.receipt)


class abstract_task:
    def __init__(self, receipt, step_name):
        self.receipt = receipt
        self.step_name = step_name
        self.output_path = os.path.join(receipt['dirs']['status_dir'], f'step_{self.step_name}.txt')

    @property
    @abstractmethod
    def task_name(self):
        pass

    @property
    @abstractmethod
    def required_keys(self):
        pass

    def process(self, 
                new_params:dict = None, 
                p_range = None, 
                t_range = None):

        start_time = time.time() 

        # adds the step name to step order
        if self.step_name not in self.receipt['step_order']:
            self.receipt['step_order'].append(self.step_name)
        
        if self.step_name not in self.receipt['steps'].keys():
            self.receipt['steps'][self.step_name] = {}
        
        # loads data associated with receipt using data_loader
        self.data = load_data(self.receipt)

        # change parameters at run time
        if new_params:
            for k, v in new_params.items():
                self.receipt['steps'][self.step_name][k] = v

        # check that required params have been check
        self.check_required_arguments()

        # preallocate need memory in parallel manner
        self.preallocate_memmory()

        # runs either the specified ranges or all
        for p in p_range if p_range else range(self.data['pp']):
            for t in t_range if t_range else range(self.data['tt']):
                self.image_processing_task(p, t) # runs image processing

        # makes sure that the task_name is save (you can have multiple tasks in the same )
        self.receipt['steps'][self.step_name]['task_name'] = self.task_name

        # compress and release memory
        self.compress_and_release_memory()

        # records completion. This will mark completion for luigi
        with open(self.output_path, "a") as f:
            f.write(f"{self.step_name} completed in {time.time() - start_time:.2f} seconds\n")

        return self.receipt

    @abstractmethod
    def image_processing_task(self, p, t):
        pass

    def get_luigi_task(self, input_task):
        self.data = load_data(self.receipt)
        task = abstract_luigi_task(receipt=self.receipt, step_name=self.step_name, input_path=input_task.out, output_path=self.data)
        task.process = self.process
        return task

    def check_required_arguments(self, receipt=None):
        if receipt is None:
            receipt = self.receipt

        for rk in self.required_keys:
            if rk not in receipt['steps'][self.step_name].keys():
                raise KeyError(f'required key {rk} not found for step {self.step_name}')

    # Implement if needed 
    def preallocate_memmory(self):
        pass

    # Implement if needed 
    def compress_and_release_memory(self):
        pass

def get_data_loader(name):
    if name == 'pycromanager_data_loader':
        from .Data_Loaders import pycromanager_data_loader
        return pycromanager_data_loader
    raise NotImplementedError(f"Data loader '{name}' is not implemented.")


def load_data(receipt):
    data_loader = get_data_loader(receipt['meta_arguments']['data_loader'])
    data = data_loader(receipt)
    return data
















