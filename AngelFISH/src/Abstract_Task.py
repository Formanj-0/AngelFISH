from abc import ABC, abstractmethod
import sciluigi as sl
import os 


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
        
    def process(self, 
                new_params:dict = None, 
                p_range = None, 
                t_range = None, 
                use_gui:bool = False):
        
        self.data = load_data(self.receipt)

        if new_params:
            for k, v in new_params.items():
                self.receipt['steps'][self.step_name][k] = v

        if use_gui:
            pass  # Implement GUI logic here if needed
        else:
            for p in p_range if p_range else range(self.data['pp']):
                for t in t_range if t_range else range(self.data['tt']):
                    self.image_processing(p, t)

        return self.receipt

    @abstractmethod
    def image_processing(self, p, t):
        pass

    def get_luigi_task(self, input_task):
        self.data = load_data(self.receipt)
        task = abstract_luigi_task(receipt=self.receipt, step_name=self.step_name, input_path=input_task.out, output_path=self.data)
        task.process = self.process
        return task


def get_data_loader(name):
    if name == 'pycromanager_data_loader':
        from .Data_Loaders import pycromanager_data_loader
        return pycromanager_data_loader
    raise NotImplementedError(f"Data loader '{name}' is not implemented.")


def load_data(receipt):
    data_loader = get_data_loader(receipt['meta_arguments']['data_loader'])
    data = data_loader(receipt)
    return data
















