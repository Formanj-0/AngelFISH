from collections import UserDict
import json
import os
import shutil

class Receipt(UserDict):
    def __init__(self, analysis_name=None, nas_location=None, data_loader=None, local_location=None, path=None):
        """
        This is a formated dictionary, with a few specific properties
        
        The three key attributes are:
        - arguments:dict - information concerning the pipeline
        - steps:dict - user specified arguments of the steps
        - step_order:list - the order the steps run

        additional attributes
        - dirs: dict of key directories (recalculated on changes to local_location)

        """
        if path is not None:
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            data = {
                'arguments': {
                    'nas_location': None,
                    'local_location': None,
                    'data_loader': None,
                    'analysis_name': None,
                },
                'steps': {},
                'step_order': []
            }

        if nas_location is not None:
            data['arguments']['nas_location'] = nas_location
        if local_location is not None:
            data['arguments']['local_location'] = local_location
        if data_loader is not None:
            data['arguments']['data_loader'] = data_loader
        if analysis_name is not None:
            data['arguments']['analysis_name'] = analysis_name

        if data['arguments']['nas_location']: # and data['arguments']['local_location'] is None: 
            # if nas-location is give we will awlays recalc the locations
            # this makes the nas location dominate 
            database_loc = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            database_loc = os.path.join(database_loc, 'database')
            nas_location = data['arguments']['nas_location']
            name = os.path.basename(nas_location.replace('\\', os.sep).replace('/', os.sep))
            local_location = os.path.join(database_loc, name)

            data['arguments']['local_location'] = local_location
        
        super().__init__(data)
        self.recalc_dirs()

    def __setitem__(self, key, value):
        # Intercept changes to ['arguments']['local_location']
        # correct me if I am wrong, but i think this can only happen after __init__
        if key == 'arguments' and isinstance(value, dict):
            old_local = self['arguments'].get('local_location') if 'arguments' in self else None
            new_local = value.get('local_location')
            super().__setitem__(key, value)
            if old_local != new_local:
                self.recalc_dirs()
        else:
            super().__setitem__(key, value)

    def recalc_dirs(self):
        """
        Recalculate key directories based on current arguments.
        Occurs when ['arguments']['local_location'] is intercepted
        Makes directories consistent with the local_location

        """
        local_location = self['arguments']['local_location']
        analysis_name = self['arguments']['analysis_name']

        dirs = {}
        analysis_dir = os.path.join(local_location, analysis_name)
        dirs['analysis_dir'] = analysis_dir

        results_dir = os.path.join(analysis_dir, 'results')
        dirs['results_dir'] = results_dir

        status_dir = os.path.join(analysis_dir, 'status')
        dirs['status_dir'] = status_dir

        mask_dir = os.path.join(local_location, 'masks')
        dirs['masks_dir'] = mask_dir

        fig_dir = os.path.join(analysis_dir, 'figures')
        dirs['fig_dir'] = fig_dir

        self['dirs'] = dirs
        if os.path.exists(local_location):
            os.makedirs(analysis_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(status_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            os.makedirs(fig_dir, exist_ok=True)

    def save(self, filepath):
        data_to_save = {}
        data_to_save['arguments'] = self.data['arguments']
        data_to_save['steps'] = self.data['steps']
        data_to_save['step_order'] = self.data['step_order']
        
        # os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, indent=4)

    def save_default(self, name, dir:str=None):
        data_to_save = {
                'arguments': {
                    'nas_location': None,
                    'local_location': None,
                    'data_loader': self.arguments['data_loader'],
                    'analysis_name': self.arguments['analysis_name'],
                },
                'steps': {},
                'step_order': []
            }
        
        data_to_save['steps'] = self.data['steps']
        data_to_save['step_order'] = self.data['step_order']
        # stores in default place if dir is none
        path = os.path.join(dir, name) if dir is not None else os.path.abspath(os.path.join(os.path.dirname(__file__), 'default_pipelines', name))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data_to_save, f, indent=4)



    def update_step(self, step_name, key, value):
        if step_name not in self['steps']:
            self['steps'][step_name] = {}
        self['steps'][step_name][key] = value

    def get_step_param(self, step_name, key, default=None):
        return self['steps'].get(step_name, {}).get(key, default)

    @property
    def steps(self):
        return self['steps']

    @steps.setter
    def steps(self, value):
        self['steps'] = value

    @property
    def step_order(self):
        return self['step_order']

    @step_order.setter
    def step_order(self, value):
        self['step_order'] = value

    @property
    def arguments(self):
        return self['arguments']

    @arguments.setter
    def arguments(self, value):
        self['arguments'] = value





