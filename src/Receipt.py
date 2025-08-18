from collections import UserDict
import json
import os
import shutil

class Receipt(UserDict):
    def __init__(self, analysis_name=None, nas_location=None, data_loader=None, local_location=None, path=None):
    
        if path is not None:
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            data = {
                'meta_arguments': {
                    'nas_location': None,
                    'local_location': None,
                    'data_loader': None,
                    'analysis_name': None,
                },
                'steps': {},
                'step_order': []
            }

        if nas_location is not None:
            data['meta_arguments']['nas_location'] = nas_location
        if local_location is not None:
            data['meta_arguments']['local_location'] = local_location
        if data_loader is not None:
            data['meta_arguments']['data_loader'] = data_loader
        if analysis_name is not None:
            data['meta_arguments']['analysis_name'] = analysis_name

        if data['meta_arguments']['nas_location'] and data['meta_arguments']['local_location'] is None: 
            # if nas-location is give we will awlays recalc the locations
            # this gives the nas location dominate 
            database_loc = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            database_loc = os.path.join(database_loc, 'database')

            name = os.path.basename(data['meta_arguments']['nas_location'])
            local_location = os.path.join(database_loc, name)

            data['meta_arguments']['local_location'] = local_location
        
        super().__init__(data)
        self.recalc_dirs()

    def __setitem__(self, key, value):
        # Intercept changes to meta_arguments['local_location']
        if key == 'meta_arguments' and isinstance(value, dict):
            old_local = self['meta_arguments'].get('local_location') if 'meta_arguments' in self else None
            new_local = value.get('local_location')
            super().__setitem__(key, value)
            if old_local != new_local:
                self.recalc_dirs()
        else:
            super().__setitem__(key, value)

    def recalc_dirs(self):
        """Recalculate key directories based on current meta_arguments."""
        local_location = self['meta_arguments']['local_location']
        analysis_name = self['meta_arguments']['analysis_name']

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
        data_to_save['meta_arguments'] = self.data['meta_arguments']
        data_to_save['steps'] = self.data['steps']
        data_to_save['step_order'] = self.data['step_order']
        
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, indent=4)

    def update_step(self, step_name, key, value):
        if step_name not in self['steps']:
            self['steps'][step_name] = {}
        self['steps'][step_name][key] = value

    def get_step_param(self, step_name, key, default=None):
        return self['steps'].get(step_name, {}).get(key, default)







