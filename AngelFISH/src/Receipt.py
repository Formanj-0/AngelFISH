from collections import UserDict
import json
import os

class Receipt(UserDict):
    def __init__(self, analysis_name, nas_location, data_loader, local_location=None):
        data = {
            'meta_arguments': {
                'nas_location': nas_location,
                'local_location': local_location,
                'data_loader': data_loader,
                'analysis_name': analysis_name,
            },
            'steps': {},
            'step_order': []
        }

        if data['meta_arguments']['nas_location']: 
            # if nas-location is give we will awlays recalc the locations
            # this gives the nas location dominate 
            database_loc = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            database_loc = os.path.join(database_loc, 'database')

            name = os.path.basename(data['meta_arguments']['nas_location'])
            local_location = os.path.join(database_loc, name)

            data['meta_arguments']['local_location'] = local_location

        # key directories (these will always be re-calculated)
        data['dirs'] = {}
        analysis_name = data['meta_arguments']['analysis_name']
        analysis_dir = os.path.join(local_location, analysis_name)
        data['dirs']['analysis_dir'] = analysis_dir

        results_dir = os.path.join(analysis_dir, 'results')
        data['dirs']['results_dir'] = results_dir

        status_dir = os.path.join(analysis_dir, 'status')
        os.makedirs(status_dir, exist_ok=True)
        data['dirs']['status_dir'] = status_dir

        

        super().__init__(data)



    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=4)

    def update_step(self, step_name, key, value):
        if step_name not in self['steps']:
            self['steps'][step_name] = {}
        self['steps'][step_name][key] = value

    def get_step_param(self, step_name, key, default=None):
        return self['steps'].get(step_name, {}).get(key, default)