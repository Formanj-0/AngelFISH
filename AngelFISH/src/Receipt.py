from collections import UserDict
import json

class Receipt(UserDict):
    def __init__(self):
        data = {
            'meta_arguments': {
                'nas_location': None,
                'local_location': None,
                'data_loader': None,
                'analysis_name': None,
            },
            'steps': {}  # Store results/configs for each step
        }
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