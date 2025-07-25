import os
import pathlib
from src.NASConnection import NASConnection
import time

from src import abstract_task, load_data


class download_data(abstract_task):
    @property
    def task_name(self):
        return 'download_data'
    
    def process(self):
        
        start_time = time.time()
        
        self.data = load_data(self.receipt)

        self.image_processing()

        if self.step_name not in self.receipt['step_order']:
            self.receipt['step_order'].append(self.step_name)

        if self.step_name not in self.receipt['steps'].keys():
            self.receipt['steps'][self.step_name] = {}
            
        with open(self.output_path, "a") as f:
            f.write(f"{self.step_name} completed in {time.time() - start_time:.2f} seconds\n")

        self.receipt['steps'][self.step_name]['task_name'] = self.task_name 

        return self.receipt

    def image_processing(self):
        local_location = self.receipt['meta_arguments'].get('local_location', None)
        nas_location = self.receipt['meta_arguments'].get('nas_location', None)

        connection_config_location = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        connection_config_location = os.path.join(connection_config_location, 'config_nas.yml')

        # Check if local_location exists
        if local_location and os.path.exists(local_location):
            print('local file already exist')
        else:
            # Determine if nas_location is a file or folder
            is_file = any(nas_location.endswith(ext) for ext in ['.h5', '.tif', '.zip', '.log'])
            is_folder = not is_file

            if is_folder:
                self.download_folder_from_NAS(nas_location, local_location, connection_config_location)
            else:
                self.download_file_from_NAS(nas_location, os.path.dirname(local_location), connection_config_location)

        self.receipt['meta_arguments']['local_location'] = local_location

    def download_folder_from_NAS(self, remote_path, local_folder_path, connection_config_location):
        # Downloads a folder from the NAS, confirms that the it has not already been downloaded
        if not os.path.exists(local_folder_path):
            os.makedirs(local_folder_path, exist_ok=True)
            nas = NASConnection(pathlib.Path(connection_config_location))
            nas.copy_folder(remote_folder_path=pathlib.Path(remote_path), 
                        local_folder_path=local_folder_path)
            
    def download_file_from_NAS(self, remote_path, local_folder_path, connection_config_location):
        # Downloads a folder from the NAS, confirms that the it has not already been downloaded
        if not os.path.exists(os.path.join(local_folder_path, os.path.basename(remote_path))):
            os.makedirs(local_folder_path, exist_ok=True)
            nas = NASConnection(pathlib.Path(connection_config_location))
            nas.download_file(remote_file_path=pathlib.Path(remote_path), 
                        local_folder_path=local_folder_path)