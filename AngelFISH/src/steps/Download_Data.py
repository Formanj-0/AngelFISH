import os
import pathlib
from src.NASConnection import NASConnection

from src import abstract_task, load_data


class download_data(abstract_task):
    @property
    def task_name(self):
        return 'download_data'
    
    def process(self, 
            receipt, 
            step_name,
            new_params:dict = None, 
            p_range = None, 
            t_range = None, 
            use_gui:bool = False):
        
        data = load_data(receipt)

        if new_params:
            for k, v in new_params.items():
                receipt[step_name][k] = v

        if use_gui:
            pass  # Implement GUI logic here if needed
        else:
            self.image_processing(None, None, receipt, data)

        return receipt

    def image_processing(self, p, t, receipt, data):
        local_location = receipt['meta_arguments'].get('local_location', None)
        nas_location = receipt['meta_arguments'].get('nas_location', None)

        connection_config_location = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        connection_config_location = os.path.join(connection_config_location, 'config_nas.yml')

        # save to a single location
        database_loc = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        database_loc = os.path.join(database_loc, 'database')

        if local_location is None:
            name = os.path.basename(nas_location)
            local_location = os.path.join(database_loc, name)

        # Check if local_location exists
        if os.path.exists(local_location):
            print('local file already exist')
        else:
            # Determine if nas_location is a file or folder
            is_file = any(nas_location.endswith(ext) for ext in ['.h5', '.tif', '.zip', '.log'])
            is_folder = not is_file

            if is_folder:
                self.download_folder_from_NAS(nas_location, local_location, connection_config_location)
            else:
                self.download_file_from_NAS(nas_location, os.path.dirname(local_location), connection_config_location)

        receipt['meta_arguments']['local_location'] = local_location

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