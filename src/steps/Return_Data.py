import os
from datetime import datetime
from src.NASConnection import NASConnection
import time

from src import abstract_task

class return_data(abstract_task):

    @classmethod
    def task_name(cls):
        return 'return_data'
    
    def process(self):
        start_time = time.time() 

        analysis_name = self.receipt['meta_arguments']['analysis_name']
        analysis_dir = self.receipt['dirs']['analysis_dir']
        masks_dir = self.receipt['dirs']['masks_dir']
        nas_dir = self.receipt['meta_arguments']['nas_location']
        results_dir = self.receipt['dirs']['results_dir']
        status_dir = self.receipt['dirs']['status_dir']

        connection_config_location = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        connection_config_location = os.path.join(connection_config_location, 'config_nas.yml')
        NASConnection(connection_config_location).write_folder_to_NAS(masks_dir, nas_dir)
        NASConnection(connection_config_location).write_folder_to_NAS(analysis_dir, nas_dir)
        NASConnection(connection_config_location).write_folder_to_NAS(results_dir, os.path.join(nas_dir, analysis_name))
        NASConnection(connection_config_location).write_folder_to_NAS(status_dir, os.path.join(nas_dir, analysis_name))

        # records completion. This will mark completion for luigi
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "a") as f:
            f.write(f"{self.step_name} completed in {time.time() - start_time:.2f} seconds\n")

        return self.receipt


    @staticmethod
    def image_processing_function():
        pass



























