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
        figure_dir = self.receipt['dirs']['fig_dir']

        connection_config_location = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        connection_config_location = os.path.join(connection_config_location, 'config_nas.yml')
        NASConnection(connection_config_location).write_folder_to_NAS(masks_dir, nas_dir)
        NASConnection(connection_config_location).write_folder_to_NAS(analysis_dir, nas_dir)
        NASConnection(connection_config_location).write_folder_to_NAS(figure_dir, os.path.join(nas_dir, analysis_name))
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




class copy_results(abstract_task):
    @classmethod
    def task_name(cls):
        return 'copy_results'
    
    def process(self):
        start_time = time.time() 
        results_dir = self.receipt['dirs']['results_dir']
        nas_location = self.receipt['steps'][self.step_name]['nas_location']
        result_name = self.receipt['steps'][self.step_name]['result_name']

        connection_config_location = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        connection_config_location = os.path.join(connection_config_location, 'config_nas.yml')
        NASConnection(connection_config_location).write_files_to_NAS(os.path.join(results_dir, result_name), nas_location)

        # records completion. This will mark completion for luigi
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "a") as f:
            f.write(f"{self.step_name} completed in {time.time() - start_time:.2f} seconds\n")

        return self.receipt

    @staticmethod
    def image_processing_function():
        pass






















