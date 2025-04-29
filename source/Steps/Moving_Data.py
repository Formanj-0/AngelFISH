import pathlib
import numpy as np
import shutil
from fpdf import FPDF
import os
import pickle
import trackpy as tp
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import dask.array as da
from datetime import datetime
from abc import abstractmethod

from src.GeneralStep import FinalizingStepClass
from src.Parameters import Parameters, DataContainer
from source.NASConnection import NASConnection



#%% abstract class for moving data
class Moving_Data(FinalizingStepClass):
    @abstractmethod
    def main(self, **kwargs):
        pass

#%% class for moving data to NAS
class return_to_NAS(Moving_Data):
    def main(self, local_dataset_location, initial_data_location, connection_config_location, share_name, log_location, **kwargs):

            if local_dataset_location[0].endswith('.h5'):
                is_h5 = True
                is_ds = False
            else:
                is_h5 = False
                is_ds = True
            
            if is_h5:
                for i, h5_file in enumerate(local_dataset_location):
                    log_file = os.path.splitext(os.path.basename(h5_file))[0]+'.log'
                    with open(log_file, 'a') as log:
                        log.write(f'{datetime.now()}: {h5_file} -> {initial_data_location[i]}\n')
                    NASConnection(connection_config_location, share_name=share_name).write_files_to_NAS(log_file, log_location)
                    NASConnection(connection_config_location, share_name=share_name).write_files_to_NAS(h5_file, initial_data_location[i])
            else:
                Analysis_name = kwargs['analysis_name']
                today = datetime.today()
                date = today.strftime("%Y-%m-%d")
                group_name = f'Analysis_{Analysis_name}_{date}'

                for i, location in enumerate(local_dataset_location):
                    lf = os.path.splitext(os.path.basename(location))[0] + '.log'
                    with open(lf, 'a') as log:
                        log.write(f'{datetime.now()}: {os.path.basename(location)} -> {initial_data_location[i]}\n')
                    NASConnection(connection_config_location, share_name=share_name).write_files_to_NAS(lf, log_location)

                    for item in os.listdir(location):
                        if group_name in item or 'masks' in item:
                            NASConnection(connection_config_location, share_name=share_name).write_folder_to_NAS(os.path.join(location, item), initial_data_location[i])

class remove_local_data(Moving_Data):

    def run(self, p: int = None, t:int = None, data_container = None, parameters = None):
        data_container = DataContainer() if data_container is None else data_container
        parameters = Parameters() if parameters is None else parameters
        kwargs = self.load_in_parameters(p, t, parameters)
        local_dataset_location = kwargs['local_dataset_location']
        temp = kwargs['temp']
        del kwargs
        results = self.main(local_dataset_location, temp) 
        data_container.save_results(results, p, t, parameters)
        data_container.load_temp_data()
        return results

    def main(self, local_dataset_location, temp, **kwargs):
        images = kwargs.get('images')
        del images
        
        for key in list(kwargs.keys()):
            del kwargs[key]

        import gc
        gc.collect()

        for folder in local_dataset_location: # TODO: fix this so that it does not remove the entire directory
            if folder.endswith(".h5"):
                os.remove(folder)
            else:
                shutil.rmtree(os.path.dirname(folder))
        temp.cleanup()

class remove_local_data_but_keep_h5(Moving_Data):
    def main(self, local_dataset_location, **kwargs):
        for folder in local_dataset_location:
            folder = os.path.abspath(folder)
            for file in os.listdir(os.path.dirname(folder)):
                if file.endswith(".h5"):
                    continue
                else:
                    os.remove(os.path.join(os.path.dirname(folder), file))

class remove_all_temp(Moving_Data):
    def main(self, temp, **kwargs):
        temp.cleanup()


class save_copy(Moving_Data):
    def main(self, data_names:list[str], copy_locations:list[str], temp:str, connection_config_location, share_name, **kwargs):
        import random
        if len(data_names) != len(copy_locations):
            raise BaseException('data and locations need to align')
        
        for i, dn in enumerate(data_names):
            data = kwargs[dn]
            if isinstance(data, pd.DataFrame):
                rn = random.choice(range(0,10000))
                csv_path = os.path.join(os.getcwd(), dn +f'{rn}'+ '.csv')
                data.to_csv(csv_path)
                NASConnection(connection_config_location, share_name=share_name).write_files_to_NAS(csv_path, copy_locations[i])
                os.remove(csv_path)
            else:
                raise NotImplemented()



















































# %%
