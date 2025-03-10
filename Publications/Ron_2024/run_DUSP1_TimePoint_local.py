#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    # Get the number of GPUs available
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Get the name of each GPU
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")

    # Get the current GPU memory usage
    for i in range(num_gpus):
        gpu_memory_allocated = torch.cuda.memory_allocated(i)
        gpu_memory_reserved = torch.cuda.memory_reserved(i)
        print(f"GPU {i} memory allocated: {gpu_memory_allocated / (1024 ** 3):.2f} GB")
        print(f"GPU {i} memory reserved: {gpu_memory_reserved / (1024 ** 3):.2f} GB")
else:
    print("CUDA is not available.")


# In[ ]:


import sys
import os
import pickle
import logging
import numba
import matplotlib
import matplotlib.pyplot as plt
logging.getLogger('matplotlib.font_manager').disabled = True
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

# Add the src directory to sys.path
src_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
print(src_path)
sys.path.append(src_path)

from src.IndependentSteps import Pycromanager2H5, FFF2H5

from src.SequentialSteps import BIGFISH_SpotDetection, SimpleCellposeSegmentaion, IlluminationCorrection, \
                                Calculate_BIGFISH_Threshold, CellProperties, Automatic_BIGFISH_Threshold

from src.FinalizationSteps import Save_Outputs, Save_Images, Save_Parameters, Save_Masks, return_to_NAS, remove_local_data_but_keep_h5, \
                                     remove_local_data

from src.Parameters import Parameters, Experiment, Settings, ScopeClass, DataContainer

from src.Displays import Display

from src.GUI import GUI, StepGUI

from src.Pipeline import Pipeline


# In[ ]:


def runner(idl, ip, name: str = None):
    # Initialize Parameters
    scope = ScopeClass()
    scope.voxel_size_yx = 160
    scope.spot_yx = 300
    scope.spot_z = 500 
    data = DataContainer()
    data.clear_after_error = False
    settings = Settings(name)
    experiment = Experiment()
    experiment.independent_params = ip
    experiment.initial_data_location = idl

    settings.load_in_mask = True
    experiment.FISHChannel = 0
    experiment.nucChannel = 2
    experiment.cytoChannel = 1
    experiment.voxel_size_z = 500

    settings.bigfish_threshold = 'mean'
    settings.verbose = False
    settings.MAX_NUM_IMAGES_TO_AUTOMATICALLY_CALCULATE_THRESHOLD = 1_000_000

    FFF2H5()
    Automatic_BIGFISH_Threshold()
    BIGFISH_SpotDetection()
    CellProperties() 
    Save_Parameters()
    Save_Outputs()
    return_to_NAS()
    remove_local_data()

    pipeline = Pipeline()

    pipeline.run() 

    pipeline.clear_pipeline() # I dont think this is nesscary but its probablily good to do
    


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220225/DUSP1_Dex_0min_20220224' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'D'                   # Replica name. 
time_list  = [0]        # Time of image acquisition
DexConc_list = [0]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R1_0min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R1_0min)
runner(initial_data_location, DUSP1_TS_R1_0min, 'DUSP1_TS_R1_0min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220225/DUSP1_Dex_10min_20220224' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'D'                   # Replica name. 
time_list  = [10]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R1_10min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R1_10min)
runner(initial_data_location, DUSP1_TS_R1_10min, 'DUSP1_TS_R1_10min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220225/DUSP1_Dex_20min_20220224' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'D'                   # Replica name. 
time_list  = [20]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R1_20min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R1_20min)
runner(initial_data_location, DUSP1_TS_R1_20min, 'DUSP1_TS_R1_20min')


# In[ ]:


initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220225/DUSP1_Dex_30min_20220224' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'D'                   # Replica name.
time_list  = [30]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R1_30min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R1_30min)
runner(initial_data_location, DUSP1_TS_R1_30min, 'DUSP1_TS_R1_30min')


# In[ ]:


initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220303/DUSP1_Dex_40min_20220224' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'D'                   # Replica name.
time_list  = [40]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R1_40min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R1_40min)
runner(initial_data_location, DUSP1_TS_R1_40min, 'DUSP1_TS_R1_40min')


# In[ ]:


initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220303/DUSP1_Dex_50min_20220224' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'D'                   # Replica name.
time_list  = [50]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R1_50min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R1_50min)
runner(initial_data_location, DUSP1_TS_R1_50min, 'DUSP1_TS_R1_50min')


# In[ ]:


initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220304/DUSP1_Dex_60min_20220224' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'D'                   # Replica name.
time_list  = [60]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R1_60min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R1_60min)
runner(initial_data_location, DUSP1_TS_R1_60min, 'DUSP1_TS_R1_60min')


# In[ ]:


initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220304/DUSP1_Dex_75min_20220224' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'D'                   # Replica name.
time_list  = [75]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R1_75min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R1_75min)
runner(initial_data_location, DUSP1_TS_R1_75min, 'DUSP1_TS_R1_75min')


# In[ ]:


initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220305/DUSP1_Dex_90min_20220224' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'D'                   # Replica name.
time_list  = [90]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R1_90min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R1_90min)
runner(initial_data_location, DUSP1_TS_R1_90min, 'DUSP1_TS_R1_90min')


# In[ ]:


initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220305/DUSP1_Dex_120min_20220224' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'D'                   # Replica name.
time_list  = [120]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R1_120min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R1_120min)
runner(initial_data_location, DUSP1_TS_R1_120min, 'DUSP1_TS_R1_120min')


# In[ ]:


initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220305/DUSP1_Dex_150min_20220224' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'D'                   # Replica name.
time_list  = [150]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R1_150min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R1_150min)
runner(initial_data_location, DUSP1_TS_R1_150min, 'DUSP1_TS_R1_150min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220305/DUSP1_Dex_180min_20220224' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'D'                   # Replica name. 
time_list  = [180]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R1_180min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R1_180min)
runner(initial_data_location, DUSP1_TS_R1_180min, 'DUSP1_TS_R1_180min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220420/DUSP1_100nM_Dex_R3_20220419_0min' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'E'                   # Replica name. 
time_list  = [0]        # Time of image acquisition
DexConc_list = [0]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R2_0min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R2_0min)
runner(initial_data_location, DUSP1_TS_R2_0min, 'DUSP1_TS_R2_0min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220420/DUSP1_100nM_Dex_R3_20220419_10min' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'E'                   # Replica name. 
time_list  = [10]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R2_10min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R2_10min)
runner(initial_data_location, DUSP1_TS_R2_10min, 'DUSP1_TS_R2_10min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220420/DUSP1_100nM_Dex_R3_20220419_20min' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'E'                   # Replica name. 
time_list  = [20]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R2_20min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R2_20min)
runner(initial_data_location, DUSP1_TS_R2_20min, 'DUSP1_TS_R2_20min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220420/DUSP1_100nM_Dex_R3_20220419_30min' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'E'                   # Replica name. 
time_list  = [30]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R2_30min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R2_30min)
runner(initial_data_location, DUSP1_TS_R2_30min, 'DUSP1_TS_R2_30min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220421/DUSP1_100nM_Dex_R3_20220419_40min' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'E'                   # Replica name. 
time_list  = [40]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R2_40min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R2_40min)
runner(initial_data_location, DUSP1_TS_R2_40min, 'DUSP1_TS_R2_40min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220421/DUSP1_100nM_Dex_R3_20220419_50min' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'E'                   # Replica name. 
time_list  = [50]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R2_50min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R2_50min)
runner(initial_data_location, DUSP1_TS_R2_50min, 'DUSP1_TS_R2_50min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220421/DUSP1_100nM_Dex_R3_20220419_60min' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'E'                   # Replica name. 
time_list  = [60]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R2_60min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R2_60min)
runner(initial_data_location, DUSP1_TS_R2_60min, 'DUSP1_TS_R2_60min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220421/DUSP1_100nM_Dex_R3_20220419_75min' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'E'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R2_75min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R2_75min)
runner(initial_data_location, DUSP1_TS_R2_75min, 'DUSP1_TS_R2_75min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220421/DUSP1_100nM_Dex_R3_20220419_90min' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'E'                   # Replica name. 
time_list  = [90]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R2_90min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R2_90min)
runner(initial_data_location, DUSP1_TS_R2_90min, 'DUSP1_TS_R2_90min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220421/DUSP1_100nM_Dex_R3_20220419_120min' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'E'                   # Replica name. 
time_list  = [120]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R2_120min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R2_120min)
runner(initial_data_location, DUSP1_TS_R2_120min, 'DUSP1_TS_R2_120min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220425/DUSP1_100nM_Dex_R3_20220419_150min' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'E'                   # Replica name. 
time_list  = [150]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R2_150min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R2_150min)
runner(initial_data_location, DUSP1_TS_R2_150min, 'DUSP1_TS_R2_150min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220425/DUSP1_100nM_Dex_R3_20220419_180min' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'E'                   # Replica name. 
time_list  = [180]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R2_180min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R2_180min)
runner(initial_data_location, DUSP1_TS_R2_180min, 'DUSP1_TS_R2_180min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220602/DUSP1_Dex_100nM_0min_NoSpin_052722' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'F'                   # Replica name. 
time_list  = [0]        # Time of image acquisition
DexConc_list = [0]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R3_0min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R3_0min)
runner(initial_data_location, DUSP1_TS_R3_0min, 'DUSP1_TS_R3_0min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220602/DUSP1_Dex_100nM_10min_NoSpin_052722' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'F'                   # Replica name. 
time_list  = [10]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R3_10 = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R3_10)
runner(initial_data_location, DUSP1_TS_R3_10, 'DUSP1_TS_R3_10')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220602/DUSP1_Dex_100nM_20min_NoSpin_052722' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'F'                   # Replica name. 
time_list  = [20]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R3_20min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R3_20min)
runner(initial_data_location, DUSP1_TS_R3_20min, 'DUSP1_TS_R3_20min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220602/DUSP1_Dex_100nM_30min_NoSpin_052722' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'F'                   # Replica name. 
time_list  = [30]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R3_30min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R3_30min)
runner(initial_data_location, DUSP1_TS_R3_30min, 'DUSP1_TS_R3_30min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220602/DUSP1_Dex_100nM_40min_NoSpin_052722' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'F'                   # Replica name. 
time_list  = [40]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R3_40min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R3_40min)
runner(initial_data_location, DUSP1_TS_R3_40min, 'DUSP1_TS_R3_40min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220602/DUSP1_Dex_100nM_50min_NoSpin_052722' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'F'                   # Replica name. 
time_list  = [50]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R3_50min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R3_50min)
runner(initial_data_location, DUSP1_TS_R3_50min, 'DUSP1_TS_R3_50min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220606/DUSP1_Dex_100nM_60min_NoSpin_052722' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'F'                   # Replica name. 
time_list  = [60]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R3_60min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R3_60min)
runner(initial_data_location, DUSP1_TS_R3_60min, 'DUSP1_TS_R3_60min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220606/DUSP1_Dex_100nM_75min_NoSpin_052722' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'F'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R3_75min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R3_75min)
runner(initial_data_location, DUSP1_TS_R3_75min, 'DUSP1_TS_R3_75min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220606/DUSP1_Dex_100nM_90min_NoSpin_052722' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'F'                   # Replica name. 
time_list  = [90]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R3_90min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R3_90min)
runner(initial_data_location, DUSP1_TS_R3_90min, 'DUSP1_TS_R3_90min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220606/DUSP1_Dex_100nM_120min_NoSpin_052722' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'F'                   # Replica name. 
time_list  = [120]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R3_120min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R3_120min)
runner(initial_data_location, DUSP1_TS_R3_120min, 'DUSP1_TS_R3_120min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220606/DUSP1_Dex_100nM_150min_NoSpin_052722' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'F'                   # Replica name. 
time_list  = [150]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R3_150min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R3_150min)
runner(initial_data_location, DUSP1_TS_R3_150min, 'DUSP1_TS_R3_150min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220606/DUSP1_Dex_100nM_180min_NoSpin_052722' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'F'                   # Replica name. 
time_list  = [180]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TS_R3_180min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R3_180min)
runner(initial_data_location, DUSP1_TS_R3_180min, 'DUSP1_TS_R3_180min')


# # Concentration Sweep

# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220628/DUSP1_conc_sweep_0min_060322' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'G'                   # Replica name. 
time_list  = [0]        # Time of image acquisition
DexConc_list = [0]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R1_0min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R1_0min)
runner(initial_data_location, DUSP1_CS_R1_0min, 'DUSP1_CS_R1_0min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220707/DUSP1_conc_sweep_1pM_75min_060322'  )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'G'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [0.001]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R1_1pM = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R1_1pM)
runner(initial_data_location, DUSP1_CS_R1_1pM, 'DUSP1_CS_R1_1pM')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220707/DUSP1_conc_sweep_10pM_75min_060322' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'G'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [0.01]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R1_10pM = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R1_10pM)
runner(initial_data_location, DUSP1_CS_R1_10pM, 'DUSP1_CS_R1_10pM')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220707/DUSP1_conc_sweep_100pM_75min_060322' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'G'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [0.1]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R1_100pM = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R1_100pM)
runner(initial_data_location, DUSP1_CS_R1_100pM, 'DUSP1_CS_R1_100pM')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220705/DUSP1_conc_sweep_1nM_75min_060322' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'G'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [1]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R1_1nM = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R1_1nM)
runner(initial_data_location, DUSP1_CS_R1_1nM, 'DUSP1_CS_R1_1nM')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220705/DUSP1_conc_sweep_10nM_75min_060322' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'G'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [10]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R1_10nM = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R1_10nM)
runner(initial_data_location, DUSP1_CS_R1_10nM, 'DUSP1_CS_R1_10nM')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220705/DUSP1_conc_sweep_100nM_75min_060322' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'G'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R1_100nM = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R1_100nM)
runner(initial_data_location, DUSP1_CS_R1_100nM, 'DUSP1_CS_R1_100nM')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220705/DUSP1_conc_sweep_1uM_75min_060322' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'G'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [1000]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R1_1uM = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R1_1uM)
runner(initial_data_location, DUSP1_CS_R1_1uM, 'DUSP1_CS_R1_1uM')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220628/DUSP1_conc_sweep_10uM_75min_060322')
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'G'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [10000]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R1_10uM = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R1_10uM)
runner(initial_data_location, DUSP1_CS_R1_10uM, 'DUSP1_CS_R1_10uM')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220718/DUSP1_conc_sweep_R2_0min_071422' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'H'                   # Replica name. 
time_list  = [0]        # Time of image acquisition
DexConc_list = [0]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R2_0min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R2_0min)
runner(initial_data_location, DUSP1_CS_R2_0min, 'DUSP1_CS_R2_0min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220721/DUSP1_conc_sweep_R2_1pM_75min_071422' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'H'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [0.001]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R2_1pM = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R2_1pM)
runner(initial_data_location, DUSP1_CS_R2_1pM, 'DUSP1_CS_R2_1pM')


# In[ ]:


# List of directories to process
initial_data_location=(  
        'smFISH_images/Eric_smFISH_images/20220721/DUSP1_conc_sweep_R2_10pM_75min_071422' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'H'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [0.01]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R2_10pM = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R2_10pM)
runner(initial_data_location, DUSP1_CS_R2_10pM, 'DUSP1_CS_R2_10pM')


# In[ ]:


initial_data_location=(   
        'smFISH_images/Eric_smFISH_images/20220721/DUSP1_conc_sweep_R2_100pM_75min_071422' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'H'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [0.1]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R2_100pM = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R2_100pM)
runner(initial_data_location, DUSP1_CS_R2_100pM, 'DUSP1_CS_R2_100pM')


# In[ ]:


initial_data_location=(    
        'smFISH_images/Eric_smFISH_images/20220718/DUSP1_conc_sweep_R2_1nM_75min_071422' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'H'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [1]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R2_1nM = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R2_1nM)
runner(initial_data_location, DUSP1_CS_R2_1nM, 'DUSP1_CS_R2_1nM')


# In[ ]:


initial_data_location=(    
        'smFISH_images/Eric_smFISH_images/20220718/DUSP1_conc_sweep_R2_10nM_75min_071422' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'H'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [10]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R2_10nM = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R2_10nM)
runner(initial_data_location, DUSP1_CS_R2_10nM, 'DUSP1_CS_R2_10nM')


# In[ ]:


initial_data_location=(    
        'smFISH_images/Eric_smFISH_images/20220718/DUSP1_conc_sweep_R2_1uM_75min_071422' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'H'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [1000]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R2_1uM = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R2_1uM)
runner(initial_data_location, DUSP1_CS_R2_1uM, 'DUSP1_CS_R2_1uM')


# In[ ]:


initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220718/DUSP1_conc_sweep_R2_10uM_75min_071422' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'H'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [10000]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R2_10uM = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R2_10uM)
runner(initial_data_location, DUSP1_CS_R2_10uM, 'DUSP1_CS_R2_10uM')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221003/DUSP1_Dex_Sweep_0nM_0min_Control_092022')
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'I'                   # Replica name. 
time_list  = [0]        # Time of image acquisition
DexConc_list = [0]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R3_0min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R3_0min)
runner(initial_data_location, DUSP1_CS_R3_0min, 'DUSP1_CS_R3_0min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221003/DUSP1_Dex_Sweep_1pM_75min_092022' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'I'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [0.001]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R3_1pM = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R3_1pM)
runner(initial_data_location, DUSP1_CS_R3_1pM, 'DUSP1_CS_R3_1pM')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221003/DUSP1_Dex_Sweep_10pM_75min_092022' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'I'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [0.01]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R3_10pM = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R3_10pM)
runner(initial_data_location, DUSP1_CS_R3_10pM, 'DUSP1_CS_R3_10pM')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221003/DUSP1_Dex_Sweep_100pM_75min_092022' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'I'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [0.1]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R3_100pm = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R3_100pm)
runner(initial_data_location, DUSP1_CS_R3_100pm, 'DUSP1_CS_R3_100pm')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221006/DUSP1_Dex_Sweep_1nM_75min_092022' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'I'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [1]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R3_1nM = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R3_1nM)
runner(initial_data_location, DUSP1_CS_R3_1nM, 'DUSP1_CS_R3_1nM')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221006/DUSP1_Dex_Sweep_10nM_75min_092022' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'I'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [10]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R3_10nM = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R3_10nM)
runner(initial_data_location, DUSP1_CS_R3_10nM, 'DUSP1_CS_R3_10nM')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221006/DUSP1_Dex_Sweep_100nM_75min_092022' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'I'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R3_100nM = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R3_100nM)
runner(initial_data_location, DUSP1_CS_R3_100nM, 'DUSP1_CS_R3_100nM')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221011/DUSP1_Dex_Sweep_1uM_75min_092022' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'I'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [1000]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R3_1uM = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R3_1uM)
runner(initial_data_location, DUSP1_CS_R3_1uM, 'DUSP1_CS_R3_1uM')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221011/DUSP1_Dex_Sweep_10uM_75min_092022' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'I'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [10000]        # Dex concentration

# Creating the list of dictionaries
DUSP1_CS_R3_10uM = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_CS_R3_10uM)
runner(initial_data_location, DUSP1_CS_R3_10uM, 'DUSP1_CS_R3_10uM')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230306/DUSP1_0nM_Dex_0min_012623' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'J'                   # Replica name. 
time_list  = [0]        # Time of image acquisition
DexConc_list = [0]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R1_0min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R1_0min)
runner(initial_data_location, DUSP1_TCS_R1_0min, 'DUSP1_TCS_R1_0min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230306/DUSP1_300pM_Dex_30min_012623' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'J'                   # Replica name. 
time_list  = [30]        # Time of image acquisition
DexConc_list = [0.3]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R1_300pM_30min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R1_300pM_30min)
runner(initial_data_location, DUSP1_TCS_R1_300pM_30min, 'DUSP1_TCS_R1_300pM_30min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230306/DUSP1_300pM_Dex_50min_012623' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'J'                   # Replica name. 
time_list  = [50]        # Time of image acquisition
DexConc_list = [0.3]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R1_300pM_50min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R1_300pM_50min)
runner(initial_data_location, DUSP1_TCS_R1_300pM_50min, 'DUSP1_TCS_R1_300pM_50min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230306/DUSP1_300pM_Dex_75min_012623' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'J'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [0.3]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R1_300pM_75min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R1_300pM_75min)
runner(initial_data_location, DUSP1_TCS_R1_300pM_75min, 'DUSP1_TCS_R1_300pM_75min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230309/DUSP1_300pM_Dex_90min_012623' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'J'                   # Replica name. 
time_list  = [90]        # Time of image acquisition
DexConc_list = [0.3]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R1_300pM_90min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R1_300pM_90min)
runner(initial_data_location, DUSP1_TCS_R1_300pM_90min, 'DUSP1_TCS_R1_300pM_90min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230309/DUSP1_300pM_Dex_120min_012623' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'J'                   # Replica name. 
time_list  = [120]        # Time of image acquisition
DexConc_list = [0.3]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R1_300pM_120min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R1_300pM_120min)
runner(initial_data_location, DUSP1_TCS_R1_300pM_120min, 'DUSP1_TCS_R1_300pM_120min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230309/DUSP1_300pM_Dex_180min_012623' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'J'                   # Replica name. 
time_list  = [180]        # Time of image acquisition
DexConc_list = [0.3]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R1_300pM_180min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R1_300pM_180min)
runner(initial_data_location, DUSP1_TCS_R1_300pM_180min, 'DUSP1_TCS_R1_300pM_180min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230309/DUSP1_1nM_Dex_30min_012623' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'J'                   # Replica name. 
time_list  = [30]        # Time of image acquisition
DexConc_list = [1]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R1_1nM_30min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R1_1nM_30min)
runner(initial_data_location, DUSP1_TCS_R1_1nM_30min, 'DUSP1_TCS_R1_1nM_30min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230310/DUSP1_1nM_Dex_50min_012623' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'J'                   # Replica name. 
time_list  = [50]        # Time of image acquisition
DexConc_list = [1]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R1 = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R1)


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230310/DUSP1_1nM_Dex_75min_012623' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'J'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [1]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R1_1nM_75min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R1_1nM_75min)
runner(initial_data_location, DUSP1_TCS_R1_1nM_75min, 'DUSP1_TCS_R1_1nM_75min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230310/DUSP1_1nM_Dex_90min_012623' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'J'                   # Replica name. 
time_list  = [90]        # Time of image acquisition
DexConc_list = [1]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R1_1nM_90min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R1_1nM_90min)
runner(initial_data_location, DUSP1_TCS_R1_1nM_90min, 'DUSP1_TCS_R1_1nM_90min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230310/DUSP1_1nM_Dex_120min_012623' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'J'                   # Replica name. 
time_list  = [120]        # Time of image acquisition
DexConc_list = [1]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R1_1nM_120min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R1_1nM_120min)
runner(initial_data_location, DUSP1_TCS_R1_1nM_120min, 'DUSP1_TCS_R1_1nM_120min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230316/DUSP1_1nM_Dex_180min_012623' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'J'                   # Replica name. 
time_list  = [180]        # Time of image acquisition
DexConc_list = [1]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R1_1nM_180min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R1_1nM_180min)
runner(initial_data_location, DUSP1_TCS_R1_1nM_180min, 'DUSP1_TCS_R1_1nM_180min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230316/DUSP1_10nM_Dex_30min_012623' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'J'                   # Replica name. 
time_list  = [30]        # Time of image acquisition
DexConc_list = [10]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R1_10nM_30min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R1_10nM_30min)
runner(initial_data_location, DUSP1_TCS_R1_10nM_30min, 'DUSP1_TCS_R1_10nM_30min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230316/DUSP1_10nM_Dex_50min_012623' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'J'                   # Replica name. 
time_list  = [50]        # Time of image acquisition
DexConc_list = [10]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R1_10nM_50min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R1_10nM_50min)
runner(initial_data_location, DUSP1_TCS_R1_10nM_50min, 'DUSP1_TCS_R1_10nM_50min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230316/DUSP1_10nM_Dex_75min_012623' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'J'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [10]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R1_10nM_75min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R1_10nM_75min)
runner(initial_data_location, DUSP1_TCS_R1_10nM_75min, 'DUSP1_TCS_R1_10nM_75min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230316/DUSP1_10nM_Dex_90min_012623' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'J'                   # Replica name. 
time_list  = [90]        # Time of image acquisition
DexConc_list = [10]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R1_10nM_90min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R1_10nM_90min)
runner(initial_data_location, DUSP1_TCS_R1_10nM_90min, 'DUSP1_TCS_R1_10nM_90min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230316/DUSP1_10nM_Dex_120min_012623' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'J'                   # Replica name. 
time_list  = [120]        # Time of image acquisition
DexConc_list = [10]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R1_10nM_120min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R1_10nM_120min)
runner(initial_data_location, DUSP1_TCS_R1_10nM_120min, 'DUSP1_TCS_R1_10nM_120min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230316/DUSP1_10nM_Dex_180min_012623' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' 
Replica = 'J'                   # Replica name. 
time_list  = [180]        # Time of image acquisition
DexConc_list = [10]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R1_10nM_180min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R1_10nM_180min)
runner(initial_data_location, DUSP1_TCS_R1_10nM_180min, 'DUSP1_TCS_R1_10nM_180min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230413/DUSP1_DexTimeConcSweep_0min_041223' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'K'                   # Replica name. 
time_list  = [0]        # Time of image acquisition
DexConc_list = [0]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R2_0min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R2_0min)
runner(initial_data_location, DUSP1_TCS_R2_0min, 'DUSP1_TCS_R2_0min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230413/DUSP1_DexTimeConcSweep_300pM_30min_041223' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'K'                   # Replica name. 
time_list  = [30]        # Time of image acquisition
DexConc_list = [0.3]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R2_300pM_30min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R2_300pM_30min)
runner(initial_data_location, DUSP1_TCS_R2_300pM_30min, 'DUSP1_TCS_R2_300pM_30min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230413/DUSP1_DexTimeConcSweep_300pM_50min_041223')
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'K'                   # Replica name. 
time_list  = [50]        # Time of image acquisition
DexConc_list = [0.3]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R2_300pM_50min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R2_300pM_50min)
runner(initial_data_location, DUSP1_TCS_R2_300pM_50min, 'DUSP1_TCS_R2_300pM_50min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230413/DUSP1_DexTimeConcSweep_300pM_75min_041223' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'K'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [0.3]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R2_300pM_75min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R2_300pM_75min)
runner(initial_data_location, DUSP1_TCS_R2_300pM_75min, 'DUSP1_TCS_R2_300pM_75min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230413/DUSP1_DexTimeConcSweep_300pM_90min_041223' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'K'                   # Replica name. 
time_list  = [90]        # Time of image acquisition
DexConc_list = [0.3]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R2_300pM_90min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R2_300pM_90min)
runner(initial_data_location, DUSP1_TCS_R2_300pM_90min, 'DUSP1_TCS_R2_300pM_90min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230424/DUSP1_DexTimeConcSweep_300pM_120min_041223' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'K'                   # Replica name. 
time_list  = [120]        # Time of image acquisition
DexConc_list = [0.3]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R2_300pM_120min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R2_300pM_120min)
runner(initial_data_location, DUSP1_TCS_R2_300pM_120min, 'DUSP1_TCS_R2_300pM_120min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230424/DUSP1_DexTimeConcSweep_300pM_180min_041223' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'K'                   # Replica name. 
time_list  = [180]        # Time of image acquisition
DexConc_list = [0.3]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R2_300pM_180min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R2_300pM_180min)
runner(initial_data_location, DUSP1_TCS_R2_300pM_180min, 'DUSP1_TCS_R2_300pM_180min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230424/DUSP1_DexTimeConcSweep_1nM_30min_041223' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'K'                   # Replica name. 
time_list  = [30]        # Time of image acquisition
DexConc_list = [1]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R2_1nM_30min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R2_1nM_30min)
runner(initial_data_location, DUSP1_TCS_R2_1nM_30min, 'DUSP1_TCS_R2_1nM_30min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230424/DUSP1_DexTimeConcSweep_1nM_50min_041223' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'K'                   # Replica name. 
time_list  = [50]        # Time of image acquisition
DexConc_list = [1]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R2_1nM_50min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R2_1nM_50min)
runner(initial_data_location, DUSP1_TCS_R2_1nM_50min, 'DUSP1_TCS_R2_1nM_50min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230424/DUSP1_DexTimeConcSweep_1nM_75min_041223' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'K'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [1]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R2_1nM_75min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R2_1nM_75min)
runner(initial_data_location, DUSP1_TCS_R2_1nM_75min, 'DUSP1_TCS_R2_1nM_75min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230424/DUSP1_DexTimeConcSweep_1nM_90min_041223' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'K'                   # Replica name. 
time_list  = [90]        # Time of image acquisition
DexConc_list = [1]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R2_1nM_90min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R2_1nM_90min)
runner(initial_data_location, DUSP1_TCS_R2_1nM_90min, 'DUSP1_TCS_R2_1nM_90min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230427/DUSP1_DexTimeConcSweep_1nM_120min_041223' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'K'                   # Replica name. 
time_list  = [120]        # Time of image acquisition
DexConc_list = [1]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R2_1nM_120min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R2_1nM_120min)
runner(initial_data_location, DUSP1_TCS_R2_1nM_120min, 'DUSP1_TCS_R2_1nM_120min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230427/DUSP1_DexTimeConcSweep_1nM_180min_041223' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'K'                   # Replica name. 
time_list  = [180]        # Time of image acquisition
DexConc_list = [1]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R2_1nM_180min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R2_1nM_180min)
runner(initial_data_location, DUSP1_TCS_R2_1nM_180min, 'DUSP1_TCS_R2_1nM_180min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230427/DUSP1_DexTimeConcSweep_10nM_30min_041223' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'K'                   # Replica name. 
time_list  = [30]        # Time of image acquisition
DexConc_list = [10]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R2_10nM_30min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R2_10nM_30min)
runner(initial_data_location, DUSP1_TCS_R2_10nM_30min, 'DUSP1_TCS_R2_10nM_30min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230427/DUSP1_DexTimeConcSweep_10nM_50min_041223' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'K'                   # Replica name. 
time_list  = [50]        # Time of image acquisition
DexConc_list = [10]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R2_10nM_50min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R2_10nM_50min)
runner(initial_data_location, DUSP1_TCS_R2_10nM_50min, 'DUSP1_TCS_R2_10nM_50min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230511/DUSP1_DexTimeConcSweep_10nM_75min_041223' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'K'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [10]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R2_10nM_75min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R2_10nM_75min)
runner(initial_data_location, DUSP1_TCS_R2_10nM_75min, 'DUSP1_TCS_R2_10nM_75min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230511/DUSP1_DexTimeConcSweep_10nM_90min_041223')
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'K'                   # Replica name. 
time_list  = [90]        # Time of image acquisition
DexConc_list = [10]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R2_10nM_90min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R2_10nM_90min)
runner(initial_data_location, DUSP1_TCS_R2_10nM_90min, 'DUSP1_TCS_R2_10nM_90min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230511/DUSP1_DexTimeConcSweep_10nM_120min_041223' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'K'                   # Replica name. 
time_list  = [120]        # Time of image acquisition
DexConc_list = [10]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R2_10nM_120min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R2_10nM_120min)
runner(initial_data_location, DUSP1_TCS_R2_10nM_120min, 'DUSP1_TCS_R2_10nM_120min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230511/DUSP1_DexTimeConcSweep_10nM_180min_041223' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'K'                   # Replica name. 
time_list  = [180]        # Time of image acquisition
DexConc_list = [10]        # Dex concentration

# Creating the list of dictionaries
DUSP1_TCS_R2_10nM_180min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R2_10nM_180min)
runner(initial_data_location, DUSP1_TCS_R2_10nM_180min, 'DUSP1_TCS_R2_10nM_180min')


# In[ ]:





# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230530/DUSP1_Dex_time_conc_sweep_0min_050223_R3' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'L'                   # Replica name. 
time_list  = [0]        # Time of image acquisition
DexConc_list = [0]        # Dex concentration

DUSP1_TCS_R3_0min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R3_0min)
runner(initial_data_location, DUSP1_TCS_R3_0min, 'DUSP1_TCS_R3_0min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230530/DUSP1_Dex_300pM_30min_050223_R3' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'L'                   # Replica name. 
time_list  = [30]        # Time of image acquisition
DexConc_list = [0.3]        # Dex concentration

DUSP1_TCS_R3_300pM_30min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R3_300pM_30min)
runner(initial_data_location, DUSP1_TCS_R3_300pM_30min, 'DUSP1_TCS_R3_300pM_30min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230530/DUSP1_Dex_300pM_50min_050223_R3' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'L'                   # Replica name. 
time_list  = [50]        # Time of image acquisition
DexConc_list = [0.3]        # Dex concentration

DUSP1_TCS_R3_300pM_50min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R3_300pM_50min)
runner(initial_data_location, DUSP1_TCS_R3_300pM_50min, 'DUSP1_TCS_R3_300pM_50min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230605/DUSP1_Dex_300pM_75min_050223_R3')
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'L'                   # Replica name. 
time_list  = [0, 30, 50, 75, 90, 120, 180, 30, 50, 90, 120, 180, 30, 50, 75, 90, 120, 180]        # Time of image acquisition
DexConc_list = [0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 10]        # Dex concentration

DUSP1_TCS_R3_300pM_75min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R3_300pM_75min)
runner(initial_data_location, DUSP1_TCS_R3_300pM_75min, 'DUSP1_TCS_R3_300pM_75min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230605/DUSP1_Dex_300pM_90min_050223_R3' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'L'                   # Replica name. 
time_list  = [90]        # Time of image acquisition
DexConc_list = [0.3]        # Dex concentration

DUSP1_TCS_R3_300pM_90min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R3_300pM_90min)
runner(initial_data_location, DUSP1_TCS_R3_300pM_90min, 'DUSP1_TCS_R3_300pM_90min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230605/DUSP1_Dex_300pM_120min_050223_R3' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'L'                   # Replica name. 
time_list  = [120]        # Time of image acquisition
DexConc_list = [0.3]        # Dex concentration

DUSP1_TCS_R3_300pM_120min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R3_300pM_120min)
runner(initial_data_location, DUSP1_TCS_R3_300pM_120min, 'DUSP1_TCS_R3_300pM_120min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230605/DUSP1_Dex_300pM_180min_050223_R3' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'L'                   # Replica name. 
time_list  = [180]        # Time of image acquisition
DexConc_list = [0.3]        # Dex concentration

DUSP1_TCS_R3_300pM_180min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R3_300pM_180min)
runner(initial_data_location, DUSP1_TCS_R3_300pM_180min, 'DUSP1_TCS_R3_300pM_180min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230605/DUSP1_Dex_1nM_30min_050223_R3' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'L'                   # Replica name. 
time_list  = [30]        # Time of image acquisition
DexConc_list = [1]        # Dex concentration

DUSP1_TCS_R3_300pM_30min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R3_300pM_30min)
runner(initial_data_location, DUSP1_TCS_R3_300pM_30min, 'DUSP1_TCS_R3_300pM_30min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230605/DUSP1_Dex_1nM_50min_050223_R3' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'L'                   # Replica name. 
time_list  = [50]        # Time of image acquisition
DexConc_list = [1]        # Dex concentration

DUSP1_TCS_R3_1nM_50min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R3_1nM_50min)
runner(initial_data_location, DUSP1_TCS_R3_1nM_50min, 'DUSP1_TCS_R3_1nM_50min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230608/DUSP1_Dex_1nM_90min_050223_R3' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'L'                   # Replica name. 
time_list  = [90]        # Time of image acquisition
DexConc_list = [1]        # Dex concentration

DUSP1_TCS_R3_1nM_90min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R3_1nM_90min)
runner(initial_data_location, DUSP1_TCS_R3_1nM_90min, 'DUSP1_TCS_R3_1nM_90min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230608/DUSP1_Dex_1nM_120min_050223_R3' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'L'                   # Replica name. 
time_list  = [120]        # Time of image acquisition
DexConc_list = [1]        # Dex concentration

DUSP1_TCS_R3_1nM_120min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R3_1nM_120min)
runner(initial_data_location, DUSP1_TCS_R3_1nM_120min, 'DUSP1_TCS_R3_1nM_120min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230608/DUSP1_Dex_1nM_180min_050223_R3' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'L'                   # Replica name. 
time_list  = [180]        # Time of image acquisition
DexConc_list = [1]        # Dex concentration

DUSP1_TCS_R3_1nM_180min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R3_1nM_180min)
runner(initial_data_location, DUSP1_TCS_R3_1nM_180min, 'DUSP1_TCS_R3_1nM_180min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230614/DUSP1_Dex_10nM_30min_050223_R3_redo' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'L'                   # Replica name. 
time_list  = [30]        # Time of image acquisition
DexConc_list = [10]        # Dex concentration

DUSP1_TCS_R3_10nM_30min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R3_10nM_30min)
runner(initial_data_location, DUSP1_TCS_R3_10nM_30min, 'DUSP1_TCS_R3_10nM_30min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230608/DUSP1_Dex_10nM_50min_050223_R3' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'L'                   # Replica name. 
time_list  = [50]        # Time of image acquisition
DexConc_list = [10]        # Dex concentration

DUSP1_TCS_R3_10nM_50min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R3_10nM_50min)
runner(initial_data_location, DUSP1_TCS_R3_10nM_50min, 'DUSP1_TCS_R3_10nM_50min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230608/DUSP1_Dex_10nM_75min_050223_R3' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'L'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [10]        # Dex concentration

DUSP1_TCS_R3_10nM_75min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R3_10nM_75min)
runner(initial_data_location, DUSP1_TCS_R3_10nM_75min, 'DUSP1_TCS_R3_10nM_75min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230608/DUSP1_Dex_10nM_90min_050223_R3' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'L'                   # Replica name. 
time_list  = [90]        # Time of image acquisition
DexConc_list = [10]        # Dex concentration

DUSP1_TCS_R3_10nM_90min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R3_10nM_90min)
runner(initial_data_location, DUSP1_TCS_R3_10nM_90min, 'DUSP1_TCS_R3_10nM_90min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230608/DUSP1_Dex_10nM_120min_050223_R3' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'L'                   # Replica name. 
time_list  = [120]        # Time of image acquisition
DexConc_list = [10]        # Dex concentration

DUSP1_TCS_R3_10nM_120min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R3_10nM_120min)
runner(initial_data_location, DUSP1_TCS_R3_10nM_120min, 'DUSP1_TCS_R3_10nM_120min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20230608/DUSP1_Dex_10nM_180min_050223_R3' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'L'                   # Replica name. 
time_list  = [180]        # Time of image acquisition
DexConc_list = [10]        # Dex concentration

DUSP1_TCS_R3_10nM_180min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TCS_R3_10nM_180min)
runner(initial_data_location, DUSP1_TCS_R3_10nM_180min, 'DUSP1_TCS_R3_10nM_180min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220725/DUSP1_Dex_100nM_6hr_0min_072022' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'M'                   # Replica name. 
time_list  = [0]        # Time of image acquisition
DexConc_list = [0]        # Dex concentration

DUSP1_TS_R4_0min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R4_0min)
runner(initial_data_location, DUSP1_TS_R4_0min, 'DUSP1_TS_R4_0min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220725/DUSP1_Dex_100nM_6hr_150min_072022' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'M'                   # Replica name. 
time_list  = [150]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

DUSP1_TS_R4_150min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R4_150min)
runner(initial_data_location, DUSP1_TS_R4_150min, 'DUSP1_TS_R4_150min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220725/DUSP1_Dex_100nM_6hr_180min_072022' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'M'                   # Replica name. 
time_list  = [180]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

DUSP1_TS_R4_180min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R4_180min)
runner(initial_data_location, DUSP1_TS_R4_180min, 'DUSP1_TS_R4_180min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220927/DUSP1_100nM_Dex_0min_081822' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'N'                   # Replica name. 
time_list  = [0]        # Time of image acquisition
DexConc_list = [0]        # Dex concentration

DUSP1_TS_R5_0min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R5_0min)
runner(initial_data_location, DUSP1_TS_R5_0min, 'DUSP1_TS_R5_0min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220927/DUSP1_100nM_Dex_30min_081822' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'N'                   # Replica name. 
time_list  = [30]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

DUSP1_TS_R5_30min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R5_30min)
runner(initial_data_location, DUSP1_TS_R5_30min, 'DUSP1_TS_R5_30min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220927/DUSP1_100nM_Dex_60min_081822' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'N'                   # Replica name. 
time_list  = [60]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

DUSP1_TS_R5_60min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R5_60min)
runner(initial_data_location, DUSP1_TS_R5_60min, 'DUSP1_TS_R5_60min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220928/DUSP1_100nM_Dex_90min_081822' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'N'                   # Replica name. 
time_list  = [90]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

DUSP1_TS_R5_90min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R5_90min)
runner(initial_data_location, DUSP1_TS_R5_90min, 'DUSP1_TS_R5_90min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220928/DUSP1_100nM_Dex_120min_081822' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'N'                   # Replica name. 
time_list  = [120]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

DUSP1_TS_R5_120min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R5_120min)
runner(initial_data_location, DUSP1_TS_R5_120min, 'DUSP1_TS_R5_120min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220928/DUSP1_100nM_Dex_150min_081822' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'N'                   # Replica name. 
time_list  = [150]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

DUSP1_TS_R5_150min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R5_150min)
runner(initial_data_location, DUSP1_TS_R5_150min, 'DUSP1_TS_R5_150min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20220929/DUSP1_100nM_Dex_180min_081822' )
# Parameters
Condition = 'DUSP1_timesweep'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep'
Replica = 'N'                   # Replica name. 
time_list  = [180]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration

DUSP1_TS_R5_180min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc
    }
    for time, DexConc in zip(time_list, DexConc_list)
]
# Output the result
print(DUSP1_TS_R5_180min)
runner(initial_data_location, DUSP1_TS_R5_180min, 'DUSP1_TS_R5_180min')


# ## ```DUSP1_TPL```  experiments

# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221019/DUSP1_Dex_TPL_3hr_0min_Control_101422' )
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'O'                   # Replica name. 
time_list  = [0]        # Time of image acquisition
DexConc_list = [0]        # Dex concentration
time_TPL = [None]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R1_0minControl = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R1_0minControl)
runner(initial_data_location, DUSP1_TPL_R1_0minControl, 'DUSP1_TPL_R1_0minControl')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221019/DUSP1_Dex100nM_75min_Control_101422' )
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'O'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration
time_TPL = [None]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R1_75minControl = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R1_75minControl)
runner(initial_data_location, DUSP1_TPL_R1_75minControl, 'DUSP1_TPL_R1_75minControl')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221019/DUSP1_Dex100nM_75min_TPL5uM_10min_101422' )
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'O'                   # Replica name. 
time_list  = [85]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration
time_TPL = [75]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R1_75min_tpl10min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R1_75min_tpl10min)
runner(initial_data_location, DUSP1_TPL_R1_75min_tpl10min)


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221019/DUSP1_Dex100nM_75min_TPL5uM_30min_101422' )
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'O'                   # Replica name. 
time_list  = [105]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration
time_TPL = [75]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R1_75min_tpl30min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R1_75min_tpl30min)
runner(initial_data_location, DUSP1_TPL_R1_75min_tpl30min)


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221019/DUSP1_Dex100nM_75min_TPL5uM_60min_101422' )
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'O'                   # Replica name. 
time_list  = [135]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration
time_TPL = [75]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R1_75min_tpl60min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R1_75min_tpl60min)
runner(initial_data_location, DUSP1_TPL_R1_75min_tpl60min)


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221020/DUSP1_Dex100nM_150min_Control_101422' )
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'O'                   # Replica name. 
time_list  = [150]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration
time_TPL = [None]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R1_150minControl = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R1_150minControl)
runner(initial_data_location, DUSP1_TPL_R1_150minControl)


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221020/DUSP1_Dex100nM_150min_TPL5uM_10min_101422' )
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'O'                   # Replica name. 
time_list  = [160]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration
time_TPL = [150]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R1_150min_tpl10min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R1_150min_tpl10min)
runner(initial_data_location, DUSP1_TPL_R1_150min_tpl10min)


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221020/DUSP1_Dex100nM_150min_TPL5uM_30min_101422' )
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'O'                   # Replica name. 
time_list  = [180]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration
time_TPL = [150]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R1_150min_tpl30min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R1_150min_tpl30min)
runner(initial_data_location, DUSP1_TPL_R1_150min_tpl30min)


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221020/DUSP1_Dex100nM_150min_TPL5uM_60min_101422' )
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'O'                   # Replica name. 
time_list  = [210]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration
time_TPL = [150]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R1_150min_tpl60min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R1_150min_tpl60min)
runner(initial_data_location, DUSP1_TPL_R1_150min_tpl60min, 'DUSP1_TPL_R1_150min_tpl60min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221103/DUSP1_Dex_0min_TPL_0min_110222' )
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'P'                   # Replica name. 
time_list  = [0]        # Time of image acquisition
DexConc_list = [0]        # Dex concentration
time_TPL = [None]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R2_0minControl = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R2_0minControl)
runner(initial_data_location, DUSP1_TPL_R2_0minControl, 'DUSP1_TPL_R2_0minControl')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221103/DUSP1_Dex_0min_TPL_15min_110222' )
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'P'                   # Replica name. 
time_list  = [15]        # Time of image acquisition
DexConc_list = [0]        # Dex concentration
time_TPL = [0]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R2_0min_tpl15min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R2_0min_tpl15min)
runner(initial_data_location, DUSP1_TPL_R2_0min_tpl15min, 'DUSP1_TPL_R2_0min_tpl15min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221103/DUSP1_Dex_0min_TPL_30min_110222' )
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'P'                   # Replica name. 
time_list  = [30]        # Time of image acquisition
DexConc_list = [0]        # Dex concentration
time_TPL = [0]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R2_0min_tpl30min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R2_0min_tpl30min)
runner(initial_data_location, DUSP1_TPL_R2_0min_tpl30min, 'DUSP1_TPL_R2_0min_tpl30min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221103/DUSP1_Dex_0min_TPL_60min_110222')
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'P'                   # Replica name. 
time_list  = [60]        # Time of image acquisition
DexConc_list = [0]        # Dex concentration
time_TPL = [0]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R2_0min_tpl60min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R2_0min_tpl60min)
runner(initial_data_location, DUSP1_TPL_R2_0min_tpl60min, 'DUSP1_TPL_R2_0min_tpl60min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221107/DUSP1_Dex_20min_TPL_0min_110222' )
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'P'                   # Replica name. 
time_list  = [20]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration
time_TPL = [None]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R2_20minControl = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R2_20minControl)
runner(initial_data_location, DUSP1_TPL_R2_20minControl, 'DUSP1_TPL_R2_20minControl')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221107/DUSP1_Dex_20min_TPL_15min_110222' )
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'P'                   # Replica name. 
time_list  = [35]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration
time_TPL = [20]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R2_20min_tpl15min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R2_20min_tpl15min)
runner(initial_data_location, DUSP1_TPL_R2_20min_tpl15min, 'DUSP1_TPL_R2_20min_tpl15min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221107/DUSP1_Dex_20min_TPL_30min_110222' )
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'P'                   # Replica name. 
time_list  = [50]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration
time_TPL = [20]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R2_20min_tpl30min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R2_20min_tpl30min)
runner(initial_data_location, DUSP1_TPL_R2_20min_tpl30min, 'DUSP1_TPL_R2_20min_tpl30min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221107/DUSP1_Dex_20min_TPL_60min_110222' )
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'P'                   # Replica name. 
time_list  = [80]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration
time_TPL = [20]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R2_20min_tpl60min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R2_20min_tpl60min)
runner(initial_data_location, DUSP1_TPL_R2_20min_tpl60min, 'DUSP1_TPL_R2_20min_tpl60min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221107/DUSP1_Dex_75min_TPL_0min_110222' )
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'P'                   # Replica name. 
time_list  = [75]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration
time_TPL = [None]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R2_75minControl = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R2_75minControl)
runner(initial_data_location, DUSP1_TPL_R2_75minControl, 'DUSP1_TPL_R2_75minControl')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221107/DUSP1_Dex_75min_TPL_15min_110222' )
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'P'                   # Replica name. 
time_list  = [90]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration
time_TPL = [75]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R2_75min_tpl15min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R2_75min_tpl15min)
runner(initial_data_location, DUSP1_TPL_R2_75min_tpl15min, 'DUSP1_TPL_R2_75min_tpl15min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221107/DUSP1_Dex_75min_TPL_30min_110222' )
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'P'                   # Replica name. 
time_list  = [105]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration
time_TPL = [75]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R2_75min_tpl30min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R2_75min_tpl30min)
runner(initial_data_location, DUSP1_TPL_R2_75min_tpl30min, 'DUSP1_TPL_R2_75min_tpl30min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221107/DUSP1_Dex_75min_TPL_60min_110222' )
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'P'                   # Replica name. 
time_list  = [135]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration
time_TPL = [75]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R2_75min_tpl60min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R2_75min_tpl60min)
runner(initial_data_location, DUSP1_TPL_R2_75min_tpl60min, 'DUSP1_TPL_R2_75min_tpl60min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221109/DUSP1_Dex_180min_TPL_0min_110222' )
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'P'                   # Replica name. 
time_list  = [180]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration
time_TPL = [None]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R2_180minControl = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R2_180minControl)
runner(initial_data_location, DUSP1_TPL_R2_180minControl, 'DUSP1_TPL_R2_180minControl')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221129/DUSP1_Dex_180min_TPL_15min_110222' )
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'P'                   # Replica name. 
time_list  = [195]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration
time_TPL = [180]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R2_180min_tpl15min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R2_180min_tpl15min)
runner(initial_data_location, DUSP1_TPL_R2_180min_tpl15min, 'DUSP1_TPL_R2_180min_tpl15min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221129/DUSP1_Dex_180min_TPL_30min_110222' )
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'P'                   # Replica name. 
time_list  = [210]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration
time_TPL = [180]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R2_180min_tpl30min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R2_180min_tpl30min)
runner(initial_data_location, DUSP1_TPL_R2_180min_tpl30min, 'DUSP1_TPL_R2_180min_tpl30min')


# In[ ]:


# List of directories to process
initial_data_location=(
        'smFISH_images/Eric_smFISH_images/20221129/DUSP1_Dex_180min_TPL_60min_110222' )
# Parameters
Condition = 'DUSP1_TPL'      # Experimental condition. The options are:  'GR_timesweep' , 'DUSP1_timesweep' , 'DUSP1_TPL'
Replica = 'P'                   # Replica name. 
time_list  = [240]        # Time of image acquisition
DexConc_list = [100]        # Dex concentration
time_TPL = [180]  # Time of TPL addition relitive to Dex timepoint

DUSP1_TPL_R2_180min_tpl60min = [
    {
        "condition": Condition,
        "replica": Replica,
        "time": time,
        "Dex_Conc": DexConc,
        "time_tpl": tpl_time
    }
    for time, DexConc, tpl_time in zip(time_list, DexConc_list, time_TPL)
]
# Output the result
print(DUSP1_TPL_R2_180min_tpl60min)
runner(initial_data_location, DUSP1_TPL_R2_180min_tpl60min, 'DUSP1_TPL_R2_180min_tpl60min')

