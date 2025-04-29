# %%
import os
import sys
cwd = os.getcwd()
source_path = os.path.join(cwd, '..')
print(source_path)
sys.path.append(source_path)

import logging
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import numpy as np
import dask.array as da
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from source import Data, Parameters, Pycro, DownloadData, SimpleCellposeSegmentaion, BIGFISH_SpotDetection, CellProperties, Save_Images, Save_Masks, Save_Outputs, Save_Parameters, remove_local_data, return_to_NAS


# %% [markdown]
# # Step 1) Pick a notebook from notebook and modify parameters

# %%
params = Parameters()
data = Data('Demo_Pipeline')
params.display_plots = False

# %%
params.load_in_mask = True
params.initial_data_location = [r"Users\Jack\_Datasets\IntronDiffusion\04032025_TriptolideTimeseries_FISH\JF001_0min_5uM_12"]
params.independent_params = {}

params.local_dataset_location = DownloadData().main(**params.__dict__)['local_dataset_location']
if data.images is None:
    output = Pycro().main(**params.__dict__)
    data.append(output)

pp = data.images.shape[0]
tt = data.images.shape[1]
cc = data.images.shape[2]
zz = data.images.shape[3]
yy = data.images.shape[4]
xx = data.images.shape[5]

print(params.local_dataset_location)

# %%
data.nuc_masks = da.zeros([pp, tt, yy, xx])
params.nucChannel = 2


count = 0
for p in range(pp):
    for t in range(tt):
        if count >= params.num_chunks_to_run:
            break
        kwargs = {**params.__dict__, **data.__dict__}
        kwargs['image'] = data.images[p, t]
        kwargs['fov'] = p
        kwargs['timepoint'] = t
        output = SimpleCellposeSegmentaion().main(**kwargs)
        data.nuc_masks[p,t] = output['nuc_mask']
        count += 1


# %%
params.FISHChannel = [0]
params.bigfish_alpha = 0.99
params.bigfish_beta = 1
params.bigfish_gamma = 5
params.voxel_size_yx = 130
params.voxel_size_z = 500
params.spot_yx = 160
params.spot_z = 1000

count = 0
for p in range(pp):
    for t in range(tt):
        if count >= params.num_chunks_to_run:
            break
        kwargs = {**params.__dict__, **data.__dict__}
        kwargs['image'] = data.images[p, t]
        if data.nuc_masks is not None:
            kwargs['nuc_mask'] = np.array(data.nuc_masks[p,t])
        if data.cell_masks is not None:
            kwargs['cell_mask'] = np.array(data.cell_masks[p, t])
        kwargs['fov'] = p
        kwargs['timepoint'] = t
        output = BIGFISH_SpotDetection().main(**kwargs)
        data.append(output)
        count += 1


# %%
count = 0
for p in range(pp):
    for t in range(tt):
        if count >= params.num_chunks_to_run:
            break
        kwargs = {**params.__dict__, **data.__dict__}
        kwargs['image'] = data.images[p, t]
        if data.nuc_masks is not None:
            kwargs['nuc_mask'] = data.nuc_masks[p, t, :, :]
            kwargs['nuc_mask'] = kwargs['nuc_mask'][np.newaxis, :, :]

        if data.cell_masks is not None:
            kwargs['cell_mask'] = data.cell_masks[p, t, :, :]
            kwargs['cell_mask'] = kwargs['cell_masks'][np.newaxis, :, :]
            
        kwargs['fov'] = p
        kwargs['timepoint'] = t
        output = CellProperties().main(**kwargs)
        data.append(output)
        count += 1

# %%
# Save_Images().main(**{**params.__dict__, **data.__dict__})

Save_Masks().main(**{**params.__dict__, **data.__dict__, 
                    'masks':{'nuc_masks': data.nuc_masks}})

Save_Parameters().main(params, **{**params.__dict__, **data.__dict__})

Save_Outputs().main(data.__dict__, **{**params.__dict__, 'position_indexs': data.position_indexs})


# %%
return_to_NAS().main(**params.__dict__)

# %%
print(params)
print(data)


