
import os
import pathlib

import numpy as np
import pathlib
import os
import tifffile
import matplotlib.pyplot as plt
from cellpose import models
from skimage.io import imread
import skimage as sk
import bigfish
import bigfish.stack as stack
# import bigfish.segmentation as segmentation
import bigfish.multistack as multistack
import bigfish.plot as plot
import dask.array as da
from abc import ABC, abstractmethod
import tifffile
import pandas as pd
from scipy import ndimage as ndi
from copy import copy
from magicgui import magicgui
import pathlib

# from src import SequentialStepsClass
# from src.Parameters import Parameters

from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt
import zarr
import shutil
import napari

from AngelFISH.src import abstract_task, load_data


class segment(abstract_task):

    @classmethod
    def task_name(cls):
        return 'segment'

    @property
    def required_keys(self):
        return ['mask_name', 'channel']

    def extract_args(self, p, t):
                # get the user defined arguments
        given_args = self.receipt['steps'][self.step_name]

        # gets the data specific to the call
        data_to_send = {}
        zyx_image = self.data['images'][p, t, given_args['channel'], :, :, :].compute()
        data_to_send['zyx_image'] = zyx_image

        all_args = {**given_args, **data_to_send}

        return all_args

    def preallocate_memmory(self):
        if hasattr(self, 'temp_dir') and self.temp_dir is not None and os.path.exists(self.temp_dir):
            # temp_dir exists
            pass
        else:
            self.create_temp_mask()

    def create_temp_mask(self):
        """Create a temporary mask file on disk that can be accessed by multiple processes."""
        if not hasattr(self, 'mask_file'):
            # Get mask name from receipt
            mask_name = self.receipt['steps'].get(self.step_name, {}).get('mask_name', 'temp_mask')

            # Create mask file path
            self.mask_file = os.path.join(self.receipt['dirs']['analysis_dir'], f"{mask_name}.zarr")

            # Get shape from images, removing channel dimension
            image_shape = self.data['images'].shape
            mask_shape = image_shape[:2] + image_shape[3:]  # Remove channel dimension

            # Specify chunk size or use 'auto'
            # chunks = 'auto'

            # Check if mask already exists
            if os.path.exists(self.mask_file):
                # Load existing mask as writable
                self.mask = zarr.open(self.mask_file, mode='r+')
            else:
                # Create new mask initialized with zeros and make it writable
                self.mask = zarr.open(self.mask_file, mode='a', shape=mask_shape, dtype=np.int32)
                self.mask[:] = 0  # Initialize with zeros

        return self.mask

    @staticmethod
    def image_processing_function(zyx_image, 
                                  pretrained_model_name: str = None,  
                                  diameter: float = 180, 
                                  invert: bool = False, 
                                  normalize: bool = True, 
                                  do_3D:bool=False, 
                                  min_size:float=500,
                                  flow_threshold:float=0, 
                                  cellprob_threshold:float=0, 
                                  **kwargs):
        print('running image processing')
        # we will always take in an image with this shape struct
        zz, yy, xx = zyx_image.shape

        model_location = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        pretrained_model = os.path.join(model_location, pretrained_model_name) if pretrained_model_name else None

        model  = models.CellposeModel(gpu=True, pretrained_model=pretrained_model)

        channel_axis = 0
        z_axis = 1

        if not do_3D:
            zyx_image = np.max(zyx_image, axis=0)
            # zyx_image = np.expand_dims(zyx_image, axis=0)
            z_axis = None
            

        zyx_image = np.expand_dims(zyx_image, axis=0) # add fake channel axis

        mask, flows, styles = model.eval(zyx_image,
                                    diameter=diameter, 
                                    invert=invert, 
                                    normalize=normalize, 
                                    channel_axis=channel_axis,
                                    z_axis=z_axis,
                                    do_3D=do_3D,
                                    min_size=min_size, 
                                    flow_threshold=flow_threshold, 
                                    cellprob_threshold=cellprob_threshold)


        if not do_3D and zz > 1: # expand 2d image to 3d 
            new_masks = np.zeros((zz, yy, xx))
            for z in range(zz):
                new_masks[z, :, :] = mask
            mask = new_masks

        return mask, flows, styles

    def compress_and_release_memory(self):
        output_path = os.path.join(self.receipt['dirs']['masks_dir'], f"{self.receipt['steps'][self.step_name]['mask_name']}.tiff")
        if self.receipt['steps'][self.step_name]['mask_name'] in self.data.keys():
            self.data[self.receipt['steps'][self.step_name]['mask_name']] = self.mask.astype(np.uint16)
        else:
            tifffile.imwrite(output_path, self.mask.astype(np.uint16))
        self.remove_temp_mask()

    def remove_temp_mask(self):
        """Remove the temporary mask file from disk if it exists."""
        if hasattr(self, 'mask_file'):
            if os.path.exists(self.mask_file):
                try:
                    shutil.rmtree(self.mask_file)
                except Exception as e:
                    print(f"Error removing mask file: {e}")
            del self.mask_file

    def write_results(self, results, p, t):
        self.mask[p, t] = results[0]

    def handle_previous_run(self):
        output_path = os.path.join(self.receipt['dirs']['masks_dir'], f"{self.receipt['steps'][self.step_name]['mask_name']}.tiff")
        if os.path.exists(output_path):
            return self.receipt['meta_arguments'].get('load_masks', False)
        else:
            return True

    def write_args_to_receipt(self,
                            mask_name,
                            channel,
                            pretrained_model_name,
                            diameter,
                            invert,
                            normalize,
                            do_3D,
                            min_size,
                            flow_threshold,
                            cellprob_threshold,):
        self.receipt['steps'][self.step_name]['mask_name'] = mask_name
        self.receipt['steps'][self.step_name]['channel'] = channel
        self.receipt['steps'][self.step_name]['pretrained_model_name'] = pretrained_model_name
        self.receipt['steps'][self.step_name]['diameter'] = diameter
        self.receipt['steps'][self.step_name]['invert'] = invert
        self.receipt['steps'][self.step_name]['normalize'] = normalize
        self.receipt['steps'][self.step_name]['do_3D'] = do_3D
        self.receipt['steps'][self.step_name]['min_size'] = min_size
        self.receipt['steps'][self.step_name]['flow_threshold'] = flow_threshold
        self.receipt['steps'][self.step_name]['cellprob_threshold'] = cellprob_threshold

    def run_process(self, p, t):
        print(p, t)
        self.iterate_over_data(p_range=[p], t_range=[t], run_in_parallel=False)

    @magicgui(
            call_button='Run',
            pretrained_model_name={"choices": 
                                   [fname
                                    for fname in os.listdir(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models'))
                                    if os.path.isfile(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models'), fname)) and '.' not in fname
                                    ]}
    )
    def interface(self, 
                p:int=0, t:int=0, 
                mask_name:str='default_name',
                channel:int=0, 
                pretrained_model_name: pathlib.Path = None,  
                diameter: float = 180, 
                invert: bool = False, 
                normalize: bool = True, 
                do_3D:bool=False, 
                min_size:float=500,
                flow_threshold:float=0, 
                cellprob_threshold:float=0):
        try:
            self.write_args_to_receipt(
                mask_name,
                channel,
                str(pretrained_model_name) if pretrained_model_name else None,
                diameter,
                invert,
                normalize,
                do_3D,
                min_size,
                flow_threshold,
                cellprob_threshold
            )
            self.preallocate_memmory()
            self.run_process(p, t)
        except Exception as e:
            print(f"[Error] Exception during segmentation: {e}")

        def add_dummy_channels(mask):
            """
            Expand a 5D mask (p, t, z, y, x) into 6D (p, t, c, z, y, x) with zeros in extra channels.
            """
            mask = np.array(mask)
            temp = np.zeros_like(self.data['images'])
            temp[:, :, self.receipt['steps'][self.step_name]['channel'], :, :, :] = mask
            return temp

        mask_layer_name = self.receipt['steps'][self.step_name]['mask_name']

        temp_mask = add_dummy_channels(self.mask)

        if mask_layer_name in self.viewer.layers:
            self.viewer.layers[mask_layer_name].data = temp_mask
        else:
            self.viewer.add_labels(
                temp_mask,
                name=mask_layer_name,
                axis_labels=('p', 't', 'c', 'z', 'y', 'x'),
                scale=[1, 1, 1, self.voxel_size_z/self.voxel_size_yx, 1, 1]
            )
        self.viewer.layers[mask_layer_name].refresh()



import time

class match_masks(abstract_task):
    @classmethod
    def task_name(cls):
        return 'match_masks'
    
    @property
    def required_keys(self):
        return None

    def handle_previous_run(self):
        return True

    def process(self, 
            new_params:dict = None, 
            p_range = None, 
            t_range = None,
            run_in_parallel:bool = False):
        start_time = time.time() 

        # loads data associated with receipt using data_loader
        self.data = load_data(self.receipt)

        # change parameters at run time
        if new_params:
            for k, v in new_params.items():
                self.receipt['steps'][self.step_name][k] = v

        nuc_mask_name = self.receipt['steps'][self.step_name].get('nuc_mask_name', 'nuc_masks')
        cyto_mask_name = self.receipt['steps'][self.step_name].get('cyto_mask_name', 'cyto_masks')
        single_nuc = self.receipt['steps'][self.step_name].get('single_nuc', True)
        cell_alone = self.receipt['steps'][self.step_name].get('cell_alone', False)

        nuc_masks = self.data[nuc_mask_name]
        cyto_masks = self.data[cyto_mask_name]


        for p in range(nuc_masks.shape[0]):
            for t in range(nuc_masks.shape[1]):
                nuc_masks[p, t], cyto_masks[p, t] = multistack.match_nuc_cell(
                                                            nuc_masks[p, t], 
                                                            cyto_masks[p, t], 
                                                            single_nuc,
                                                            cell_alone
                                                            )
        
        self.receipt['steps'][self.step_name]['nuc_mask_name'] = nuc_mask_name
        self.receipt['steps'][self.step_name]['cyto_mask_name'] = cyto_mask_name
        self.receipt['steps'][self.step_name]['single_nuc'] = single_nuc
        self.receipt['steps'][self.step_name]['cell_alone'] = cell_alone

        # records completion. This will mark completion for luigi
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "a") as f:
            f.write(f"{self.step_name} completed in {time.time() - start_time:.2f} seconds\n")

        return self.receipt










