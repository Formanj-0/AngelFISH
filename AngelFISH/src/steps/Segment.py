from src import abstract_task, load_data

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

# from src import SequentialStepsClass
# from src.Parameters import Parameters

from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt

class segment(abstract_task):

    @property
    def task_name(self):
        return 'segment'

    @property
    def required_keys(self):
        return ['mask_name', 'channel']

    def image_processing_task(self, p, t):
        # get the user defined arguments
        given_args = self.receipt['steps'][self.step_name]

        # gets the data specific to the call
        data_to_send = {}
        zyx_image = self.data['images'][p, t, given_args['channel'], :, :, :].compute()
        data_to_send['zyx_image'] = zyx_image

        all_args = {**given_args, **data_to_send}

        # runs the image processing 
        mask, flows, styles, diams = self.image_processing_function(**all_args)

    def preallocate_memmory(self):
        if hasattr(self, 'temp_dir') and self.temp_dir is not None and os.path.exists(self.temp_dir):
            # temp_dir exists
            pass
        else:
            self.temp_dir = os.path.join(self.receipt['dirs']['analysis_dir'], f'temp_{self.step_name}')
            os.makedirs(self.temp_dir, exist_ok=True)
            self.create_temp_mask()

    def create_temp_mask(self):
        """Create a temporary mask file on disk that can be accessed by multiple processes."""
        if not hasattr(self, 'mask_file'):
            # Get mask name from receipt
            mask_name = self.receipt['steps'].get(self.step_name, {}).get('mask_name', 'temp_mask')
            
            # Create mask file path
            self.mask_file = os.path.join(self.temp_dir, f"{mask_name}.zarr")
            
            # Get shape from images, removing channel dimension
            image_shape = self.data['images'].shape
            mask_shape = image_shape[:2] + image_shape[3:]  # Remove channel dimension
            
            # Create a dask array for the mask
            chunks = 'auto'  # Or specify a chunk size that works for your data
            
            # Check if mask already exists
            if os.path.exists(self.mask_file):
                # Load existing mask
                self.mask = da.from_zarr(self.mask_file)
            else:
                # Create new mask initialized with zeros
                self.mask = da.zeros(mask_shape, dtype=np.int32, chunks=chunks)
                self.mask.to_zarr(self.mask_file)
                
        return self.mask

    @staticmethod
    def image_processing_function(zyx_image, 
                                  pretrained_model_name: str | bool = False,  
                                  cellpose_model_type:str='cyto3', 
                                  diameter: float = 180, 
                                  invert: bool = False, 
                                  normalize: bool = True, 
                                  channel_axis: int = 0, 
                                  do_3D:bool=False, 
                                  min_size:float=500,
                                  flow_threshold:float=0, 
                                  cellprob_threshold:float=0, 
                                  **kwargs):
        model_location = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        pretrained_model = os.path.join(model_location, pretrained_model_name) if pretrained_model_name else None

        cp = models.CellposeModel(model_type=cellpose_model_type, gpu=True, pretrained_model=pretrained_model)
        sz = models.SizeModel(cp)

        model = models.Cellpose(gpu=True)
        model.cp = cp
        model.sz = sz
        channels = [0, 0]
        mask, flows, styles, diams = model.eval(zyx_image,
                                    channels=channels, 
                                    diameter=diameter, 
                                    invert=invert, 
                                    normalize=normalize, 
                                    channel_axis=channel_axis, 
                                    do_3D=do_3D,
                                    min_size=min_size, 
                                    flow_threshold=flow_threshold, 
                                    cellprob_threshold=cellprob_threshold)
                                            # net_avg=True

        return mask, flows, styles, diams























