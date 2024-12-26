import numpy as np
from skimage.io import imread
import tifffile
import os
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
from skimage.measure import find_contours
from scipy import signal
from scipy import ndimage
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import matplotlib as mpl
# import bigfish.segmentation as segmentation

mpl.rc('image', cmap='viridis')
plt.style.use('ggplot')  # ggplot  #default

import socket
import pathlib
import yaml
import shutil
from fpdf import FPDF
import gc
import pickle
import pycromanager as pycro
import pandas as pd
import cellpose
from cellpose import models
import torch
import warnings
# import tensorflow as tf
import bigfish
import bigfish.stack as stack
import bigfish.detection as detection
import bigfish.multistack as multistack
import bigfish.plot as plot
from typing import Union
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.io import imsave
import seaborn as sns
from skimage import exposure
import io

warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)

# Selecting the GPU. This is used in case multiple scripts run in parallel.
try:
    import torch

    number_gpus = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    if number_gpus > 1:  # number_gpus
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(np.random.randint(0, number_gpus, 1)[0])
except:
    print('No GPUs are detected on this computer. Please follow the instructions for the correct installation.')
import zipfile
import seaborn as sns
import scipy.stats as stats
from matplotlib.ticker import FuncFormatter
from matplotlib_scalebar.scalebar import ScaleBar

font_props = {'size': 16}
import joypy
from matplotlib import cm
from scipy.ndimage import binary_dilation
import sys
import skimage as sk
from skimage import exposure
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import exposure
from tifffile import imsave
import copy
from scipy.optimize import curve_fit
from abc import abstractmethod
import dask.array as da
import dask


# append the path two directories before this file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src import SequentialStepsClass, IndependentStepClass # TODO: remove this
from src.Parameters import Parameters



#%%
class FilteredImages(IndependentStepClass):
    def main(self, da, sigma_dict, display_plots: bool=False, **kwargs):
        """
        Main function to run the filters.

        Parameters:
        - da: Dask array with shape [p, t, c, y, x]
        - sigma_dict: Dictionary with sigma values per channel {channel_index: sigma_value}
        - display_plots: Boolean to control plotting

        Returns:
        - output: FiltersOutputClass object
        """
        # Step 1: Apply the filters
        corrected_images = self.average_illumination_profile(da,sigma_dict, display_plots)

        return {'images': corrected_images}

    @abstractmethod
    def average_illumination_profile(self, **kwargs) -> da.array:
        """
        Abstract method to be implemented in the child classes.

        Parameters:
        - kwargs: Dictionary with the required parameters

        Returns:
        - corrected_images: Dask array with shape [p, t, c, y, x]
        """
        pass

class rescale_images(IndependentStepClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def main(self, images: np.array, channel_to_stretch: int = None, 
             stretching_percentile:float = 99.9, display_plots: bool = False, **kwargs):
        
        for p in range(images.shape[0]):
            for t in range(images.shape[1]):
                img = images[p, t, :, :, :, :]
                img = stack.rescale(img.compute(), channel_to_stretch=channel_to_stretch, stretching_percentile=stretching_percentile)
                images[p, t, :, :, :, :] = img

        if display_plots:
            for c in range(images.shape[2]):
                plt.imshow(np.max(images[0, 0, c, :, :, :], axis=0))
                plt.title(f'channel {c}')
                plt.show()

        return {'images': images}

        

class remove_background(SequentialStepsClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def main(self, image: np.array, FISHChannel: list[int], id: int, spot_z, spot_yx, voxel_size_z, voxel_size_yx,
             filter_type: str = 'gaussian', sigma: float = None, display_plots:bool = False, 
             kernel_shape: str = 'disk', kernel_size = 200, **kwargs):

        rna = np.squeeze(image[:, :, :, FISHChannel[0]])

        if display_plots:
            plt.imshow(np.max(rna, axis=0) if len(rna.shape) > 2 else rna)
            plt.title(f'pre-filtered image')
            plt.show()

        if filter_type == 'gaussian':
            if sigma is None:
                voxel_size_nm = (int(voxel_size_z), int(voxel_size_yx), int(voxel_size_yx)) if len(rna.shape) == 3 else (int(voxel_size_yx), int(voxel_size_yx))
                spot_size_nm = (int(spot_z), int(spot_yx), int(spot_yx)) if len(rna.shape) == 3 else (int(spot_yx), int(spot_yx))
                sigma = detection.get_object_radius_pixel(
                        voxel_size_nm=voxel_size_nm, 
                        object_radius_nm=spot_size_nm, 
                        ndim=3 if len(rna.shape) == 3 else 2)
            rna = stack.remove_background_gaussian(rna, sigma=sigma)

        elif filter_type == 'log_filter':
            rna = stack.log_filter(rna, sigma=sigma)

        elif filter_type == 'mean':
            rna = stack.remove_background_mean(np.max(rna, axis=0) if len(rna.shape) > 2 else rna, 
                                               kernel_shape=kernel_shape, kernel_size=kernel_size)
        else:
            raise ValueError('Invalid filter type')
        
        image[:, :, :, FISHChannel[0]] = rna

        if display_plots:
            plt.imshow(np.max(rna, axis=0) if len(rna.shape) > 2 else rna)
            plt.title(f'filtered image, type: {filter_type}, sigma: {sigma}')
            plt.show()

        return {'image': image}


