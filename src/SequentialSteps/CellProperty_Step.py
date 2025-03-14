import numpy as np
import pathlib
import os
import tifffile
import matplotlib.pyplot as plt
from cellpose import models
from skimage.io import imread
import skimage as sk
import dask.array as da
from copy import copy, deepcopy
import pandas as pd

from src.GeneralStep import SequentialStepsClass



class CellProperties(SequentialStepsClass):
    def __init__(self):
        super().__init__()

    def main(self, image, nuc_mask, cell_mask, fov, timepoint, 
             props_to_measure= ['label', 'bbox', 'area', 'centroid', 'intensity_max', 'intensity_mean', 'intensity_min', 'intensity_std'], **kwargs):
        image = np.max(image, axis=1)
        image = image.squeeze()
        image = np.moveaxis(image, 0, -1)
    
        nuc_mask = nuc_mask[0, :, :]
        cell_mask = cell_mask[0, :, :]

        nuc_mask = nuc_mask.squeeze()
        cell_mask = cell_mask.squeeze()

        # make cyto mask
        cyto_mask = copy(cell_mask)
        cyto_mask[nuc_mask > 0] = 0

        def touching_border(df, image):
            """
            Checks if the region touches any border of the image.
            
            Parameters:
            - region: A regionprops object.
            - image_shape: Shape of the original image (height, width).
            
            Returns:
            - True if the region touches any border, False otherwise.
            """
            min_row, min_col, max_row, max_col = df['cell_bbox-0'], df['cell_bbox-1'], df['cell_bbox-2'], df['cell_bbox-3']
            return (min_row == 0) | (min_col == 0) | (max_row == image.shape[0]) | (max_col == image.shape[1])

        nuc_props = sk.measure.regionprops_table(nuc_mask.astype(int), image, properties=props_to_measure)
        cell_props = sk.measure.regionprops_table(cell_mask.astype(int), image, properties=props_to_measure)
        cyto_props = sk.measure.regionprops_table(cyto_mask.astype(int), image, properties=props_to_measure)

        nuc_df = pd.DataFrame(nuc_props)
        cell_df = pd.DataFrame(cell_props)
        cyto_df = pd.DataFrame(cyto_props)

        nuc_df.columns = ['nuc_' + col for col in nuc_df.columns]
        cell_df.columns = ['cell_' + col for col in cell_df.columns]
        cyto_df.columns = ['cyto_' + col for col in cyto_df.columns]

        cell_df ['touching_border'] = touching_border(cell_df, image)

        combined_df = pd.concat([nuc_df, cell_df, cyto_df], axis=1)
        combined_df['fov'] = [fov]*len(combined_df)
        combined_df['timepoint'] = [timepoint]*len(combined_df)

        return {'cell_properties': combined_df}






























