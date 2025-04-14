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

    def main(self, image, fov, timepoint, cell_mask=None, nuc_mask=None,
             props_to_measure= ['label', 'bbox', 'area', 'centroid', 'intensity_max', 'intensity_mean', 'intensity_min', 'intensity_std'], **kwargs):
        image = np.max(image, axis=1)
        image = image.squeeze()
        image = np.moveaxis(image, 0, -1)
    
        if nuc_mask is not None:
            nuc_mask = nuc_mask[0, :, :]
            nuc_mask = nuc_mask.squeeze()

        if cell_mask is not None:
            cell_mask = cell_mask[0, :, :]
            cell_mask = cell_mask.squeeze()





        # make cyto mask
        if cell_mask is not None and nuc_mask is not None:
            cyto_mask = copy(cell_mask)
            cyto_mask[nuc_mask > 0] = 0
        else:
            cyto_mask = None

        def touching_border(df, image):
            """
            Checks if the region touches any border of the image.
            
            Parameters:
            - region: A regionprops object.
            - image_shape: Shape of the original image (height, width).
            
            Returns:
            - True if the region touches any border, False otherwise.
            """
            try:
                min_row, min_col, max_row, max_col = df['cell_bbox-0'], df['cell_bbox-1'], df['cell_bbox-2'], df['cell_bbox-3']
            except KeyError:
                min_row, min_col, max_row, max_col = df['nuc_bbox-0'], df['nuc_bbox-1'], df['nuc_bbox-2'], df['nuc_bbox-3']
            return (min_row == 0) | (min_col == 0) | (max_row == image.shape[0]) | (max_col == image.shape[1])

        if nuc_mask is not None:
            nuc_props = sk.measure.regionprops_table(nuc_mask.astype(int), image, properties=props_to_measure)
            nuc_df = pd.DataFrame(nuc_props)
            nuc_df.columns = ['nuc_' + col for col in nuc_df.columns]
        else:
            nuc_df = None


        if cell_mask is not None:
            cell_props = sk.measure.regionprops_table(cell_mask.astype(int), image, properties=props_to_measure)
            cell_df = pd.DataFrame(cell_props)
            cell_df.columns = ['cell_' + col for col in cell_df.columns]
        else:
            cell_df = None


        if cyto_mask is not None:
            cyto_props = sk.measure.regionprops_table(cyto_mask.astype(int), image, properties=props_to_measure)
            cyto_df = pd.DataFrame(cyto_props)
            cyto_df.columns = ['cyto_' + col for col in cyto_df.columns]
        else:
            cyto_df = None

        cell_df ['touching_border'] = touching_border(cell_df if cell_df is not None else nuc_df, image)

        combined_df = pd.concat([nuc_df, cell_df, cyto_df], axis=1)
        combined_df['fov'] = [fov]*len(combined_df)
        combined_df['timepoint'] = [timepoint]*len(combined_df)

        return {'cell_properties': combined_df}






























