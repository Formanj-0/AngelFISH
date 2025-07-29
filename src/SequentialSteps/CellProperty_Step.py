import numpy as np
import pathlib
import os
import tifffile
import matplotlib.pyplot as plt
from cellpose import models
from skimage.io import imread
import skimage as sk
from skimage.filters import threshold_otsu
import dask.array as da
from copy import copy, deepcopy
import pandas as pd

from src.GeneralStep import SequentialStepsClass

class CellProperties(SequentialStepsClass):
    def __init__(self):
        super().__init__()

    def main(self, image, nuc_mask, cell_mask, fov, timepoint, 
             props_to_measure= ['label', 'bbox', 'area', 'centroid', 'intensity_max', 'intensity_mean', 'intensity_min', 'intensity_std'], **kwargs):
        image = np.max(image.compute(), axis=1)
        image = image.squeeze()
        image = np.moveaxis(image, 0, -1)
    
        nuc_mask = nuc_mask[0, :, :].compute()
        cell_mask = cell_mask[0, :, :].compute()

        nuc_mask = nuc_mask.squeeze()
        cell_mask = cell_mask.squeeze()

        # make cyto mask
        cyto_mask = copy(cell_mask)
        cyto_mask[nuc_mask > 0] = 0

        def touching_border(df, image):
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

        # Background estimation using 10th percentile
        flat_image = image.ravel()
        p10 = np.percentile(flat_image, 10)
        background_pixels = flat_image[flat_image <= p10]
        background_mean = np.mean(background_pixels)
        background_median = np.median(background_pixels)
        background_std = np.std(background_pixels)

        # Background estimation using Otsu
        otsu_thresh = threshold_otsu(flat_image)
        background_otsu_pixels = flat_image[flat_image <= otsu_thresh]
        background_mean_otsu = np.mean(background_otsu_pixels)
        background_median_otsu = np.median(background_otsu_pixels)
        background_std_otsu = np.std(background_otsu_pixels)

        background_stats = {
            'background_p10': p10,
            'background_mean': background_mean,
            'background_median': background_median,
            'background_std': background_std,
            'background_otsu_thresh': otsu_thresh,
            'background_mean_otsu': background_mean_otsu,
            'background_median_otsu': background_median_otsu,
            'background_std_otsu': background_std_otsu
        }

        for k, v in background_stats.items():
            combined_df[k] = v

        return {'cell_properties': combined_df}