import h5py
import matplotlib.pyplot as plt
import dask.array as da
import numpy as np
import dask.dataframe as dp
import pandas as pd
from typing import Union
from abc import ABC, abstractmethod
import os
from skimage.measure import find_contours
from skimage import exposure
import matplotlib.patches as mpatches
import copy
import matplotlib.cm as cm
import matplotlib.colors as mcolors


class AnalysisManager:
    """
    This class is made to select data for further analysis.
    It provides methods to load, filter, and retrieve datasets from HDF5 files.
    """
    def __init__(self, location:Union[str, list[str]]=None, log_location:str=None, mac:bool=False):
        # given:
        # h5 locations
        #   give me a location
        #   give me a list of locations
        #   give me none -> got to here and display these \\munsky-nas.engr.colostate.edu\share\Users\Jack\All_Analysis
        if location is None: # TODO make these if statement better
            self.get_locations(log_location, mac)
        elif isinstance(location, str):
            self.location = [location]
        elif isinstance(location, list): # TODO make sure its a list of str
            self.location = location
        else:
            raise ValueError('Location is not properly defined')
        
        self._load_in_h5()

    def get_locations(self, log_location, mac:bool=False) -> list[str]: 
        # log_location = r'Y:\Users\Jack\All_Analysis' # TODO: make this work for all users
        # get the log files 
        log_files = os.listdir(log_location)

        # read the log files and spit on ' -> '
        self.location = []
        for l in log_files:
            with open(os.path.join(log_location, l), 'r') as file:
                content = file.read()
                
            content = content.split('\n')[-2]
            first, second = content.split(' -> ')
            name = first.split(r'/')[-1]
            if mac:
                drive = '/Volumes/share/'
            else:
                drive = os.path.splitdrive(log_location)[0] + os.sep
            second = os.path.join(*second.split('/')).strip()
            location = os.path.join(drive, second, name)
            # print(location)
            self.location.append(location)

    def select_analysis(self, analysis_name: str = None, date_range: list[str] = None):
        self._find_analysis_names()
        self._filter_on_date(date_range)
        self._filter_on_name(analysis_name)
        self._find_analysis() # this give you self.analysis
        self._handle_duplicates()
        return self.h5_files

    def list_analysis_names(self):
        self._find_analysis_names()
        for name in self.analysis_names:
            print(name)
        return self.analysis_names

    def select_datasets(self, dataset_name) -> list:
        if hasattr(self, 'analysis'):
            self.datasets = []
            for h in self.analysis:
                try:
                    self.datasets.append(h[dataset_name])
                except KeyError:
                    print(f'{h} missing datasets')
                    self.datasets.append(None)
            return self.datasets
        else:
            print('select an anlysis')

    def list_datasets(self):
        if hasattr(self, 'analysis'):
            for d in self.analysis:
                print(d.name, list(d.keys()))
        else:
            print('select an analysis')

    def _filter_on_name(self, analysis_name):
        # self.analysis_names = [s.split('_')[1] for s in self.analysis_names]
        self.analysis_names = ['_'.join(s.split('_')[1:-1]) for s in self.analysis_names]
        if analysis_name is not None:
            self.analysis_names = [s for s in self.analysis_names if s == analysis_name]

    def _filter_on_date(self, date_range):
        self.dates = set([s.split('_')[-1] for s in self.analysis_names])
        if date_range is not None:
            start_date, end_date = date_range
            self.dates = [date for date in self.dates if start_date <= date <= end_date]
        self.dates = list(self.dates)

    def _find_analysis(self):
        # select data sets with self.data, and self.datasete
        self.analysis = []
        bad_idx = []
        for h_idx, h in enumerate(self.h5_files):
            for dataset_name in set(self.analysis_names):
                combos = [f'Analysis_{dataset_name}_{date}' for date in self.dates]
                if any(combo in h.keys() for combo in combos):
                    for combo in combos: # seach for the combo
                        if combo in h.keys():
                            self.analysis.append(h[combo])
                            break
                else:
                    bad_idx.append(h_idx)
        for i in bad_idx:
            self.h5_files[i].close()
        self.h5_files = [h for i, h in enumerate(self.h5_files) if i not in bad_idx]

    def _handle_duplicates(self): # requires user input
        pass
        # TODO: check if h5 has multiple analysis in it

    def _find_analysis_names(self):
        self.analysis_names = []
        for h in self.h5_files:
            self.analysis_names.append(list(h.keys()))
        self.analysis_names = set([dataset for sublist in self.analysis_names for dataset in sublist])
        self.analysis_names = [d for d in self.analysis_names if 'Analysis' in d]

    def _load_in_h5(self):
        self.h5_files = []
        for l in self.location:
            if l not in [h.filename for h in self.h5_files]:
                self.h5_files.append(h5py.File(l, 'a'))
    
    def get_images_and_masks(self):
        self.raw_images = [da.from_array(h['raw_images']) for h in self.h5_files]
        self.masks = [da.from_array(h['masks']) for h in self.h5_files]
        return self.raw_images, self.masks
    
    def close(self):
        for h in self.h5_files:
            h.close()

    def delete_selected_analysis(self, password):
        if password == 'you_better_know_what_your_deleting':
            for analysis_group in self.analysis:
                parent_group = analysis_group.parent
                filename = copy.copy(parent_group.file.filename)
                del parent_group[analysis_group.name]
        else:
            print('you_better_know_what_your_deleting')
#%% Analysis outline
class Analysis(ABC):

    """
    Analysis is an abstract base class (ABC) that serves as a blueprint for further analysis classes. 
    It ensures that any subclass implements the necessary methods for data handling and validation.
    The confirmation that a class is working should be random to ensure minmum bias

    Attributes:
        am: An instance of a class responsible for managing analysis-related operations.
        seed (float, optional): A seed value for random number generation to ensure reproducibility.
    Methods:
        get_data():
            Abstract method to be implemented by subclasses for retrieving data.
        save_data():
            Abstract method to be implemented by subclasses for saving data.
        display():
            Abstract method to be implemented by subclasses for displaying data or results.
        validate():
            Abstract method to be implemented by subclasses for validating the analysis or data.
        close():
            Closes the analysis manager instance.    
    """
    def __init__(self, am, seed: float = None):
        super().__init__()
        self.am = am
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def save_data(self):
        pass

    @abstractmethod
    def display(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    def close(self):
        self.am.close()

#%%
# Analysis children
########################################
# Helper functions for drawing spots
########################################
def draw_spot_circle(ax, x, y, radius=4, color='gold'):
    """
    Draws an unfilled circle around (x,y).
    """
    circle = mpatches.Circle((x, y), radius=radius, fill=False,
                             edgecolor=color, linewidth=2)
    ax.add_patch(circle)

def draw_spot_arrow(ax, x, y, color='gold'):
    """
    Draws a small arrow from (x, y) to (x+3, y).
    """
    ax.arrow(x-4, y, 3, 0, head_width=3,
             color=color, length_includes_head=True)


########################################
# Main Class
########################################
class SpotDetection_Confirmation(Analysis):
    """
    SpotDetection_Confirmation class with:
      1) Full-FOV segmentation (nucleus & cytoplasm)
      2) Zoom on a single cell with circles or arrows for spots
         + percentile-based contrast
      3) (Optional) further zoom on one spot
      4) 2x2 figure for SNR thresholds (show kept vs. discarded)
    """

    def __init__(self, am, seed=None):
        super().__init__(am, seed)
        self.fov = None
        self.h5_idx = None
        self.cell_label = None

    def get_data(self):
        """
        Loads spots, clusters, cellprops, cellspots from HDF5,
        plus images and masks.
        """
        h = self.am.h5_files
        d_spot     = self.am.select_datasets('spotresults')
        d_cellres  = self.am.select_datasets('cellresults')
        d_props    = self.am.select_datasets('cell_properties')
        d_cluster  = self.am.select_datasets('clusterresults')

        self.spots = []
        self.clusters = []
        self.cellprops = []
        self.cellspots = []
        for i, s in enumerate(h):
            # Spots
            df_spot = pd.read_hdf(s.filename, d_spot[i].name)
            df_spot['h5_idx'] = i
            self.spots.append(df_spot)

            # Clusters
            try:
                df_clust = pd.read_hdf(s.filename, d_cluster[i].name)
                df_clust['h5_idx'] = i
                self.clusters.append(df_clust)
            except AttributeError:
                pass  # If missing cluster data

            # Cellprops
            df_prop = pd.read_hdf(s.filename, d_props[i].name)
            df_prop['h5_idx'] = i
            self.cellprops.append(df_prop)

            # Cellspots
            df_cellres = pd.read_hdf(s.filename, d_cellres[i].name)
            df_cellres['h5_idx'] = i
            self.cellspots.append(df_cellres)

        self.spots = pd.concat(self.spots, axis=0)
        self.clusters = pd.concat(self.clusters, axis=0)
        self.cellprops = pd.concat(self.cellprops, axis=0)
        self.cellspots = pd.concat(self.cellspots, axis=0)

        self.images, self.masks = self.am.get_images_and_masks()

    def save_data(self, location):
        """
        Saves the DataFrames to CSV.
        """
        self.spots.to_csv(os.path.join(location, 'spots.csv'), index=False)
        self.clusters.to_csv(os.path.join(location, 'clusters.csv'), index=False)
        self.cellprops.to_csv(os.path.join(location, 'cellprops.csv'), index=False)
        self.cellspots.to_csv(os.path.join(location, 'cellspots.csv'), index=False)

    ############################################################
    # MAIN DISPLAY: orchestrates the steps
    ############################################################
    def display(self, newFOV=True, newCell=True, spotChannel=0, cytoChannel=1, nucChannel=2):
        """
        Steps:
        1) Full-FOV segmentation
        2) Zoom on chosen cell (percentile-based contrast, circles/arrows)
        3) Further zoom on a single spot
        4) SNR threshold 2x2 figure
        """
        # Possibly pick a new random FOV/cell
        if self.fov is None or newFOV:
            self.h5_idx = np.random.choice(self.spots['h5_idx'])
            self.fov = np.random.choice(self.spots[self.spots['h5_idx'] == self.h5_idx]['fov'])
        if self.cell_label is None or newCell:
            valid_labels = self.cellprops[
                (self.cellprops['h5_idx'] == self.h5_idx) &
                (self.cellprops['fov'] == self.fov) &
                (self.cellprops['cell_label'] != 0)
            ]['cell_label'].unique()
            if len(valid_labels) == 0:
                print("No valid cell_label in this FOV. Aborting.")
                return
            self.cell_label = np.random.choice(valid_labels)

        # 1) Full-FOV segmentation
        self._display_full_fov_segmentation(cytoChannel, nucChannel)

        # 2) Zoom on cell
        chosen_spot = self._display_zoom_on_cell(spotChannel, cytoChannel, nucChannel)

        # 3) Further zoom on a single spot
        if chosen_spot is not None:
            self._display_zoom_on_one_spot(spotChannel, chosen_spot)

        # 4) SNR threshold figure
        self._display_snr_thresholds(spotChannel, cytoChannel, nucChannel)

    ############################################################
    # 1) Full-FOV segmentation (no spots)
    ############################################################
    def _display_full_fov_segmentation(self, cytoChannel, nucChannel):
        """
        Shows nucleus channel + mask (left),
        cytoplasm channel + mask (right).
        """
        img_nuc = self.images[self.h5_idx][self.fov, 0, nucChannel, :, :, :]
        img_cyto = self.images[self.h5_idx][self.fov, 0, cytoChannel, :, :, :]

        mask_nuc = self.masks[self.h5_idx][self.fov, 0, nucChannel, :, :, :]
        mask_cyto = self.masks[self.h5_idx][self.fov, 0, cytoChannel, :, :, :]

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        for ax in axs:
            ax.axis("off")

        # Nucleus
        axs[0].imshow(np.max(img_nuc, axis=0), cmap='gray')
        axs[0].imshow(np.max(mask_nuc, axis=0), cmap='jet', alpha=0.3)
        axs[0].set_title(f"FOV={self.fov} - Nucleus segmentation")

        # Cytoplasm
        axs[1].imshow(np.max(img_cyto, axis=0), cmap='gray')
        axs[1].imshow(np.max(mask_cyto, axis=0), cmap='jet', alpha=0.3)
        axs[1].set_title(f"FOV={self.fov} - Cytoplasm segmentation")

        plt.tight_layout()
        plt.show()

    ############################################################
    # 2) Zoom on cell with percentile-based contrast
    ############################################################
    def _display_zoom_on_cell(self, spotChannel, cytoChannel, nucChannel):
        """
        Zoom on bounding box, compute percentile-based contrast,
        then overlay spots with circles or arrows. Highlights the chosen spot.
        """
        cdf = self.cellprops[
            (self.cellprops['h5_idx'] == self.h5_idx) &
            (self.cellprops['fov'] == self.fov) &
            (self.cellprops['cell_label'] == self.cell_label)
        ]
        if cdf.empty:
            print("Cell not found. Aborting cell zoom.")
            return None

        row_min = int(cdf['cell_bbox-0'].iloc[0])
        col_min = int(cdf['cell_bbox-1'].iloc[0])
        row_max = int(cdf['cell_bbox-2'].iloc[0])
        col_max = int(cdf['cell_bbox-3'].iloc[0])

        # Spot channel -> 2D
        img_spot_3d = self.images[self.h5_idx][self.fov, 0, spotChannel, :, :, :]
        img_spot_2d = np.max(img_spot_3d, axis=0)

        # Slice and compute
        crop_spot_dask = img_spot_2d[row_min:row_max, col_min:col_max]
        crop_spot = crop_spot_dask.compute()

        # Percentile-based contrast
        if crop_spot.size > 0:
            p1, p99 = np.percentile(crop_spot, (1, 99))
            crop_spot_stretched = exposure.rescale_intensity(
                crop_spot, in_range=(p1, p99), out_range=(0, 1)
            )
        else:
            crop_spot_stretched = crop_spot

        # Masks -> also slice
        mask_nuc_3d = self.masks[self.h5_idx][self.fov, 0, nucChannel, :, :, :]
        mask_cyto_3d = self.masks[self.h5_idx][self.fov, 0, cytoChannel, :, :, :]
        mask_nuc_2d = np.max(mask_nuc_3d, axis=0)
        mask_cyto_2d = np.max(mask_cyto_3d, axis=0)

        crop_nuc_dask = mask_nuc_2d[row_min:row_max, col_min:col_max]
        crop_nucmask = crop_nuc_dask.compute()

        crop_cyto_dask = mask_cyto_2d[row_min:row_max, col_min:col_max]
        crop_cytomask = crop_cyto_dask.compute()

        # Spots in this cell
        cell_spots = self.spots[
            (self.spots['h5_idx'] == self.h5_idx) &
            (self.spots['fov'] == self.fov) &
            (self.spots['cell_label'] == self.cell_label)
        ]
        if cell_spots.empty:
            print("No spots found in the cell.")
            return None

        chosen_spot = cell_spots.sample(1).iloc[0]  # Randomly choose one spot

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Left: raw spot channel + mask overlay
        axs[0].imshow(crop_spot_stretched, cmap='gray')
        axs[0].imshow(crop_nucmask, cmap='Blues', alpha=0.3)
        axs[0].imshow(crop_cytomask, cmap='Reds', alpha=0.3)
        axs[0].set_title(f"Cell {self.cell_label} - Masks overlay")

        # Right: contrast-stretched channel
        axs[1].imshow(crop_spot_stretched, cmap='gray')
        axs[1].imshow(crop_nucmask, cmap='Blues', alpha=0.2)
        axs[1].imshow(crop_cytomask, cmap='Reds', alpha=0.2)
        axs[1].set_title(f"Cell {self.cell_label} - Stretched Spot Channel")

        dx, dy = col_min, row_min

        # Mark spots
        for _, spot in cell_spots.iterrows():
            sx = spot['x_px'] - dx
            sy = spot['y_px'] - dy
            color = 'blue' if (spot['x_px'] == chosen_spot['x_px'] and spot['y_px'] == chosen_spot['y_px']) else 'gold'
            draw_spot_circle(axs[1], sx, sy, radius=4, color=color)

        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

        return chosen_spot

    ############################################################
    # 3) Further zoom on a single spot
    ############################################################
    def _display_zoom_on_one_spot(self, spotChannel, chosen_spot):
        """
        Further zoom on a single spot. Highlights the chosen spot and others in the patch.
        """
        sx = int(chosen_spot['x_px'])
        sy = int(chosen_spot['y_px'])

        pad = 15
        x1 = max(sx - pad, 0)
        x2 = sx + pad
        y1 = max(sy - pad, 0)
        y2 = sy + pad

        img_spot_3d = self.images[self.h5_idx][self.fov, 0, spotChannel, :, :, :]
        img_spot_2d = np.max(img_spot_3d, axis=0)

        sub_img_dask = img_spot_2d[y1:y2, x1:x2]
        sub_img = sub_img_dask.compute()

        # Percentile
        if sub_img.size > 0:
            p1, p99 = np.percentile(sub_img, (1, 99))
            sub_img_stretched = exposure.rescale_intensity(
                sub_img, in_range=(p1, p99), out_range=(0, 1)
            )
        else:
            sub_img_stretched = sub_img

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(sub_img_stretched, cmap='gray')
        ax.set_title(f"Spot zoom (cell {self.cell_label})")

        # Mark chosen spot in blue
        draw_spot_circle(ax, pad, pad, radius=4, color='blue')

        # Mark other spots in red if they are in the patch
        cell_spots = self.spots[
            (self.spots['h5_idx'] == self.h5_idx) &
            (self.spots['fov'] == self.fov) &
            (self.spots['cell_label'] == self.cell_label)
        ]
        for _, srow in cell_spots.iterrows():
            rx = srow['x_px'] - x1
            ry = srow['y_px'] - y1
            if (rx < 0 or ry < 0 or rx >= (x2 - x1) or ry >= (y2 - y1)):
                continue
            if (srow['x_px'] == chosen_spot['x_px'] and srow['y_px'] == chosen_spot['y_px']):
                continue
            draw_spot_circle(ax, rx, ry, radius=3, color='gold')

        ax.axis("off")
        plt.tight_layout()
        plt.show()


    ############################################################
    # 4) Single figure with 2x2 subplots for SNR thresholds
    ############################################################
    def _display_snr_thresholds(self, spotChannel, cytoChannel, nucChannel, thresholds=[0, 2, 3, 4]):
        """
        Show a figure for each threshold T:
        - Left subplot: gold circles for spots above SNR >= T, red circles for spots below, with different shades of red for new spots below the threshold.
        - Right subplot: gold circles for detected spots with SNR >= T.
        """
        cdf = self.cellprops[
            (self.cellprops['h5_idx'] == self.h5_idx) &
            (self.cellprops['fov'] == self.fov) &
            (self.cellprops['cell_label'] == self.cell_label)
        ]
        if cdf.empty:
            print("Cell not found. Skipping SNR thresholds.")
            return

        row_min = int(cdf['cell_bbox-0'].iloc[0])
        col_min = int(cdf['cell_bbox-1'].iloc[0])
        row_max = int(cdf['cell_bbox-2'].iloc[0])
        col_max = int(cdf['cell_bbox-3'].iloc[0])

        img_spot_3d = self.images[self.h5_idx][self.fov, 0, spotChannel, :, :, :]
        img_spot_2d = np.max(img_spot_3d, axis=0)

        crop_spot_dask = img_spot_2d[row_min:row_max, col_min:col_max]
        crop_spot = crop_spot_dask.compute()

        if crop_spot.size > 0:
            p1, p99 = np.percentile(crop_spot, (1, 99))
            crop_spot_stretched = exposure.rescale_intensity(
                crop_spot, in_range=(p1, p99), out_range=(0, 1)
            )
        else:
            crop_spot_stretched = crop_spot

        # Masks -> also slice
        mask_nuc_3d = self.masks[self.h5_idx][self.fov, 0, nucChannel, :, :, :]
        mask_cyto_3d = self.masks[self.h5_idx][self.fov, 0, cytoChannel, :, :, :]
        mask_nuc_2d = np.max(mask_nuc_3d, axis=0)
        mask_cyto_2d = np.max(mask_cyto_3d, axis=0)

        crop_nuc_dask = mask_nuc_2d[row_min:row_max, col_min:col_max]
        crop_nucmask = crop_nuc_dask.compute()

        crop_cyto_dask = mask_cyto_2d[row_min:row_max, col_min:col_max]
        crop_cytomask = crop_cyto_dask.compute()    

        fig, axs = plt.subplots(len(thresholds), 2, figsize=(12, len(thresholds) * 6))
        dx, dy = col_min, row_min

        # All spots in cell
        cell_spots_all = self.spots[
            (self.spots['h5_idx'] == self.h5_idx) &
            (self.spots['fov'] == self.fov) &
            (self.spots['cell_label'] == self.cell_label)
        ]

        # Define different shades of red for thresholds
        shades_of_red = [
            (1, 0.8, 0.8),  # Light red
            (1, 0.6, 0.6),  # Medium-light red
            (1, 0.4, 0.4),  # Medium red
            (1, 0.2, 0.2)   # Dark red
        ]

        for i, thr in enumerate(thresholds):
            # Left: Spots shaded red depending on SNR threshold
            ax = axs[i, 0]
            ax.imshow(crop_spot_stretched, cmap='gray')
            ax.imshow(crop_nucmask, cmap='Blues', alpha=0.2)
            ax.imshow(crop_cytomask, cmap='Reds', alpha=0.2)
            ax.set_title(f"SNR >= {thr} (Shaded Reds and Gold)")

            # Mark spots below threshold with progressively darker reds
            for j, t in enumerate(thresholds[:i + 1]):  # Up to the current threshold
                cell_spots_below = cell_spots_all[
                    (cell_spots_all['snr'] < t) & (cell_spots_all['snr'] >= (thresholds[j - 1] if j > 0 else 0))
                ]
                shade_color = shades_of_red[j % len(shades_of_red)]
                for _, spot in cell_spots_below.iterrows():
                    sx = spot['x_px'] - dx
                    sy = spot['y_px'] - dy
                    draw_spot_circle(ax, sx, sy, radius=4, color=shade_color)

            # Mark spots above threshold in gold
            cell_spots_in = cell_spots_all[cell_spots_all['snr'] >= thr]
            for _, spot in cell_spots_in.iterrows():
                sx = spot['x_px'] - dx
                sy = spot['y_px'] - dy
                draw_spot_circle(ax, sx, sy, radius=4, color='gold')

            ax.axis("off")

            # Right: Gold circles for detected spots
            ax = axs[i, 1]
            ax.imshow(crop_spot_stretched, cmap='gray')
            ax.set_title(f"SNR >= {thr} (Detected Spots)")

            for _, spot in cell_spots_in.iterrows():
                sx = spot['x_px'] - dx
                sy = spot['y_px'] - dy
                draw_spot_circle(ax, sx, sy, radius=4, color='gold')

            ax.axis("off")

        plt.tight_layout()
        plt.show()

    def validate(self):
        # check cyto, cell, and nuc labels are the same
        if np.all(self.cellprops['cell_label'] == self.cellprops['nuc_label']):
            print('nuc and cell labels match')
        else:
            print('ERROR: nuc and cell labels dont match')

        if np.all(self.cellprops['cell_label'] == self.cellprops['cyto_label']):
            print('cyto and cell labels match')
        else:
            print('ERROR: cyto and cell labels dont match')

        if np.all(self.cellprops['nuc_label'] == self.cellprops['cyto_label']):
            print('cyto and nuc labels match')
        else:
            print('ERROR: cyto and nuc labels dont match')

        # confirm spots belong to the correct label TODO

    def set_relationships(self):
        # create local keys
        self.spots['spot_key'] = np.arange(self.spots.shape[0])
        self.cellprops['cell_key'] = np.arange(self.cellprops.shape[0])
        self.clusters['cluster_key'] = np.arange(self.clusters.shape[0])

        # relate cellprops to cluster and spots
        for i, cell in self.cellprops.iterrows():
            matching_spots = (self.spots['cell_label'] == cell['nuc_label']) & \
                     (self.spots['fov'] == cell['fov']) & \
                     (self.spots['timepoint'] == cell['timepoint']) & \
                     (self.spots['NAS_location'] == cell['NAS_location'])
            self.spots.loc[matching_spots, 'cell_key'] = cell['cell_key']

            matching_clusters = (self.clusters['cell_label'] == cell['nuc_label']) & \
                    (self.clusters['fov'] == cell['fov']) & \
                    (self.clusters['timepoint'] == cell['timepoint']) & \
                    (self.clusters['NAS_location'] == cell['NAS_location'])
            self.clusters.loc[matching_clusters, 'cell_key'] = cell['cell_key']

        # confirm results, count the number of spots per each cell_key
        for cell_key in np.random.choice(self.cellprops['cell_key'], 10):
            cell = self.cellprops[self.cellprops['cell_key'] == cell_key]
            spot_count = self.spots[self.spots['cell_key'] == cell_key].shape[0]
            cell_spot_count = self.cellspots[self.cellspots['cell_key'] == cell_key]['nb_rna'].sum() # there should only be one of these 
            assert spot_count == cell_spot_count, f"Mismatch in spot count for cell_key {cell_key}: {spot_count} vs {cell_spot_count}"


    
    def gate():
        pass

class GR_Confirmation(Analysis):
    """
    GR_Confirmation is a class designed to confirm the accuracy of image processing results 
    for ICC of GR (Glucocorticoid Receptor). It handles two channels (nucleus & GR), 
    applies illumination correction, and visualizes segmentation masks.
    
    Attributes:
    -----------
    am : object
        The AnalysisManager instance for dataset and image operations.
    seed : int, optional
        For reproducibility, default None.
    cellprops : pd.DataFrame
        Per-cell properties loaded from HDF5 files.
    illumination_profiles : dask.array or ndarray
        Illumination profiles for correction.
    corrected_IL_profile: dask.array or ndarray
        Corrected illumination profile.    
    images : array-like or dask.array
        Image data: shape might be (fov, 1, channels, z, y, x) or similar.
    masks : array-like or dask.array
        Mask data: shape might be (fov, 1, channels, z, y, x) or similar.

    Methods:
    --------
    __init__(self, am, seed=None)
    get_data(self)
    save_data(self, location)
    display(self, GR_Channel=0, Nuc_Channel=1)
    validate(self, cell_mask, nuc_mask, image, label, measurements)
    """
    def __init__(self, am, seed=None):
        super().__init__(am, seed)

    def get_data(self):
        """ Load data from the AnalysisManager. """
        h = self.am.h5_files
        d = self.am.select_datasets('cell_properties')

        # Concatenate cellprops from each file, track which h5 they came from
        self.cellprops = []
        for i, s in enumerate(h):
            df = pd.read_hdf(s.filename, d[i].name)
            df['h5_idx'] = [i] * len(df)
            self.cellprops.append(df)
        self.cellprops = pd.concat(self.cellprops, axis=0)

        # Illumination profiles (assume shape: (n_channels, y, x))
        self.illumination_profiles = da.from_array(
            self.am.select_datasets('illumination_profiles')
        )[0, :, :, :]

        # Images and masks (often shape: (n_fov, 1, n_channels, [z,] y, x))
        self.images, self.masks = self.am.get_images_and_masks()

    def save_data(self, location):
        """ Save cell properties to CSV. """
        self.cellprops.to_csv(location, index=False)

    def display(self, GR_Channel: int = 0, Nuc_Channel: int = 1):
        """
        Displays:
          1) Illumination correction (raw, profile, corrected, corrected profile) for GR channel
          2) Segmentation overlays 
             - Left: raw Nuc + nuc_mask
             - Right: raw GR + pseudo_cyto
          3) Intensity histograms (using dask.histogram to avoid large memory usage)
          4) (Optional) bounding box + validation for a random cell
        """
        # ------------------------------------------------
        # 0. RANDOMLY SELECT A FOV
        # ------------------------------------------------
        h5_idx = np.random.choice(self.cellprops['h5_idx'])
        fov = np.random.choice(self.cellprops[self.cellprops['h5_idx'] == h5_idx]['fov'])

        # Extract data
        gr_raw  = self.images[h5_idx][fov, 0, GR_Channel, ...]  # shape (z,y,x)
        nuc_raw = self.images[h5_idx][fov, 0, Nuc_Channel, ...]

        gr_mask  = self.masks[h5_idx][fov, 0, GR_Channel, ...]
        nuc_mask = self.masks[h5_idx][fov, 0, Nuc_Channel, ...]

        # Pseudo-cyto = GR mask - nuc mask
        pseudo_cyto_3d = gr_mask - nuc_mask

        # Illumination for GR
        gr_illum = self.illumination_profiles[GR_Channel]  # (y, x)
        gr_corrected = self.corrected_IL_profile[GR_Channel]  # (y, x)

        # ------------------------------------------------
        # 1. ILLUMINATION CORRECTION
        # ------------------------------------------------
        fig1, axs1 = plt.subplots(1, 4, figsize=(12, 4))
        for ax in axs1:
            ax.axis("off")

        # Make 2D for display if needed (max projection)
        if gr_raw.ndim == 3:
            gr_raw_2d = np.max(gr_raw, axis=0)
        else:
            gr_raw_2d = gr_raw

        axs1[0].imshow(gr_raw_2d, cmap='gray')
        axs1[0].set_title("GR Raw")

        axs1[1].imshow(gr_illum, cmap='gray')
        axs1[1].set_title("GR Illumination")
        axs1[1].contour(self.illumination_profiles[GR_Channel], levels=10, colors='white', linewidths=0.5)

        # Correct the image
        epsilon = 1e-6
        gr_corrected = gr_raw / (gr_illum + epsilon)

        if gr_corrected.ndim == 3:
            gr_corrected_2d = np.max(gr_corrected, axis=0)
        else:
            gr_corrected_2d = gr_corrected
        axs1[2].imshow(gr_corrected_2d, cmap='gray')
        axs1[2].set_title("GR Corrected")

        axs1[3].imshow(self.corrected_IL_profile[GR_Channel], cmap='gray')
        axs1[3].set_title("Corrected Profile")
        axs1[3].contour(self.corrected_IL_profile[GR_Channel], levels=10, colors='white', linewidths=0.5)

        plt.tight_layout()
        plt.show()

        # ------------------------------------------------
        # 2. SEGMENTATION OVERLAYS
        # ------------------------------------------------
        fig2, axs2 = plt.subplots(1, 2, figsize=(8, 4))
        for ax in axs2:
            ax.axis("off")

        # nuc raw, nuc mask
        if nuc_raw.ndim == 3:
            nuc_raw_2d = np.max(nuc_raw, axis=0)
        else:
            nuc_raw_2d = nuc_raw
        if nuc_mask.ndim == 3:
            nuc_mask_2d = np.max(nuc_mask, axis=0)
        else:
            nuc_mask_2d = nuc_mask

        axs2[0].imshow(nuc_raw_2d, cmap='gray')
        axs2[0].imshow(nuc_mask_2d, cmap='jet', alpha=0.6)
        axs2[0].set_title("Nuc Raw + Nuc Mask")

        # gr raw, pseudo-cyto
        if gr_raw.ndim == 3:
            gr_raw_2d = np.max(gr_raw, axis=0)
        else:
            gr_raw_2d = gr_raw
        if pseudo_cyto_3d.ndim == 3:
            pseudo_cyto_2d = np.max(pseudo_cyto_3d, axis=0)
        else:
            pseudo_cyto_2d = pseudo_cyto_3d

        axs2[1].imshow(gr_raw_2d, cmap='gray')
        axs2[1].imshow(pseudo_cyto_2d, cmap='jet', alpha=0.6)
        axs2[1].set_title("GR Raw + Pseudo-cyto")

        plt.tight_layout()
        plt.show()

        # ------------------------------------------------
        # 3. INTENSITY HISTOGRAMS (use Dask's da.histogram)
        # ------------------------------------------------
        fig3, axs3 = plt.subplots(1, 2, figsize=(8, 4))

        #
        # 3a. GR Raw histogram
        #
        # If gr_raw is a NumPy array, wrap it in dask; if it's already dask, keep as is
        if not isinstance(gr_raw, da.Array):
            gr_raw_da = da.from_array(gr_raw)
        else:
            gr_raw_da = gr_raw

        # Compute min/max to define histogram range
        raw_min, raw_max = gr_raw_da.min(), gr_raw_da.max()
        if hasattr(raw_min, "compute"):
            raw_min = raw_min.compute()
        if hasattr(raw_max, "compute"):
            raw_max = raw_max.compute()

        hist_raw, bins_raw = da.histogram(gr_raw_da, bins=100, range=(raw_min, raw_max))

        # Conditionally compute
        if hasattr(hist_raw, "compute"):
            hist_raw = hist_raw.compute()
        if hasattr(bins_raw, "compute"):
            bins_raw = bins_raw.compute()

        # Now they're NumPy arrays
        width_raw = bins_raw[1:] - bins_raw[:-1]
        axs3[0].bar(bins_raw[:-1], hist_raw, width=width_raw, color='orange', alpha=0.7)
        axs3[0].set_title("GR Raw Histogram")

        #
        # 3b. GR Corrected histogram
        #
        if not isinstance(gr_corrected, da.Array):
            gr_corrected_da = da.from_array(gr_corrected)
        else:
            gr_corrected_da = gr_corrected

        corr_min, corr_max = gr_corrected_da.min(), gr_corrected_da.max()
        if hasattr(corr_min, "compute"):
            corr_min = corr_min.compute()
        if hasattr(corr_max, "compute"):
            corr_max = corr_max.compute()

        hist_corr, bins_corr = da.histogram(gr_corrected_da, bins=100, range=(corr_min, corr_max))

        if hasattr(hist_corr, "compute"):
            hist_corr = hist_corr.compute()
        if hasattr(bins_corr, "compute"):
            bins_corr = bins_corr.compute()

        width_corr = bins_corr[1:] - bins_corr[:-1]
        axs3[1].bar(bins_corr[:-1], hist_corr, width=width_corr, color='green', alpha=0.7)
        axs3[1].set_title("GR Corrected Histogram")

        plt.tight_layout()
        plt.show()

        # ------------------------------------------------
        # 4. RANDOM CELL & VALIDATE
        # ------------------------------------------------
        possible_labels = self.cellprops[
            (self.cellprops['h5_idx'] == h5_idx) &
            (self.cellprops['fov'] == fov)
        ]['nuc_label'].unique()

        if len(possible_labels) > 0:
            cell_label = np.random.choice(possible_labels)
            row = self.cellprops[
                (self.cellprops['h5_idx'] == h5_idx) &
                (self.cellprops['fov'] == fov) &
                (self.cellprops['nuc_label'] == cell_label)
            ]

            # Attempt to read bounding box
            try:
                row_min = int(row['cell_bbox-0'])
                col_min = int(row['cell_bbox-1'])
                row_max = int(row['cell_bbox-2'])
                col_max = int(row['cell_bbox-3'])
            except:
                print(f"Bounding box info missing for cell_label={cell_label}")
                row_min, col_min, row_max, col_max = 0, 0, 0, 0

            fig4, axs4 = plt.subplots(1, 2, figsize=(8, 4))
            for ax in axs4:
                ax.axis('off')

            # Convert corrected GR to 2D if needed
            if gr_corrected.ndim == 3:
                gr_corrected_2d = np.max(gr_corrected, axis=0)
            else:
                gr_corrected_2d = gr_corrected

            # Convert GR mask to boolean for chosen label
            if gr_mask.ndim == 3:
                gr_mask_3d = (gr_mask == cell_label)
                gr_mask_2d = np.max(gr_mask_3d, axis=0)
            else:
                gr_mask_2d = (gr_mask == cell_label)

            # Full FOV
            if isinstance(gr_corrected_2d, da.Array):
                gr_corrected_2d = gr_corrected_2d.compute()
            axs4[0].imshow(gr_corrected_2d, cmap='gray')
            axs4[0].imshow(gr_mask_2d, cmap='jet', alpha=0.6)
            axs4[0].set_title("Random Cell Overlay")

            # Zoomed bounding box
            if (row_min < row_max) and (col_min < col_max):
                crop_img  = gr_corrected_2d[row_min:row_max, col_min:col_max]
                crop_mask = gr_mask_2d[row_min:row_max, col_min:col_max]

                axs4[1].imshow(crop_img, cmap='gray')
                axs4[1].imshow(crop_mask, cmap='jet', alpha=0.6)
            axs4[1].set_title("Cell Bounding Box")

            plt.tight_layout()
            plt.show()

            # Validation
            try:
                if nuc_mask.ndim == 3:
                    nuc_mask_3d = (nuc_mask == cell_label)
                    nuc_mask_2d = np.max(nuc_mask_3d, axis=0)
                else:
                    nuc_mask_2d = (nuc_mask == cell_label)

                print(
                    self.validate(
                        gr_mask_2d, 
                        nuc_mask_2d, 
                        gr_corrected_2d, 
                        cell_label, 
                        row
                    )
                )
            except Exception as e:
                print(f"Validate failed for cell_label={cell_label}: {e}")
        else:
            print("No valid labels found in this FOV.")

    def validate(self, cell_mask, nuc_mask, image, label, measurements):
        """
        Validates the calculated measurements (area, intensity, etc.) of the cell 
        and nucleus against the provided measurements, returning a DataFrame.
        """
        # If these arrays are still dask, compute them:
        if isinstance(cell_mask, da.Array):
            cell_mask = cell_mask.compute()
        if isinstance(nuc_mask, da.Array):
            nuc_mask = nuc_mask.compute()
        if isinstance(image, da.Array):
            image = image.compute()

        cell_area = np.sum(cell_mask)
        nuc_area  = np.sum(nuc_mask)

        cell_pixels = image[cell_mask]
        nuc_pixels  = image[nuc_mask]

        cell_avgInt = np.mean(cell_pixels)
        nuc_avgInt  = np.mean(nuc_pixels)
        cell_stdInt = np.std(cell_pixels)
        nuc_stdInt  = np.std(nuc_pixels)
        cell_maxInt = np.max(cell_pixels)
        nuc_maxInt  = np.max(nuc_pixels)
        cell_minInt = np.min(cell_pixels)
        nuc_minInt  = np.min(nuc_pixels)

        # Helper to fetch measurement from `row`
        def get_float(key):
            return float(measurements.get(key, np.nan))

        def close_enough(a, b):
            return np.isclose(a, b, atol=1e-3)

        cyto_area = cell_area - nuc_area

        rows = [
            ['cell_area',     cell_area,     get_float('cell_area'),               close_enough(cell_area, get_float('cell_area'))],
            ['nuc_area',      nuc_area,      get_float('nuc_area'),                close_enough(nuc_area,  get_float('nuc_area'))],
            ['cyto_area',     cyto_area,     get_float('cyto_area'),               close_enough(cyto_area, get_float('cyto_area'))],

            ['cell_avgInt',   cell_avgInt,   get_float('cell_intensity_mean-0'),   close_enough(cell_avgInt, get_float('cell_intensity_mean-0'))],
            ['nuc_avgInt',    nuc_avgInt,    get_float('nuc_intensity_mean-0'),    close_enough(nuc_avgInt,  get_float('nuc_intensity_mean-0'))],

            ['cell_stdInt',   cell_stdInt,   get_float('cell_intensity_std-0'),    close_enough(cell_stdInt, get_float('cell_intensity_std-0'))],
            ['nuc_stdInt',    nuc_stdInt,    get_float('nuc_intensity_std-0'),     close_enough(nuc_stdInt,  get_float('nuc_intensity_std-0'))],

            ['cell_maxInt',   cell_maxInt,   get_float('cell_intensity_max-0'),    close_enough(cell_maxInt, get_float('cell_intensity_max-0'))],
            ['nuc_maxInt',    nuc_maxInt,    get_float('nuc_intensity_max-0'),     close_enough(nuc_maxInt,  get_float('nuc_intensity_max-0'))],

            ['cell_minInt',   cell_minInt,   get_float('cell_intensity_min-0'),    close_enough(cell_minInt, get_float('cell_intensity_min-0'))],
            ['nuc_minInt',    nuc_minInt,    get_float('nuc_intensity_min-0'),     close_enough(nuc_minInt,  get_float('nuc_intensity_min-0'))],
        ]

        return pd.DataFrame(
            rows,
            columns=['measurement', 'numpy calculated', 'step calculated', 'are close?']
        )        

if __name__ == '__main__':
    ana = AnalysisManager(r'\\munsky-nas.engr.colostate.edu\share\smFISH_images\Eric_smFISH_images\20220225\DUSP1_Dex_0min_20220224\DUSP1_Dex_0min_20220224.h5')
    print(ana.location)
    print(ana.h5_files)
    ana.list_analysis_names()
    ana.select_analysis()
    ana.list_datasets()
    print(ana.select_datasets('df_spotresults'))


