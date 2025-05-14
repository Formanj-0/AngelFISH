"""
Analysis_GR.py

This module contains classes for GR (Glucocorticoid Receptor) data analysis and visualization:

1. AnalysisManager:
    - Manages HDF5 file access.
    - Extracts file paths from a log directory (if no direct locations are provided).
    - Provides methods to select an analysis (by name) and load datasets from HDF5 files.
    - Saves datasets as CSV.

2. GR_Confirmation:
    - Processes cell properties and image data for GR ICC analysis.
    - Applies illumination correction using provided illumination profiles.
    - Validates segmentation masks and measurements for nucleus and cytoplasm.
    - Provides visualization routines for:
         • Illumination correction (raw, profile, corrected, corrected profile).
         • Segmentation overlays (nucleus and pseudo-cytoplasm).
         • Intensity histograms for raw and corrected GR channels.
         • Random cell validation with bounding box overlays.
    - Supports reproducibility with optional random seed.

3. DisplayManager:
    - Provides visualization routines.
    - Safely retrieves images and masks from HDF5 files.
    - Displays gating overlays (optionally highlighting a specific cell).
    - Displays cell crops filtered by an assigned total expression group.
    - Separately displays random cell crops for cells with transcription sites (TS) and with foci.
    - Contains a method to assign expression groups based on total mRNA (num_nuc_spots + num_cyto_spots) using the same bounds as for nuclear.

Author: Eric Ron
Date: March 12 2025
"""
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
from datetime import datetime
import sys
import traceback

class AnalysisManager:
    """
    This class is made to select data for further analysis.
    It provides methods to load, filter, and retrieve datasets from HDF5 files.
    """
    def __init__(self, location:Union[str, list[str]]=None, log_location:str=None, mac:bool=False, mode='r'):
        # given:
        # h5 locations
        #   give me a location
        #   give me a list of locations
        #   give me none -> got to here and display these \\munsky-nas.engr.colostate.edu\share\Users\Jack\All_Analysis
        self.mode = mode
        if location is None: # TODO make these if statement better
            self.get_locations(log_location, mac)
        elif isinstance(location, str):
            self.location = [location]
        elif isinstance(location, list): # TODO make sure its a list of str
            self.location = location
        else:
            raise ValueError('Location is not properly defined')
        
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
        self.open()
        self._find_analysis_names()
        self._filter_on_date(date_range)
        self._filter_on_name(analysis_name)
        self._filter_locations()
        self._handle_duplicates()
        self.close()

    def list_analysis_names(self):
        self.open()
        self._find_analysis_names()
        for name in self.analysis_names:
            print(name)
        self.close()
        return self.analysis_names

    def select_datasets(self, dataset_name, dtype=None) -> list:
        """ Safely loads datasets from HDF5 files with strict error handling. """

        self.open()  # Ensure setup is complete before file operations

        if not hasattr(self, 'analysis_names'):
            print('ERROR: No analysis selected. Please check your setup.')
            return []

        is_df = False
        is_array = False
        list_df = []
        list_arrays = []

        for i, file_path in enumerate(self.location):
            if not os.path.exists(file_path):
                print(f"ERROR: File does not exist: {file_path}")
                sys.exit(1)  # Immediately exit if a file is missing

            print(f"Opening file: {file_path}")

            try:
                with h5py.File(file_path, 'r') as f:
                    dataset_key = f"{self.analysis_names[i]}/{dataset_name}"
                    
                    if dataset_key not in f:
                        print(f"WARNING: Dataset '{dataset_name}' not found in {file_path}. Skipping.")
                        continue

                    if dtype == 'dataframe':
                        print(f"Reading DataFrame from: {file_path} -> {dataset_key}")
                        df = pd.read_hdf(f.filename, dataset_key)
                        df['h5_idx'] = i
                        list_df.append(df)
                        is_df = True

                    elif dtype == 'array':
                        print(f"Reading Array from: {file_path} -> {dataset_key}")
                        array = da.from_array(f[dataset_key])
                        list_arrays.append(array)
                        is_array = True

                    else:
                        raise ValueError(f"ERROR: Unknown data type '{dtype}' for {dataset_name}. Accepted types: 'dataframe', 'array'.")

            except OSError as e:
                print(f"\nCRITICAL ERROR: Unable to read {file_path}")
                print(f"HDF5 Error: {e}")
                print(f"Dataset: {dataset_name}")
                print(f"File: {file_path}")
                traceback.print_exc()  # Print full error traceback for debugging
                sys.exit(1)  # Immediately exit to prevent proceeding with bad data

        if is_df:
            print(f"Successfully loaded {len(list_df)} DataFrames. Merging...")
            return pd.concat(list_df, axis=0, ignore_index=True)

        if is_array:
            print(f"Successfully loaded {len(list_arrays)} arrays.")
            return list_arrays

        print(f"ERROR: No valid data found for dataset: {dataset_name}.")
        sys.exit(1)  # If no data is loaded, do not proceed blindly

    def list_datasets(self):
        if hasattr(self, 'analysis_names'):
            keys = []
            for i, l in enumerate(self.location):
                with h5py.File(l, 'r') as h:
                    keys.append(list(h[self.analysis_names[i]].keys()))
            flat_keys = [item for sublist in keys for item in sublist]
            print(set(flat_keys))
        else:
            print('select an analysis')

    def _filter_on_name(self, analysis_name):
        # self.analysis_names = [s.split('_')[1] for s in self.analysis_names]
        self.names = ['_'.join(s.split('_')[1:-1]) for s in self.analysis_names]
        if analysis_name is not None:
            self.analysis_names = [self.analysis_names[i] for i,s in enumerate(self.names) if analysis_name in s]

    def _filter_on_date(self, date_range: tuple):
        """
        Filters self.analysis_names and corresponding HDF5 file locations based on a given date range.
        
        Parameters:
            date_range (tuple): A tuple (start_date, end_date) where dates are in the format 'YYYY-MM-DD'.
        """
        if date_range is not None:
            # Parse start and end dates
            start_date, end_date = date_range
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.strptime(end_date, '%Y-%m-%d')

            # Filter locations based on their dates in the file paths or analysis names
            filtered_locations = []
            filtered_analysis_names = []

            for loc, analysis in zip(self.location, self.analysis_names):
                # Extract the date from the file path or analysis name (e.g., "20220707")
                try:
                    date_str = analysis.split('_')[-1]  # Assuming the date is part of the analysis name
                    file_date = datetime.strptime(date_str, '%Y-%m-%d')
                except ValueError:
                    # Handle cases where the date is not properly formatted
                    continue
                
                # Check if the date is within the range
                if start_date <= file_date <= end_date:
                    filtered_locations.append(loc)
                    filtered_analysis_names.append(analysis)
            
            # Update self.location and self.analysis_names with the filtered results
            self.location = filtered_locations
            self.analysis_names = filtered_analysis_names

            print(f"Filtered locations: {self.location}")
            print(f"Filtered analysis names: {self.analysis_names}")

    def _filter_locations(self):
        # select data sets with self.data, and self.dataset
        temp_locs = []
        temp_names = []
        for h_idx, h in enumerate(self.h5_files):
            for n in self.analysis_names:
                if n in h.keys():
                    temp_locs.append(h.filename)
                    temp_names.append(n)
        self.location = temp_locs
        self.analysis_names = temp_names

    def _handle_duplicates(self): # requires user input
        pass
        # TODO: check if h5 has multiple analysis in it

    def _find_analysis_names(self):
        self.analysis_names = []
        for h in self.h5_files:
            self.analysis_names.append(list(h.keys()))
        self.analysis_names = set([dataset for sublist in self.analysis_names for dataset in sublist])
        self.analysis_names = [d for d in self.analysis_names if 'Analysis' in d]

    def open(self):
        self.h5_files = []
        for l in self.location:
            if l not in [h.filename for h in self.h5_files]:
                self.h5_files.append(h5py.File(l, self.mode))
    
    def get_images_and_masks(self):
        self.raw_images = []
        self.masks = []

        for l in self.location:
            # with h5py.File(l, 'r') as h:
            self.h5_files.append(h5py.File(l))
            self.raw_images.append(da.from_array(self.h5_files[-1]['raw_images']))
            self.masks.append(da.from_array(self.h5_files[-1]['masks']))
        return self.raw_images, self.masks
    
    def close(self):
        for h in self.h5_files:
            h.flush()
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


    def __init__(self, am, seed=None):
        super().__init__(am, seed)
        self.fov = None
        self.h5_idx = None
        self.cell_label = None

        # Weighted threshold new attribute:
        self.snr_cutoff_2_5 = None  # 20th percentile in [2,5)

        # We store the chosen_spot and chosen_ts for re-use
        self.chosen_spot = None
        self.chosen_ts   = None 
        self.chosen_foci = None

    ############################################################
    # Data loading and saving
    ############################################################
    def get_data(self):
        """
        Loads spots, clusters, cellprops, cellspots from HDF5,
        plus images and masks.
        """
        self.spots     = self.am.select_datasets('spotresults', 'dataframe')
        self.cellspots  = self.am.select_datasets('cellresults', 'dataframe')
        self.cellprops    = self.am.select_datasets('cell_properties', 'dataframe')
        self.clusters  = self.am.select_datasets('clusterresults', 'dataframe')

        self.images, self.masks = self.am.get_images_and_masks()


    def display_gating(self):
        Cyto_Channel = 1
        required_columns = ['unique_cell_id']

        # Ensure each DataFrame has a unique cell id (uci)
        for df_name, df in [('spots', self.spots), ('cellprops', self.cellprops), ('clusters', self.clusters)]:
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"DataFrame '{df_name}' does not contain the required columns: {required_columns}")

        # Check for unique 'uci' in each DataFrame
        for df_name, df in [('cellprops', self.cellprops)]:
            if df['unique_cell_id'].duplicated().any():
                raise ValueError(f"DataFrame '{df_name}' contains duplicate 'unique_cell_id' values.")

        # Remove UCIs that are not present in all dataframes
        common_uci = set(self.cellprops['unique_cell_id']).intersection(
            self.spots['unique_cell_id'],
            # self.clusters['unique_cell_id'],
        )

        self.spots = self.spots[self.spots['unique_cell_id'].isin(common_uci)]
        self.cellprops = self.cellprops[self.cellprops['unique_cell_id'].isin(common_uci)]
        self.clusters = self.clusters[self.clusters['unique_cell_id'].isin(common_uci)]

        # select a h5_index at random
        id = np.random.choice(len(self.spots))
        spot_row = self.spots.iloc[id]
        h5_idx, fov, nas_location = spot_row['h5_idx'], spot_row['fov'], spot_row['NAS_location']
        self.h5_idx = h5_idx
        self.fov = fov
        print(f'Selected H5 Index: {h5_idx}')
        print(f'Nas Location: {nas_location}')
        print(f'FOV: {fov}' )

        img, mask = self.images[h5_idx][fov, 0, :, :, :, :], self.masks[h5_idx][fov, 0, :, :, :, :]
        mask = np.max(mask, axis=1) # This should make czyx to cyz
        img = np.max(img, axis=1)
        
        # Compute only if they are Dask arrays
        if isinstance(mask, da.Array):
            mask = mask.compute()
        if isinstance(img, da.Array):
            img = img.compute()

        # Debugging: Explicitly check what `img` is
        print("DEBUG: Type of img before percentile:", type(img))
        if hasattr(img, "shape"):
            print("DEBUG: Shape of img:", img.shape)
        else:
            print("DEBUG: img has no shape attribute")

        # Rescale the image exposure
        p1, p99 = np.percentile(img, (1, 99))
        img = exposure.rescale_intensity(
            img, in_range=(p1, p99), out_range=(0, 1)
        )

        # Find data frames of UCIs that have h5_idx, fov, nas_location
        spots_frame = self.spots[
            (self.spots['h5_idx'] == h5_idx) &
            (self.spots['fov'] == fov) &
            (self.spots['NAS_location'] == nas_location)
        ]

        cellprops_frame = self.cellprops[
            (self.cellprops['h5_idx'] == h5_idx) &
            (self.cellprops['fov'] == fov) &
            (self.cellprops['NAS_location'] == nas_location)
        ]

        clusters_frame = self.clusters[
            (self.clusters['h5_idx'] == h5_idx) &
            (self.clusters['fov'] == fov) &
            (self.clusters['NAS_location'] == nas_location)
        ]

        print(f"Spots DataFrame: {spots_frame.shape}")
        print(f"Cell Properties DataFrame: {cellprops_frame.shape}")
        print(f"Clusters DataFrame: {clusters_frame.shape}")

        # Compute mask if it's a Dask array
        if isinstance(mask, da.Array):
            mask = mask.compute()

        # Identify which cell_labels appear in cellprops => "kept"
        cell_labels_in_df = set(cellprops_frame['cell_label'].unique())

        # Identify which appear in the mask
        cell_labels_in_mask = set(np.unique(mask))
        if 0 in cell_labels_in_mask:
            cell_labels_in_mask.remove(0)

        # Colors
        kept_colors = list(mcolors.TABLEAU_COLORS.values())  # distinct non-red
        removed_color = 'red'

        # Show gating overlay
        fig, ax = plt.subplots(figsize=(10, 10))

        # Ensure img is computed if necessary
        if isinstance(img, da.Array):
            img = img.compute()

        ax.imshow(img[Cyto_Channel, :, :], cmap='gray')

        # A) Plot the "kept" cell outlines
        color_map = {
            cell_label: kept_colors[i % len(kept_colors)]
            for i, cell_label in enumerate(cell_labels_in_df)
        }

        for cell_label in cell_labels_in_df:
            cell_mask = (mask == cell_label)

            # Ensure `cell_mask` is computed only if needed
            if isinstance(cell_mask, da.Array):
                cell_mask = cell_mask.compute()

            contours = find_contours(cell_mask[Cyto_Channel, :, :], 0.5)
            if contours:
                largest_contour = max(contours, key=lambda x: x.shape[0])
                ax.plot(largest_contour[:, 1], largest_contour[:, 0],
                        linewidth=2, color=color_map[cell_label],
                        label=f'Cell {cell_label}')

        # B) Plot "discarded" cell outlines
        cell_labels_not_in_df = cell_labels_in_mask - cell_labels_in_df

        for cell_label in cell_labels_not_in_df:
            cell_mask = (mask == cell_label)

            # Ensure `cell_mask` is computed only if needed
            if isinstance(cell_mask, da.Array):
                cell_mask = cell_mask.compute()

            contours = find_contours(cell_mask[Cyto_Channel, :, :], 0.5)
            if contours:
                largest_contour = max(contours, key=lambda x: x.shape[0])
                ax.plot(largest_contour[:, 1], largest_contour[:, 0],
                        color=removed_color, linewidth=2, linestyle='dashed',
                        label=f'Removed: Cell {cell_label}')

        plt.legend()
        plt.show()
 
    def save_data(self, location):
        """
        Saves the DataFrames to CSV.
        """
        self.spots.to_csv(os.path.join(location, 'spots.csv'), index=False)
        self.clusters.to_csv(os.path.join(location, 'clusters.csv'), index=False)
        self.cellprops.to_csv(os.path.join(location, 'cellprops.csv'), index=False)
        self.cellspots.to_csv(os.path.join(location, 'cellspots.csv'), index=False)

    ############################################################
    # Weighted SNR: snr < 2 => discard, [2,5) => remove bottom 20%, keep >=5
    ############################################################
    def assign_revised_weighted_threshold(self):
        """
        1) Find 20th percentile of SNR for spots in [2,5).
        2) Discard if snr < 2 or below that percentile in [2,5).
        3) Otherwise keep.
        Returns the numeric 20th-percentile cutoff for [2,5).
        """
        # Identify spots in [2,5)
        mask_2_5 = (self.spots['snr'] >= 2) & (self.spots['snr'] < 5)
        spots_2_5 = self.spots[mask_2_5]

        if not spots_2_5.empty:
            self.snr_cutoff_2_5 = np.percentile(spots_2_5['snr'], 20)
        else:
            self.snr_cutoff_2_5 = 2.0  # fallback

        def keep_spot(row):
            snr = row['snr']
            if snr < 2:
                return False
            elif 2 <= snr < 5:
                return snr >= self.snr_cutoff_2_5
            else:
                return True  # snr >= 5 => keep

        self.spots['keep_wsnr'] = self.spots.apply(keep_spot, axis=1)

        return self.snr_cutoff_2_5

    ###########################################################################
    # 0) Choose the FOV / cell such that the cell definitely has a TS (if possible)
    ###########################################################################
    def _pick_random_FOV_and_cell_with_TS(self):
        """
        1) Picks a random h5_idx, then a random FOV among it.
        2) Tries to pick a cell with:
        - TS (is_nuc=1), Foci (is_nuc=-1), and Spot (priority).
        - If no such cell, tries TS and Spot.
        - If no such cell, tries Spot only.
        - If no such cell, picks any valid cell.
        """
        # Step 1: Randomly pick h5_idx and FOV
        self.h5_idx = np.random.choice(self.spots['h5_idx'])
        possible_fovs = self.spots[self.spots['h5_idx'] == self.h5_idx]['fov'].unique()
        self.fov = np.random.choice(possible_fovs)

        # Step 2: Find all valid cells in this FOV
        valid_cells = self.cellprops[
            (self.cellprops['h5_idx'] == self.h5_idx) &
            (self.cellprops['fov'] == self.fov) &
            (self.cellprops['cell_label'] != 0)
        ]['cell_label'].unique()

        if len(valid_cells) == 0:
            print(f"No valid cells in FOV={self.fov}. Aborting.")
            return

        # Step 3: Find cells with TS (is_nuc == 1)
        cells_with_TS = self.clusters[
            (self.clusters['h5_idx'] == self.h5_idx) &
            (self.clusters['fov'] == self.fov) &
            (self.clusters['is_nuc'] == 1)
        ]['cell_label'].unique()

        # Step 4: Find cells with Foci (is_nuc == -1)
        cells_with_Foci = self.clusters[
            (self.clusters['h5_idx'] == self.h5_idx) &
            (self.clusters['fov'] == self.fov) &
            (self.clusters['is_nuc'] == 0)
        ]['cell_label'].unique()

        # Step 5: Find cells with Spot
        cells_with_Spot = self.spots[
            (self.spots['h5_idx'] == self.h5_idx) &
            (self.spots['fov'] == self.fov)
        ]['cell_label'].unique()

        # Step 6: Apply selection logic
        # Priority: TS + Foci + Spot > TS + Spot > Spot > Any valid cell
        cells_with_TS_Foci_Spot = np.intersect1d(np.intersect1d(cells_with_TS, cells_with_Foci), cells_with_Spot)
        cells_with_TS_Spot = np.intersect1d(cells_with_TS, cells_with_Spot)
        cells_with_Spot_only = cells_with_Spot

        if len(cells_with_TS_Foci_Spot) > 0:
            self.cell_label = np.random.choice(cells_with_TS_Foci_Spot)
            print(f"Chose cell_label={self.cell_label} with TS, Foci, and Spot (FOV={self.fov}).")
        elif len(cells_with_TS_Spot) > 0:
            self.cell_label = np.random.choice(cells_with_TS_Spot)
            print(f"Chose cell_label={self.cell_label} with TS and Spot (FOV={self.fov}).")
        elif len(cells_with_Spot_only) > 0:
            self.cell_label = np.random.choice(cells_with_Spot_only)
            print(f"Chose cell_label={self.cell_label} with Spot only (FOV={self.fov}).")
        else:
            # Final fallback: pick any valid cell
            self.cell_label = np.random.choice(valid_cells)
            print(f"No cell with TS, Foci, or Spot => picked random cell_label={self.cell_label} (FOV={self.fov}).")

    ############################################################
    # 1) Full-FOV segmentation (nucleus & cytoplasm)
    ############################################################
    def _display_full_fov_segmentation(self, cytoChannel, nucChannel):
        """
        Shows nucleus channel + mask (left),
        cytoplasm channel + mask (right).
        """
        if self.h5_idx is None or self.fov is None:
            raise ValueError("h5_idx or fov is not set. Cannot display segmentation.")

        # self.fov was chosen already, so now we just read & visualize it
        img_nuc = self.images[self.h5_idx][self.fov, 0, nucChannel, :, :, :].compute()
        img_cyto = self.images[self.h5_idx][self.fov, 0, cytoChannel, :, :, :].compute()
        mask_nuc = self.masks[self.h5_idx][self.fov, 0, nucChannel, :, :, :].compute()
        mask_cyto = self.masks[self.h5_idx][self.fov, 0, cytoChannel, :, :, :].compute()

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

    ###########################################################################
    # 2) Zoom on cell with percentile-based contrast (modified to pick a spot not in TS)
    ###########################################################################
    def _display_zoom_on_cell(self, spotChannel, cytoChannel, nucChannel):
        """
        Zoom on bounding box, compute percentile-based contrast,
        then overlay:
          - gold circles for all 'regular' spots
          - chosen_spot in blue circle
          - TS in magenta arrows (with cluster size)
          - foci in cyan arrows
        """
        cdf = self.cellprops[
            (self.cellprops['h5_idx'] == self.h5_idx) &
            (self.cellprops['fov'] == self.fov) &
            (self.cellprops['cell_label'] == self.cell_label)
        ]
        if cdf.empty:
            print("Cell not found. Aborting cell zoom.")
            self.chosen_spot = None
            self.chosen_ts   = None
            self.chosen_foci = None
            return None

        row_min = int(cdf['cell_bbox-0'].iloc[0])
        col_min = int(cdf['cell_bbox-1'].iloc[0])
        row_max = int(cdf['cell_bbox-2'].iloc[0])
        col_max = int(cdf['cell_bbox-3'].iloc[0])

        # Spot channel -> 3D -> convert to 2D by max-projection
        img_spot_3d = self.images[self.h5_idx][self.fov, 0, spotChannel, :, :, :]
        img_spot_2d = np.max(img_spot_3d, axis=0)

        crop_spot_dask = img_spot_2d[row_min:row_max, col_min:col_max]
        crop_spot = crop_spot_dask.compute() if hasattr(crop_spot_dask, "compute") else crop_spot_dask

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

        crop_nucmask = mask_nuc_2d[row_min:row_max, col_min:col_max]
        crop_cytomask = mask_cyto_2d[row_min:row_max, col_min:col_max]

        # All spots in this cell
        cell_spots = self.spots[
            (self.spots['h5_idx'] == self.h5_idx) &
            (self.spots['fov'] == self.fov) &
            (self.spots['cell_label'] == self.cell_label)
        ]

        # Transcription sites in this cell
        cell_TS = self.clusters[
            (self.clusters['h5_idx'] == self.h5_idx) &
            (self.clusters['fov'] == self.fov) &
            (self.clusters['cell_label'] == self.cell_label) &
            (self.clusters['is_nuc'] == 1)
        ]

        # Foci in this cell
        cell_foci = self.clusters[
            (self.clusters['h5_idx'] == self.h5_idx) &
            (self.clusters['fov'] == self.fov) &
            (self.clusters['cell_label'] == self.cell_label) &
            (self.clusters['is_nuc'] == 0)
        ]

        if cell_foci.empty:
            print('No Foci in this cell')

        if cell_spots.empty:
            print("No spots in this cell.")
            self.chosen_spot = None
            self.chosen_ts   = None
            self.chosen_foci = None
        else:
            # Pick a TS if it exists, for reference in the next steps
            if not cell_TS.empty:
                # For example, pick one TS (largest)
                self.chosen_ts = cell_TS.iloc[0]
            else:
                self.chosen_ts = None
            
            # Pick a foci if it exists, for reference in the next steps
            if not cell_foci.empty:
                # For example, pick one foci
                self.chosen_foci = cell_foci.sample(1).iloc[0]
            else:
                self.chosen_foci = None
            # We want a spot that is NOT in any TS cluster
            if not cell_TS.empty:
                ts_cluster_ids = cell_TS['cluster_index'].unique()
                non_ts_spots = cell_spots[~cell_spots['cluster_index'].isin(ts_cluster_ids)]
                if non_ts_spots.empty:
                    # If *all* spots are in TS, fallback = just pick any
                    chosen_spot = cell_spots.sample(1).iloc[0]
                else:
                    chosen_spot = non_ts_spots.sample(1).iloc[0]
            else:
                # No TS => pick any spot
                chosen_spot = cell_spots.sample(1).iloc[0]

            self.chosen_spot = chosen_spot

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Left: raw channel + mask overlay
        axs[0].imshow(crop_spot_stretched, cmap='gray')
        axs[0].imshow(crop_nucmask, cmap='Blues', alpha=0.3)
        axs[0].imshow(crop_cytomask, cmap='Reds', alpha=0.3)
        axs[0].set_title(f"Cell {self.cell_label} - Masks overlay")

        # Right: contrast-stretched
        axs[1].imshow(crop_spot_stretched, cmap='gray')
        axs[1].imshow(crop_nucmask, cmap='Blues', alpha=0.2)
        axs[1].imshow(crop_cytomask, cmap='Reds', alpha=0.2)
        axs[1].set_title(f"Cell {self.cell_label} - Stretched Spot Channel")

        dx, dy = col_min, row_min

        # Draw all spots in gold circles
        for _, spot in cell_spots.iterrows():
            sx = spot['x_px'] - dx
            sy = spot['y_px'] - dy
            # if it's the chosen spot => color in blue
            if self.chosen_spot is not None and (spot['x_px'] == self.chosen_spot['x_px']) and (spot['y_px'] == self.chosen_spot['y_px']):
                draw_spot_circle(axs[1], sx, sy, radius=4, color='blue')
            else:
                draw_spot_circle(axs[1], sx, sy, radius=4, color='gold')

        # Mark TS in magenta arrows
        for _, tsrow in cell_TS.iterrows():
            # If your cluster DF has centroid x_px,y_px => we can arrow that
            sx = tsrow['x_px'] - dx
            sy = tsrow['y_px'] - dy
            draw_spot_arrow(axs[1], sx, sy, offset=-10, color='magenta')
            cluster_size = getattr(tsrow, 'nb_spots', 1)  # fallback
            axs[1].text(sx - 12, sy, f"{cluster_size}", color='magenta', fontsize=10)

        # Mark foci in cyan arrows
        for _, frow in cell_foci.iterrows():
            sx = frow['x_px'] - dx
            sy = frow['y_px'] - dy
            draw_spot_arrow(axs[1], sx, sy, offset=-10, color='cyan')
            foci_size = getattr(frow, 'nb_spots', 1)
            axs[1].text(sx - 12, sy, f'{foci_size}', color='cyan', fontsize=10)

        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

        return self.chosen_spot, self.chosen_ts, self.chosen_foci

    ###########################################################################
    # 3) Further zoom on the same single spot - now also highlight TS
    ###########################################################################
    def _display_zoom_on_one_spot(self, spotChannel):
        """
        Further zoom on the same chosen_spot. Also mark the chosen TS (magenta arrow).
        """
        if self.chosen_spot is None:
            print("No chosen_spot available for further zoom.")
            return

        chosen_spot = self.chosen_spot
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
        sub_img = sub_img_dask.compute() if hasattr(sub_img_dask, "compute") else sub_img_dask

        if sub_img.size > 0:
            p1, p99 = np.percentile(sub_img, (1, 99))
            sub_img_stretched = exposure.rescale_intensity(sub_img, in_range=(p1, p99), out_range=(0, 1))
        else:
            sub_img_stretched = sub_img

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(sub_img_stretched, cmap='gray')
        ax.set_title(f"Spot zoom (cell {self.cell_label})")

        # Mark chosen spot in blue
        draw_spot_circle(ax, pad, pad, radius=4, color='blue')

        # If we have a chosen_ts, draw arrow for it if it's in our patch
        if self.chosen_ts is not None:
            tx = int(self.chosen_ts['x_px'])
            ty = int(self.chosen_ts['y_px'])
            rx = tx - x1
            ry = ty - y1
            if (0 <= rx < (x2 - x1)) and (0 <= ry < (y2 - y1)):
                draw_spot_arrow(ax, rx, ry, offset=-10, color='magenta')
                # Optionally annotate TS size:
                csize = getattr(self.chosen_ts, 'nb_spots', 1)
                ax.text(rx - 12, ry, f"{csize}", color='magenta', fontsize=10)

        # If we have a chosen_foci, draw arrow for it if it's in our patch
        if self.chosen_foci is not None:
            tx = int(self.chosen_foci['x_px'])
            ty = int(self.chosen_foci['y_px'])
            rx = tx - x1
            ry = ty - y1
            if (0 <= rx < (x2 - x1)) and (0 <= ry < (y2 - y1)):
                draw_spot_arrow(ax, rx, ry, offset=-10, color='cyan')
                # Optionally annotate TS size:
                csize = getattr(self.chosen_foci, 'nb_spots', 1)
                ax.text(rx - 12, ry, f"{csize}", color='cyan', fontsize=10)

        # Mark other spots in gold if they are inside the patch
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
            # Avoid drawing chosen_spot in gold
            if (srow['x_px'] == chosen_spot['x_px']) and (srow['y_px'] == chosen_spot['y_px']):
                continue
            draw_spot_circle(ax, rx, ry, radius=3, color='gold')

        ax.axis("off")
        plt.tight_layout()
        plt.show()

    ###########################################################################
    # 4) SNR threshold figure – unchanged logic, but also draw TS in magenta
    ###########################################################################
    def _display_snr_thresholds(self, spotChannel, cytoChannel, nucChannel, thresholds=[0, 2, 3, 4]):
        """
        We simply add TS arrow in each subplot so that TS is always shown,
        ignoring SNR threshold for TS spots.
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

        # 2D max-projection
        img_spot_3d = self.images[self.h5_idx][self.fov, 0, spotChannel, :, :, :]
        img_spot_2d = np.max(img_spot_3d, axis=0)

        crop_spot_dask = img_spot_2d[row_min:row_max, col_min:col_max]
        crop_spot = crop_spot_dask.compute() if hasattr(crop_spot_dask, "compute") else crop_spot_dask

        if crop_spot.size > 0:
            p1, p99 = np.percentile(crop_spot, (1, 99))
            crop_spot_stretched = exposure.rescale_intensity(
                crop_spot, in_range=(p1, p99), out_range=(0, 1)
            )
        else:
            crop_spot_stretched = crop_spot

        mask_nuc_3d = self.masks[self.h5_idx][self.fov, 0, nucChannel, :, :, :]
        mask_cyto_3d = self.masks[self.h5_idx][self.fov, 0, cytoChannel, :, :, :]
        mask_nuc_2d = np.max(mask_nuc_3d, axis=0)
        mask_cyto_2d = np.max(mask_cyto_3d, axis=0)

        crop_nucmask = mask_nuc_2d[row_min:row_max, col_min:col_max]
        crop_cytomask = mask_cyto_2d[row_min:row_max, col_min:col_max]

        fig, axs = plt.subplots(len(thresholds), 2, figsize=(12, len(thresholds)*5))
        dx, dy = col_min, row_min

        cell_spots_all = self.spots[
            (self.spots['h5_idx'] == self.h5_idx) &
            (self.spots['fov'] == self.fov) &
            (self.spots['cell_label'] == self.cell_label)
        ]
        # TS in this cell
        cell_TS = self.clusters[
            (self.clusters['h5_idx'] == self.h5_idx) &
            (self.clusters['fov'] == self.fov) &
            (self.clusters['cell_label'] == self.cell_label) &
            (self.clusters['is_nuc'] == 1)
        ]
        # Foci in this cell
        cell_foci = self.clusters[
            (self.clusters['h5_idx'] == self.h5_idx) &
            (self.clusters['fov'] == self.fov) &
            (self.clusters['cell_label'] == self.cell_label) &
            (self.clusters['is_nuc'] == 0)
        ]
        # For coloring sub-threshold spots
        shades_of_red = [
            (1, 0.8, 0.8),
            (1, 0.6, 0.6),
            (1, 0.4, 0.4),
            (1, 0.2, 0.2)
        ]

        for i, thr in enumerate(thresholds):
            # Left
            axL = axs[i, 0]
            axL.imshow(crop_spot_stretched, cmap='gray')
            axL.imshow(crop_nucmask, cmap='Blues', alpha=0.2)
            axL.imshow(crop_cytomask, cmap='Reds', alpha=0.2)
            axL.set_title(f"SNR >= {thr} (Reds + Gold)")

            # Show TS in magenta
            for _, tsrow in cell_TS.iterrows():
                sx = tsrow['x_px'] - dx
                sy = tsrow['y_px'] - dy
                draw_spot_arrow(axL, sx, sy, offset=-5, color='magenta')

            # Draw below-thr spots in gradation of reds
            above_spots = cell_spots_all[cell_spots_all['snr'] >= thr]
            below_spots = cell_spots_all[cell_spots_all['snr'] < thr]
            shade_color = shades_of_red[i]
            # Mark below in red
            for _, spot in below_spots.iterrows():
                sx = spot['x_px'] - dx
                sy = spot['y_px'] - dy
                draw_spot_circle(axL, sx, sy, radius=4, color=shade_color)

            # Mark above threshold in gold
            for _, spot in above_spots.iterrows():
                sx = spot['x_px'] - dx
                sy = spot['y_px'] - dy
                draw_spot_circle(axL, sx, sy, radius=4, color='gold')
            axL.axis("off")

            # Right
            axR = axs[i, 1]
            axR.imshow(crop_spot_stretched, cmap='gray')
            axR.set_title(f"SNR >= {thr} (Detected Spots Only)")
            # again draw TS
            for _, tsrow in cell_TS.iterrows():
                sx = tsrow['x_px'] - dx
                sy = tsrow['y_px'] - dy
                draw_spot_arrow(axR, sx, sy, offset=-5, color='magenta')

            for _, spot in above_spots.iterrows():
                sx = spot['x_px'] - dx
                sy = spot['y_px'] - dy
                draw_spot_circle(axR, sx, sy, radius=4, color='gold')
            axR.axis("off")

        plt.tight_layout()
        plt.show()

    def _print_spots_before_after_threshold(self, thresholds):
        """
        Prints how many spots in the current cell are before/after each threshold.
        """
        cell_spots_all = self.spots[
            (self.spots['h5_idx'] == self.h5_idx) &
            (self.spots['fov'] == self.fov) &
            (self.spots['cell_label'] == self.cell_label)
        ]
        if cell_spots_all.empty:
            print("No spots in this cell for threshold printing.")
            return

        for thr in thresholds:
            n_before = len(cell_spots_all)
            n_passed = len(cell_spots_all[cell_spots_all['snr'] >= thr])
            print(f"Threshold SNR >= {thr}: {n_passed} kept / {n_before} total spots.")        

    ############################################################
    # 5) Histograms and scatter plots for SNR thresholds
    ############################################################
    def _display_threshold_histograms_scatter(self, thresholds):
        """
        For each threshold, we display histograms, and a scatter,
        but here let's show a single histogram & scatter for all spots in cell.
        """
        cell_spots_all = self.spots[
            (self.spots['h5_idx'] == self.h5_idx) &
            (self.spots['fov'] == self.fov) &
            (self.spots['cell_label'] == self.cell_label)
        ]
        if cell_spots_all.empty:
            print("No spots in cell for threshold histograms/scatter.")
            return

        fig, axs = plt.subplots(len(thresholds), 2, figsize=(18, 10))

        for i, thr in enumerate(thresholds):
            # Left: SNR histogram for spots above threshold
            cell_spots_thr = cell_spots_all[cell_spots_all['snr'] >= thr]
            axs[i, 0].hist(cell_spots_thr['signal'], bins=50, alpha=0.7, edgecolor='black')
            axs[i, 0].set_title(f"SNR >= {thr} Spot Intensity Histogram")
            axs[i, 0].set_xlabel("Intensity")
            axs[i, 0].set_ylabel("Count")

            # Right: Signal vs. SNR scatter
            axs[i, 1].scatter(cell_spots_thr['signal'], cell_spots_thr['snr'], s=8, alpha=0.6)
            axs[i, 1].set_title(f"SNR >= {thr} Signal vs. SNR")
            axs[i, 1].set_xlabel("Intensity")
            axs[i, 1].set_ylabel("SNR")
            axs[i, 1].set_xscale('log')
            axs[i, 1].set_yscale('log')
        plt.tight_layout()
        plt.show()

    ############################################################
    # 6) Further zoom on the same single spot for each threshold
    ############################################################
    def _display_further_zoom_for_thresholds(self, spotChannel, thresholds):
        """
        For each threshold T, we re-zoom around the SAME chosen_spot,
        color it gold if SNR >= T, else red.
        """
        if self.chosen_spot is None:
            print("No chosen spot for threshold-based further zoom.")
            return

        chosen_spot = self.chosen_spot
        sx = int(chosen_spot['x_px'])
        sy = int(chosen_spot['y_px'])

        pad = 15
        x1 = max(sx - pad, 0)
        x2 = sx + pad
        y1 = max(sy - pad, 0)
        y2 = sy + pad

        # 3D -> 2D
        img_spot_3d = self.images[self.h5_idx][self.fov, 0, spotChannel, :, :, :]
        img_spot_2d = np.max(img_spot_3d, axis=0)

        sub_img_dask = img_spot_2d[y1:y2, x1:x2]
        sub_img = sub_img_dask.compute() if hasattr(sub_img_dask, "compute") else sub_img_dask

        # We'll do one figure per threshold
        for thr in thresholds:
            if sub_img.size > 0:
                p1, p99 = np.percentile(sub_img, (1, 99))
                sub_img_stretched = exposure.rescale_intensity(
                    sub_img, in_range=(p1, p99), out_range=(0, 1)
                )
            else:
                sub_img_stretched = sub_img

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(sub_img_stretched, cmap='gray')
            ax.set_title(f"SNR >= {thr} Zoom (same spot)")

            # Color gold if SNR >= thr, else red
            if chosen_spot['snr'] >= thr:
                color = 'gold'
            else:
                color = 'red'
            draw_spot_circle(ax, pad, pad, radius=4, color=color)

            # Optionally, mark other spots in the patch if you want
            # or skip if you only want the chosen spot
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
                    # already drawn
                    continue
                # color them in gold/red based on threshold too if you wish
                color2 = 'gold' if srow['snr'] >= thr else 'red'
                draw_spot_circle(ax, rx, ry, radius=3, color=color2)

            ax.axis("off")
            plt.tight_layout()
            plt.show()

    ############################################################
    # 7) Single figure (1x3) for Weighted Approach
    ############################################################
    def _display_weighted_threshold_figure(self, spotChannel, cytoChannel, nucChannel):
        """
        1x3 subplots:
         A) cell w/ masks before weighted threshold (all gold)
         B) cell w/ masks after weighted threshold (gold=keep_wsnr, red=discard)
         C) cell w/out masks, only the kept in gold
        Using the same cell from self.fov/self.cell_label.
        """
        cdf = self.cellprops[
            (self.cellprops['h5_idx'] == self.h5_idx) &
            (self.cellprops['fov'] == self.fov) &
            (self.cellprops['cell_label'] == self.cell_label)
        ]
        if cdf.empty:
            print("No cell found for weighted threshold figure.")
            return

        row_min = int(cdf['cell_bbox-0'].iloc[0])
        col_min = int(cdf['cell_bbox-1'].iloc[0])
        row_max = int(cdf['cell_bbox-2'].iloc[0])
        col_max = int(cdf['cell_bbox-3'].iloc[0])

        # Convert 3D to 2D
        spot_3d = self.images[self.h5_idx][self.fov, 0, spotChannel, :, :, :]
        spot_2d = np.max(spot_3d, axis=0)
        crop_spot_dask = spot_2d[row_min:row_max, col_min:col_max]
        crop_spot = crop_spot_dask.compute() if hasattr(crop_spot_dask, 'compute') else crop_spot_dask

        if crop_spot.size > 0:
            p1, p99 = np.percentile(crop_spot, (1, 99))
            crop_spot_stretched = exposure.rescale_intensity(
                crop_spot, in_range=(p1, p99), out_range=(0, 1)
            )
        else:
            crop_spot_stretched = crop_spot

        # Nuc/Cyto
        mask_nuc_3d = self.masks[self.h5_idx][self.fov, 0, nucChannel, :, :, :]
        mask_cyto_3d = self.masks[self.h5_idx][self.fov, 0, cytoChannel, :, :, :]
        mask_nuc_2d = np.max(mask_nuc_3d, axis=0)
        mask_cyto_2d = np.max(mask_cyto_3d, axis=0)

        crop_nuc = mask_nuc_2d[row_min:row_max, col_min:col_max]
        crop_nucmask = crop_nuc.compute() if hasattr(crop_nuc, 'compute') else crop_nuc
        crop_cyto = mask_cyto_2d[row_min:row_max, col_min:col_max]
        crop_cytomask = crop_cyto.compute() if hasattr(crop_cyto, 'compute') else crop_cyto

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        dx, dy = col_min, row_min

        cell_spots_all = self.spots[
            (self.spots['h5_idx'] == self.h5_idx) &
            (self.spots['fov'] == self.fov) &
            (self.spots['cell_label'] == self.cell_label)
        ]

        # Subplot A) all gold (before Weighted threshold)
        axs[0].imshow(crop_spot_stretched, cmap='gray')
        axs[0].imshow(crop_nucmask, cmap='Blues', alpha=0.2)
        axs[0].imshow(crop_cytomask, cmap='Reds', alpha=0.2)
        axs[0].set_title("A) Before Weighted Threshold (all gold)")

        for _, spot in cell_spots_all.iterrows():
            sx = spot['x_px'] - dx
            sy = spot['y_px'] - dy
            draw_spot_circle(axs[0], sx, sy, radius=4, color='gold')
        axs[0].axis("off")

        # Subplot B) after Weighted threshold (with masks)
        axs[1].imshow(crop_spot_stretched, cmap='gray')
        axs[1].imshow(crop_nucmask, cmap='Blues', alpha=0.2)
        axs[1].imshow(crop_cytomask, cmap='Reds', alpha=0.2)
        axs[1].set_title("B) Weighted thr (gold=keep, red=discard)")

        kept_spots = cell_spots_all[cell_spots_all['keep_wsnr'] == True]
        discarded_spots = cell_spots_all[cell_spots_all['keep_wsnr'] == False]
        for _, spot in kept_spots.iterrows():
            sx = spot['x_px'] - dx
            sy = spot['y_px'] - dy
            draw_spot_circle(axs[1], sx, sy, radius=4, color='gold')
        for _, spot in discarded_spots.iterrows():
            sx = spot['x_px'] - dx
            sy = spot['y_px'] - dy
            draw_spot_circle(axs[1], sx, sy, radius=4, color='red')
        axs[1].axis("off")

        # Subplot C) after Weighted threshold w/out masks (just gold for kept)
        axs[2].imshow(crop_spot_stretched, cmap='gray')
        axs[2].set_title("C) Weighted thr, no masks (kept in gold)")
        for _, spot in kept_spots.iterrows():
            sx = spot['x_px'] - dx
            sy = spot['y_px'] - dy
            draw_spot_circle(axs[2], sx, sy, radius=4, color='gold')
        axs[2].axis("off")

        plt.tight_layout()
        plt.show()

        print(f"Weighted threshold => kept: {len(kept_spots)}, discarded: {len(discarded_spots)}")

    ############################################################
    # 8) Further zoom on the same single spot after Weighted threshold
    ############################################################
    def _display_weighted_zoom_on_spot(self, spotChannel):
        """
        Zoom on the SAME chosen_spot, coloring it gold if keep_wsnr==True, else red.
        """
        if self.chosen_spot is None:
            print("No chosen spot for weighted zoom.")
            return

        chosen_spot = self.chosen_spot
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
        sub_img = sub_img_dask.compute() if hasattr(sub_img_dask, 'compute') else sub_img_dask

        if sub_img.size > 0:
            p1, p99 = np.percentile(sub_img, (1, 99))
            sub_img_stretched = exposure.rescale_intensity(sub_img, in_range=(p1, p99), out_range=(0, 1))
        else:
            sub_img_stretched = sub_img

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(sub_img_stretched, cmap='gray')
        ax.set_title("Weighted Zoom: gold=keep, red=discard")

        # color the chosen spot based on keep_wsnr
        if chosen_spot['keep_wsnr']:
            color = 'gold'
        else:
            color = 'red'
        draw_spot_circle(ax, pad, pad, radius=3, color=color)

        # If you want to highlight other spots in patch similarly, do so:
        cell_spots_all = self.spots[
            (self.spots['h5_idx'] == self.h5_idx) &
            (self.spots['fov'] == self.fov) &
            (self.spots['cell_label'] == self.cell_label)
        ]
        for _, srow in cell_spots_all.iterrows():
            rx = srow['x_px'] - x1
            ry = srow['y_px'] - y1
            if (rx < 0 or ry < 0 or rx >= (x2 - x1) or ry >= (y2 - y1)):
                continue
            color2 = 'gold' if srow['keep_wsnr'] else 'red'
            # If it's the chosen spot again, we skip (already drawn)
            if (srow['x_px'] == chosen_spot['x_px'] and srow['y_px'] == chosen_spot['y_px']):
                continue
            draw_spot_circle(ax, rx, ry, radius=3, color=color2)

        ax.axis("off")
        plt.tight_layout()
        plt.show()
    ###########################################################################
    # 4b) Overview bar-plot by 'time': fraction of cells with 1,2,3,>=4 TS
    ###########################################################################
    def _display_ts_barplot_by_time(self):
        """
        For each 'time', display the fraction (percentage) of cells
        that have exactly 1, 2, 3, or >=4 TS.
        We consider all cells that appear in `cellprops`.
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # TS = is_nuc==1 in clusters df
        ts_df = self.clusters[self.clusters['is_nuc'] == 1]

        # Count how many TS per (time, fov, cell_label)
        ts_count_df = (
            ts_df.groupby(['h5_idx', 'time', 'fov', 'cell_label'])
                .size()
                .reset_index(name='ts_count')
        )

        # Ensure we have each cell at each time in "all_cells"
        all_cells = self.cellprops[['h5_idx', 'time', 'fov', 'cell_label']].drop_duplicates()

        # Merge with ts_count_df so cells with no transcription sites get ts_count = 0
        merged = pd.merge(
            all_cells,
            ts_count_df,
            on=['h5_idx', 'time', 'fov', 'cell_label'],
            how='left'
        )
        merged['ts_count'] = merged['ts_count'].fillna(0)

        # Categorize ts_count
        def cat_func(x):
            if x == 1:
                return '1'
            elif x == 2:
                return '2'
            elif x == 3:
                return '3'
            elif x >= 4:
                return '>=4'
            else:
                return '0'  # explicitly mark 0

        merged['ts_cat'] = merged['ts_count'].apply(cat_func)

        # Count how many cells per time & ts_cat
        grouped = merged.groupby(['time', 'ts_cat']).size().reset_index(name='count')

        # Calculate total cells per time point
        total_by_time = merged.groupby('time').size().reset_index(name='total')
        grouped = pd.merge(grouped, total_by_time, on='time')
        grouped['fraction'] = grouped['count'] / grouped['total']

        # Exclude TS == 0 from the final plot (but they remain in the denominator)
        grouped = grouped[grouped['ts_cat'] != '0']

        # Pivot so each row = time, columns = ts_cat, values = fraction
        pivoted = grouped.pivot(index='time', columns='ts_cat', values='fraction').fillna(0)

        # Reorder columns if they exist
        existing_cols = [c for c in ['1', '2', '3', '>=4'] if c in pivoted.columns]
        pivoted = pivoted[existing_cols]

        # Plot a stacked bar chart
        ax = pivoted.plot(kind='bar', stacked=False, figsize=(8, 5))
        ax.set_xlabel("time")
        ax.set_ylabel("Fraction of cells")
        ax.set_title("Fraction of cells with 1, 2, 3, >=4 TS by time")
        plt.legend(title="TS count")
        plt.tight_layout()
        plt.show()

    ###########################################################################
    # 12) Overview plots: Intensity, snr (Hitograms and scatter plots) & TS distribution bar-plot
    ###########################################################################
    def _display_overview_plots(self):
        """
        1) SNR histogram (ALL spots in entire dataset)
        2) Intensity histogram (ALL spots)
        3) Scatter (signal vs. snr) for ALL
        4) Bar plot of fraction of cells with 1,2,3,>=4 TS per time
        """
        if self.spots.empty:
            print("No spots to display in overview plots.")
            return

        fig, axs = plt.subplots(1, 3, figsize=(18,5))

        # 1) SNR histogram
        axs[0].hist(self.spots['snr'], bins=60, alpha=0.7, edgecolor='black')
        axs[0].set_title("SNR Histogram (All Spots)")
        axs[0].set_xlabel("SNR")
        axs[0].set_ylabel("Count")

        # 2) Intensity histogram
        axs[1].hist(self.spots['signal'], bins=60, alpha=0.7, edgecolor='black')
        axs[1].set_title("Intensity Histogram (All Spots)")
        axs[1].set_xlabel("Signal")
        axs[1].set_ylabel("Count")
        axs[1].set_yscale("log")

        # 3) Scatter: signal vs. snr
        axs[2].scatter(self.spots['signal'], self.spots['snr'], s=10, alpha=0.6)
        axs[2].set_title("Signal vs. SNR (All Spots)")
        axs[2].set_xlabel("Signal")
        axs[2].set_ylabel("SNR")
        axs[2].set_xscale('log')
        axs[2].set_yscale('log')

        plt.tight_layout()
        plt.show()

        # 4) Now show the bar plot for TS distribution by time
        self._display_ts_barplot_by_time()


    ###########################################################################
    # MAIN display orchestrator
    ###########################################################################
    def display(
        self, 
        newFOV=True, 
        newCell=True, 
        spotChannel=0, 
        cytoChannel=1, 
        nucChannel=2,
        thresholds=[0, 2, 3, 4]
    ):
        """
        Steps:
          1) Weighted SNR threshold assigned (keep_wsnr)
          2) Possibly pick random FOV/cell that has a TS if possible
          3) Full-FOV segmentation
          4) Zoom on cell (pick ONE chosen_spot not in TS) -> store in self.chosen_spot
          5) Zoom on that same chosen_spot + chosen_ts
          6) 2x2 figure for SNR thresholds -> same cell (TS in magenta)
          7) Print spots before/after threshold
          8) Histograms & scatter for SNR thresholds -> same cell
          9) Further zoom for each threshold T, re-using the same chosen_spot
         10) Weighted threshold figure (1x3)
         11) Weighted threshold further zoom
         12) Overview for entire dataset (including new TS bar-plot)
        """

        # 1) Weighted threshold logic
        thr_val = self.assign_revised_weighted_threshold()
        print(f"[INFO] Weighted threshold in [2,5): bottom 20% => {thr_val:.2f}")

        # 2) Possibly pick new FOV/cell (once)
        if newFOV or self.fov is None:
            self._pick_random_FOV_and_cell_with_TS()

        elif newCell or self.cell_label is None:
            # Force picking a new cell from the same FOV
            valid_cells = self.cellprops[
                (self.cellprops['h5_idx'] == self.h5_idx) &
                (self.cellprops['fov'] == self.fov) &
                (self.cellprops['cell_label'] != 0)
            ]['cell_label'].unique()
            if len(valid_cells) == 0:
                print(f"No valid cell_label in FOV={self.fov}. Aborting.")
                return
            self.cell_label = np.random.choice(valid_cells)
            print(f"Chose new random cell_label={self.cell_label} in same FOV={self.fov}.")

        # 3) Full-FOV segmentation
        self._display_full_fov_segmentation(cytoChannel, nucChannel)

        # 4) Zoom on cell
        self._display_zoom_on_cell(spotChannel, cytoChannel, nucChannel)
        if self.chosen_spot is None:
            return  # no spot => abort

        # 5) Zoom on that same chosen_spot
        self._display_zoom_on_one_spot(spotChannel)

        # 6) 2x2 figure for SNR thresholds
        self._display_snr_thresholds(spotChannel, cytoChannel, nucChannel, thresholds=thresholds)

        # 7) Print spots before/after threshold
        self._print_spots_before_after_threshold(thresholds)

        # 8) Histograms & scatter for SNR thresholds (cell-based)
        self._display_threshold_histograms_scatter(thresholds)

        # 9) Further zoom for each threshold T
        self._display_further_zoom_for_thresholds(spotChannel, thresholds)

        # 10) Weighted threshold figure (1x3)
        self._display_weighted_threshold_figure(spotChannel, cytoChannel, nucChannel)

        # 11) Weighted threshold further zoom
        self._display_weighted_zoom_on_spot(spotChannel)

        # 12) Overview for entire dataset (including bar plot of TS distribution)
        self._display_overview_plots()
   
    ###########################################################################
    # display_gating_plus_display
    ###########################################################################
    def display_gating_plus_display(
        self,
        spotChannel=0,
        cytoChannel=1,
        nucChannel=2
    ):
        """
        1) Calls display_gating() to pick a random FOV/h5_idx + show gating overlay.
        2) We then pick one "kept" cell from cellprops in that FOV for further sub-plots.
        3) Runs these sub-displays in order:
            (1) _display_full_fov_segmentation
            (2) _display_zoom_on_cell
            (3) _display_weighted_threshold_figure
            (4) _display_weighted_zoom_on_spot
            (5) _display_overview_plots
        NOTE: This assumes you have *already* pruned your spots DataFrame
              to keep only those that pass the weighted SNR threshold
              (e.g. spots = spots[spots['keep_wsnr']]).
        """

        # 1) Call display_gating => sets self.h5_idx, self.fov, prints gating overlay
        self.display_gating()

        # 2) Among the cellprops for (h5_idx, fov), pick a single cell_label
        #    that presumably is "kept" (i.e. it appears in the gating overlay as non-red).
        cellprops_frame = self.cellprops[
            (self.cellprops['h5_idx'] == self.h5_idx) &
            (self.cellprops['fov'] == self.fov)
        ]
        if cellprops_frame.empty:
            print("No 'kept' cells found in this FOV => cannot do further display.")
            return

        # pick one
        self.cell_label = np.random.choice(cellprops_frame['cell_label'].unique())
        print(f"Chosen cell_label = {self.cell_label} for further display steps.")

        # 3) Now run your five sub-routines in sequence
        # self._display_full_fov_segmentation(cytoChannel, nucChannel)
        self._display_zoom_on_cell(spotChannel, cytoChannel, nucChannel)
        self._display_weighted_zoom_on_spot(spotChannel)
        self._display_overview_plots()

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

#%%
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

    ############################################################
    # Data loading and saving
    ############################################################
    def get_data(self):
        """ Load data from the AnalysisManager. """
        h = self.am.h5_files
        self.cellprops = self.am.select_datasets('cell_properties', 'dataframe')

        self.images, self.masks = self.am.get_images_and_masks()

        # Check if illumination profiles exist before trying to load them
        try:
            self.illumination_profiles = self.am.select_datasets('illumination_profiles', 'array')[0]
        except KeyError:
            print("Warning: 'illumination_profiles' dataset not found. Proceeding without illumination correction.")
            self.illumination_profiles = None

        # Check if corrected illumination profile exists before trying to load it
        try:
            self.corrected_IL_profile = self.am.select_datasets('corrected_IL_profile', 'array')[0]
        except KeyError:
            print("Warning: 'corrected_IL_profile' dataset not found. Proceeding without corrected illumination.")
            self.corrected_IL_profile = None

    def save_data(self, location):
        """
        Saves the DataFrames to CSV.
        """
        self.cellprops.to_csv(os.path.join(location, 'cellprops.csv'), index=False)

    def display_gating(self):
        Cyto_Channel = 1
        required_columns = ['unique_cell_id']

        # Check if required columns are present
        if not all(col in self.cellprops.columns for col in required_columns):
            print("Required columns missing in cellprops DataFrame.")
            return

        # select a h5_index at random
        id = np.random.choice(len(self.cellprops))
        spot_row = self.cellprops.iloc[id]
        h5_idx, fov, nas_location = spot_row['h5_idx'], spot_row['fov'], spot_row['NAS_location']
        self.h5_idx = h5_idx
        self.fov = fov
        print(f'Selected H5 Index: {h5_idx}')
        print(f'Nas Location: {nas_location}')
        print(f'FOV: {fov}' )

        img, mask = self.images[h5_idx][fov, 0, :, :, :, :], self.masks[h5_idx][fov, 0, :, :, :, :]
        mask = np.max(mask, axis=1) # This should make czyx to cyz
        img = np.max(img, axis=1)
        mask.compute()
        img.compute()


        cellprops_frame = self.cellprops[
            (self.cellprops['h5_idx'] == h5_idx) &
            (self.cellprops['fov'] == fov) &
            (self.cellprops['NAS_location'] == nas_location)
        ]
        print(f"Cell Properties DataFrame: {cellprops_frame.shape}")


        # Identify which cell_labels appear in cellprops => "kept"
        cell_labels_in_df = set(cellprops_frame['cell_label'].unique())
        # Identify which appear in the mask
        cell_labels_in_mask = set(np.unique(mask.compute()))
        if 0 in cell_labels_in_mask:
            cell_labels_in_mask.remove(0)

        # Colors
        kept_colors = list(mcolors.TABLEAU_COLORS.values())  # distinct non-red
        removed_color = 'red'

        # Show gating overlay
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img[Cyto_Channel, :, :], cmap='gray')

        # A) Plot the "kept" cell outlines
        color_map = {
            cell_label: kept_colors[i % len(kept_colors)]
            for i, cell_label in enumerate(cell_labels_in_df)
        }
        for cell_label in cell_labels_in_df:
            cell_mask = (mask == cell_label)
            contours = find_contours(cell_mask[Cyto_Channel, :, :].compute(), 0.5)
            if contours:
                largest_contour = max(contours, key=lambda x: x.shape[0])
                ax.plot(largest_contour[:, 1], largest_contour[:, 0],
                        linewidth=2, color=color_map[cell_label],
                        label=f'Cell {cell_label}')

        # B) Plot "discarded" cell outlines
        cell_labels_not_in_df = cell_labels_in_mask - cell_labels_in_df
        for cell_label in cell_labels_not_in_df:
            cell_mask = (mask == cell_label)
            contours = find_contours(cell_mask[Cyto_Channel, :, :].compute(), 0.5)
            if contours:
                largest_contour = max(contours, key=lambda x: x.shape[0])
                ax.plot(largest_contour[:, 1], largest_contour[:, 0],
                        color=removed_color, linewidth=2, linestyle='dashed',
                        label=f'Removed: Cell {cell_label}')

        plt.legend()
        plt.show()

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
        gr_illum = self.illumination_profiles[GR_Channel].compute()  # (y, x)
        gr_corrected = self.corrected_IL_profile[GR_Channel].compute()  # (y, x)

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