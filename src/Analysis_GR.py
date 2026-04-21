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
from matplotlib import cm
import matplotlib.colors as mcolors
from datetime import datetime, time
import sys
import traceback


# Utility function to safely open HDF5 files with retries
def safe_open_h5(file_path, retries=3, delay=1.0):
    for attempt in range(1, retries + 1):
        try:
            return h5py.File(file_path, 'r')
        except OSError as e:
            print(f"[Retry {attempt}] Failed to open {file_path}: {e}")
            time.sleep(delay)
    raise RuntimeError(f"Failed to open file after {retries} retries: {file_path}")

class AnalysisManager:
    """
    Analysis manager for GR experiments.
    
    This class handles:
      - Extracting HDF5 file paths from a log directory.
      - Safely opening and closing HDF5 files.
      - Loading key datasets (e.g., 'cell_properties', 'illumination_profiles', 'corrected_IL_profile')
        from each HDF5 file.
      - Converting the data into pandas DataFrames and saving them to CSV.
    
    If no direct file locations are provided, the log directory is scanned to obtain them.
    """
    def __init__(self, location: Union[str, list[str]] = None, log_location: str = None, mac: bool = False, mode='r'):
        self.mode = mode
        if location is not None:
            if isinstance(location, str):
                self.location = [location]
            else:
                self.location = location
        elif log_location is not None:
            self.location = self.get_locations_from_log(log_location, mac)
        else:
            raise ValueError("Must provide either direct locations or a log_location.")
    
    def get_locations_from_log(self, log_location, mac: bool = False) -> list[str]:
        """
        Reads log files from the specified directory to extract HDF5 file paths.
        
        Parameters:
            log_location (str): Directory path where log files are stored.
            mac (bool): Whether running on macOS (affects drive/path construction).
            
        Returns:
            list[str]: List of full HDF5 file paths.
        """
        log_files = os.listdir(log_location)
        locations = []
        for log_filename in log_files:
            log_file_path = os.path.join(log_location, log_filename)
            try:
                with open(log_file_path, 'r') as file:
                    content = file.read()
                lines = content.split('\n')
                if len(lines) < 2:
                    continue
                # Assume the relevant line is the second-to-last line
                relevant_line = lines[-2]
                parts = relevant_line.split(' -> ')
                if len(parts) != 2:
                    continue
                first, second = parts
                name = first.split('/')[-1]
                if mac:
                    drive = '/Volumes/share/'
                else:
                    drive = os.path.splitdrive(log_location)[0] + os.sep
                second = os.path.join(*second.split('/')).strip()
                file_path = os.path.join(drive, second, name)
                locations.append(file_path)
            except Exception as e:
                print(f"Error processing log file '{log_filename}': {e}")
                traceback.print_exc()
        return locations

    def select_analysis(self, analysis_name: str = None, date_range: list[str] = None):
        """
        Opens HDF5 files, finds analysis names, filters by analysis name, filters locations,
        handles duplicates, and then closes the files.
        
        Date filtering is omitted.
        """
        self.open()
        self._find_analysis_names()
        self._filter_on_name(analysis_name)
        self._filter_locations()
        self._handle_duplicates()
        self.close()

    def list_analysis_names(self):
        """
        Opens the HDF5 files, finds and prints analysis names, then closes the files.
        """
        self.open()
        self._find_analysis_names()
        for name in self.analysis_names:
            print(name)
        self.close()
        return self.analysis_names

    def select_datasets(self, dataset_name, dtype=None):
        """
        Safely loads datasets from each HDF5 file and merges them.

        Parameters:
            dataset_name (str): The key of the dataset to load.
            dtype (str): 'dataframe' to load as a pandas DataFrame, 'array' to load as an array.

        Returns:
            pd.DataFrame if dtype=='dataframe'; otherwise, list of arrays.
        """
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
                sys.exit(1)

            print(f"Opening file: {file_path}")
            f = None
            try:
                f = safe_open_h5(file_path)
                dataset_key = f"{self.analysis_names[i]}/{dataset_name}"
                if dataset_key not in f:
                    print(f"WARNING: Dataset '{dataset_name}' not found in {file_path}. Skipping.")
                    continue

                if dtype == 'dataframe':
                    print(f"Reading DataFrame from: {file_path} -> {dataset_key}")
                    df = pd.read_hdf(f.filename, dataset_key).copy()
                    df['h5_idx'] = i
                    list_df.append(df)
                    is_df = True

                elif dtype == 'array':
                    print(f"Reading Array from: {file_path} -> {dataset_key}")
                    array = f[dataset_key][:]
                    list_arrays.append(array)
                    is_array = True

                else:
                    raise ValueError(f"Unknown dtype '{dtype}'. Accepted types: 'dataframe', 'array'.")

            except OSError as e:
                print(f"\nCRITICAL ERROR: Unable to read {file_path}")
                print(f"HDF5 Error: {e}")
                traceback.print_exc()
                sys.exit(1)

            finally:
                if f is not None:
                    try:
                        f.close()
                    except Exception as close_err:
                        print(f"WARNING: Error closing file {file_path}: {close_err}")

        if is_df:
            print(f"Successfully loaded {len(list_df)} DataFrames. Merging...")
            return pd.concat(list_df, axis=0, ignore_index=True)

        if is_array:
            print(f"Successfully loaded {len(list_arrays)} arrays.")
            return list_arrays

        print(f"ERROR: No valid data found for dataset: {dataset_name}.")
        sys.exit(1)

    def save_to_csv(self, df, csv_path):
        """
        Saves a DataFrame to a CSV file.
        """
        try:
            df.to_csv(csv_path, index=False)
            print(f"Saved CSV to {csv_path}")
        except Exception as e:
            print(f"Error saving CSV to {csv_path}: {e}")

    def load_and_save(self, dataset_name, csv_path, dtype='dataframe'):
        """
        Loads a dataset from all HDF5 files and saves it to CSV.
        """
        data = self.select_datasets(dataset_name, dtype)
        if dtype == 'dataframe' and data is not None:
            self.save_to_csv(data, csv_path)
        return data

    def _find_analysis_names(self):
        self.analysis_names = []
        for h in self.h5_files:
            self.analysis_names.append(list(h.keys()))
        self.analysis_names = set([dataset for sublist in self.analysis_names for dataset in sublist])
        self.analysis_names = [d for d in self.analysis_names if 'Analysis' in d]

    def _filter_on_name(self, analysis_name):
        self.names = ['_'.join(s.split('_')[1:-1]) for s in self.analysis_names]
        if analysis_name is not None:
            self.analysis_names = [self.analysis_names[i] for i, s in enumerate(self.names) if analysis_name in s]

    def _filter_on_date(self, date_range: tuple):
        # Date filtering is omitted.
        pass

    def _filter_locations(self):
        temp_locs = []
        temp_names = []
        for h_idx, h in enumerate(self.h5_files):
            for n in self.analysis_names:
                if n in h.keys():
                    temp_locs.append(h.filename)
                    temp_names.append(n)
        self.location = temp_locs
        self.analysis_names = temp_names

    def get_images_and_masks(self):
        self.raw_images = []
        self.masks = []

        for l in self.location:
            # with h5py.File(l, 'r') as h:
            self.h5_files.append(h5py.File(l))
            self.raw_images.append(da.from_array(self.h5_files[-1]['raw_images']))
            self.masks.append(da.from_array(self.h5_files[-1]['masks']))
        return self.raw_images, self.masks
    
    def get_corrected_images_and_masks(self):
        self.corrected_images = []
        self.masks = []

        for l, an in zip(self.location, self.analysis_names):
            # with h5py.File(l, 'r') as h:
            self.h5_files.append(h5py.File(l))
            self.corrected_images.append(da.from_array(self.h5_files[-1][an]['images']))
            self.masks.append(da.from_array(self.h5_files[-1]['masks']))
        return self.corrected_images, self.masks

    def _handle_duplicates(self):
        # Stub for duplicate handling.
        pass

    def open(self):
        self.h5_files = []
        for l in self.location:
            if l not in [h.filename for h in self.h5_files]:
                self.h5_files.append(h5py.File(l, self.mode))
    
    def close(self):
        for h in getattr(self, 'h5_files', []):
            try:
                # only flush if opened with write mode
                if self.mode and self.mode != 'r':
                    h.flush()
            except Exception:
                pass
            try:
                h.close()
            except Exception:
                pass
    
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

    def delete_selected_analysis(self, password):
        if password != 'you_better_know_what_your_deleting':
            print("you_better_know_what_your_deleting")
            return

        for filepath, analysis_name in zip(self.location, self.analysis_names):
            try:
                with h5py.File(filepath, 'a') as f:
                    if analysis_name in f:
                        del f[analysis_name]
                        print(f"Deleted analysis '{analysis_name}' from {filepath}")
                    else:
                        print(f"Analysis '{analysis_name}' not found in {filepath}")
            except Exception as e:
                print(f"Failed to delete '{analysis_name}' from {filepath}: {e}")


class GR_Confirmation:
    """
    GR_Confirmation handles display and validation for Glucocorticoid Receptor (GR) ICC image analysis.

    This class allows visualization of raw images, corrected images, illumination profiles,
    and pseudo-cytoplasmic segmentation masks for randomly selected or user-specified
    fields of view (FOVs) from HDF5 experiment datasets.

    Attributes:
    -----------
    am : AnalysisManager
        The associated analysis manager object.
    seed : int, optional
        Random seed for reproducibility.
    cellprops : pd.DataFrame
        Cell-level measurements loaded from the analysis.
    images : list
        Corrected image stacks from HDF5 files.
    illumination_profiles : array or dask.array
        Illumination correction profiles.
    corrected_IL_profile : array or dask.array
        Composite illumination profile computed from corrected images.

    Methods:
    --------
    get_data()
        Loads all required data from HDF5 files.
    display(h5_idx=None, num_fovs=1)
        Displays raw and corrected images, illumination profiles, segmentation overlays,
        and intensity histograms for one or more FOVs.
    """

    def __init__(self, am, seed=None):
        self.am = am
        self.seed = seed
        self.cellprops = None
        self.raw_images = None
        self.illumination_profiles = None
        self.corrected_IL_profile = None
        self.corrected_images = None

    def get_data(self):
        self.cellprops = self.am.select_datasets('cell_properties', 'dataframe')
        self.raw_images, _ = self.am.get_images_and_masks()
        self.corrected_images, _ = self.am.get_corrected_images_and_masks()

        try:
            self.illumination_profiles = self.am.select_datasets('illumination_profiles', 'array')[0]
        except KeyError:
            print("Warning: 'illumination_profiles' dataset not found.")

        try:
            self.corrected_IL_profile = self.am.select_datasets('corrected_IL_profile', 'array')[0]
        except KeyError:
            print("Warning: 'corrected_IL_profile' dataset not found.")

    def _select_fovs(self, h5_idx=None, num_fovs=1):
        available = self.cellprops[['h5_idx', 'fov']].drop_duplicates()
        chosen_pairs = []

        if h5_idx is None:
            sample = available.sample(1)
            return [(int(sample['h5_idx']), int(sample['fov']))]

        if isinstance(h5_idx, int):
            fovs = available[available['h5_idx'] == h5_idx]['fov'].unique()
            selected = np.random.choice(fovs, min(num_fovs, len(fovs)), replace=False)
            return [(h5_idx, int(f)) for f in selected]

        if isinstance(h5_idx, list):
            for h5 in h5_idx:
                fovs = available[available['h5_idx'] == h5]['fov'].unique()
                selected = np.random.choice(fovs, min(num_fovs, len(fovs)), replace=False)
                chosen_pairs.extend([(h5, int(f)) for f in selected])
            return chosen_pairs

        raise ValueError("Invalid h5_idx type.")

    def _load_and_project_raw_and_mask(self, file_path, fov):
        GR_Channel = 0
        Nuc_Channel = 1
        with h5py.File(file_path, 'r') as f:
            raw = da.from_array(f['raw_images'])
            mask = da.from_array(f['masks'])

            raw_2d = raw[fov, 0, GR_Channel].max(axis=0).compute()
            gr_mask = mask[fov, 0, GR_Channel].max(axis=0).compute()
            nuc_mask = mask[fov, 0, Nuc_Channel].max(axis=0).compute()

            pseudo_cyto = gr_mask - nuc_mask
            pseudo_cyto[pseudo_cyto < 0] = 0

        return raw_2d, pseudo_cyto, nuc_mask

    def _plot_display_panels(self, h5_idx, fov, raw_2d, pseudo_cyto, nuc_mask, corrected_image, illum_profile, corrected_profile):
        fig, axs = plt.subplots(2, 5, figsize=(25, 10))

        # Top row: images
        axs[0, 0].imshow(raw_2d, cmap="gray")
        axs[0, 0].set_title("Raw GR Image")
        axs[0, 0].axis("off")

        axs[0, 1].imshow(illum_profile, cmap="gray")
        cs1 = axs[0, 1].contour(illum_profile, levels=10, colors="white", linewidths=0.5)
        axs[0, 1].clabel(cs1, inline=True, fontsize=8)
        axs[0, 1].set_title("Illumination Profile")
        axs[0, 1].axis("off")

        axs[0, 2].imshow(corrected_image, cmap="gray")
        axs[0, 2].set_title("Corrected Image (no mask)")
        axs[0, 2].axis("off")

        axs[0, 3].imshow(corrected_profile, cmap="gray")
        cs2 = axs[0, 3].contour(corrected_profile, levels=10, colors="white", linewidths=0.5)
        axs[0, 3].clabel(cs2, inline=True, fontsize=8)
        axs[0, 3].set_title("Corrected IL Profile")
        axs[0, 3].axis("off")

        axs[0, 4].imshow(corrected_image, cmap="gray")
        axs[0, 4].imshow(pseudo_cyto, cmap="jet", alpha=0.4)
        axs[0, 4].set_title("Corrected Image + Masks")
        axs[0, 4].axis("off")

        # Bottom row: histograms
        bins = 100
        axs[1, 0].hist(raw_2d.ravel(), bins=bins, color='orange', alpha=0.6, label="Raw")
        axs[1, 0].hist(corrected_image.ravel(), bins=bins, color='green', alpha=0.6, label="Corrected")
        axs[1, 0].set_title("Raw vs Corrected Histogram")
        axs[1, 0].set_xlabel("Intensity")
        axs[1, 0].set_ylabel("Pixel Count")
        axs[1, 0].legend()

        cyto_vals_corr = corrected_image[pseudo_cyto > 0]
        nuc_vals_corr = corrected_image[nuc_mask > 0]

        axs[1, 1].hist(nuc_vals_corr, bins=bins, color='purple', alpha=0.6, label='Nucleus (Corr)')
        axs[1, 1].hist(cyto_vals_corr, bins=bins, color='orange', alpha=0.6, label='Pseudo-cyto (Corr)')
        axs[1, 1].set_title("Corrected Intensity by Region")
        axs[1, 1].set_xlabel("Intensity")
        axs[1, 1].set_ylabel("Pixel Count")
        axs[1, 1].legend()

        cyto_vals_raw = raw_2d[pseudo_cyto > 0]
        nuc_vals_raw = raw_2d[nuc_mask > 0]

        axs[1, 2].hist(nuc_vals_raw, bins=bins, color='purple', alpha=0.6, label='Nucleus (Raw)')
        axs[1, 2].hist(cyto_vals_raw, bins=bins, color='orange', alpha=0.6, label='Pseudo-cyto (Raw)')
        axs[1, 2].set_title("Raw Intensity by Region")
        axs[1, 2].set_xlabel("Intensity")
        axs[1, 2].set_ylabel("Pixel Count")
        axs[1, 2].legend()

        # Empty plots
        axs[1, 3].axis("off")
        axs[1, 4].axis("off")

        fig.suptitle(f"h5_idx={h5_idx}, fov={fov}", fontsize=16)
        plt.tight_layout()
        plt.show()

    def display(self, h5_idx: Union[int, list[int], None] = None, num_fovs: int = 1):
        GR_Channel = 0
        pairs = self._select_fovs(h5_idx, num_fovs)

        for h5_idx, fov in pairs:
            file_path = self.am.location[h5_idx]
            raw_2d, pseudo_cyto, nuc_mask = self._load_and_project_raw_and_mask(file_path, fov)
            corrected_stack = self.corrected_images[h5_idx][fov, 0, GR_Channel]
            corrected_2d = corrected_stack.max(axis=0).compute()

            illum_profile = self.illumination_profiles[GR_Channel]
            if hasattr(illum_profile, 'compute'):
                illum_profile = illum_profile.compute()

            corrected_profile = self.corrected_IL_profile[GR_Channel]
            if hasattr(corrected_profile, 'compute'):
                corrected_profile = corrected_profile.compute()

            # Normalize corrected profile to range [0, 1]
            if np.max(corrected_profile) > 0:
                corrected_profile = corrected_profile / np.max(corrected_profile)

            self._plot_display_panels(
                h5_idx, fov, raw_2d, pseudo_cyto, nuc_mask, corrected_2d, illum_profile, corrected_profile
            )


class GR_DisplayBasic:
    """
    GR_DisplayBasic shows raw GR images with segmentation overlays and intensity histograms.

    This class is designed for analyses that do not apply illumination correction.

    Attributes:
    -----------
    am : AnalysisManager
        The associated analysis manager.
    cellprops : pd.DataFrame
        Cell-level metadata.
    raw_images : list
        Dask arrays of raw image data.
    masks : list
        Dask arrays of segmentation masks.

    Methods:
    --------
    get_data()
        Load cell metadata, raw images, and masks from HDF5 files.
    display(h5_idx=None, num_fovs=1)
        Show raw image, segmentation overlays, and intensity histograms for selected FOV(s).
    """

    def __init__(self, am):
        self.am = am
        self.cellprops = None
        self.raw_images = None
        self.masks = None

    def get_data(self):
        self.cellprops = self.am.select_datasets('cell_properties', 'dataframe')
        self.raw_images, self.masks = self.am.get_images_and_masks()

    def _select_fovs(self, h5_idx=None, num_fovs=1):
        available = self.cellprops[['h5_idx', 'fov']].drop_duplicates()
        chosen_pairs = []

        if h5_idx is None:
            sample = available.sample(1)
            return [(int(sample['h5_idx']), int(sample['fov']))]

        if isinstance(h5_idx, int):
            fovs = available[available['h5_idx'] == h5_idx]['fov'].unique()
            selected = np.random.choice(fovs, min(num_fovs, len(fovs)), replace=False)
            return [(h5_idx, int(f)) for f in selected]

        if isinstance(h5_idx, list):
            for h5 in h5_idx:
                fovs = available[available['h5_idx'] == h5]['fov'].unique()
                selected = np.random.choice(fovs, min(num_fovs, len(fovs)), replace=False)
                chosen_pairs.extend([(h5, int(f)) for f in selected])
            return chosen_pairs

        raise ValueError("Invalid h5_idx type.")

    def display(self, h5_idx: Union[int, list[int], None] = None, num_fovs: int = 1):
        GR_Channel = 0
        Nuc_Channel = 1
        pairs = self._select_fovs(h5_idx, num_fovs)

        for h5_idx, fov in pairs:
            raw_3d = self.raw_images[h5_idx][fov, 0, GR_Channel]  # (z, y, x)
            raw_2d = raw_3d.max(axis=0).compute()

            mask_gr = self.masks[h5_idx][fov, 0, GR_Channel].max(axis=0).compute()
            mask_nuc = self.masks[h5_idx][fov, 0, Nuc_Channel].max(axis=0).compute()

            # Pseudo-cytoplasm = GR - Nucleus
            pseudo_cyto = mask_gr - mask_nuc
            pseudo_cyto[pseudo_cyto < 0] = 0

            fig, axs = plt.subplots(1, 3, figsize=(18, 5))

            axs[0].imshow(raw_2d, cmap='gray')
            axs[0].set_title("Raw GR Image")
            axs[0].axis("off")

            axs[1].imshow(raw_2d, cmap='gray')
            axs[1].imshow(pseudo_cyto, cmap='jet', alpha=0.4)
            axs[1].set_title("Raw + Pseudo-Cyto")
            axs[1].axis("off")

            # Histogram: nuclear and pseudo-cyto intensity distributions
            nuc_vals = raw_2d[mask_nuc > 0]
            cyto_vals = raw_2d[pseudo_cyto > 0]

            axs[2].hist(nuc_vals, bins=100, color='purple', alpha=0.6, label='Nucleus')
            axs[2].hist(cyto_vals, bins=100, color='orange', alpha=0.6, label='Pseudo-cyto')
            axs[2].set_title("GR Intensity: Nucleus vs Pseudo-Cyto")
            axs[2].set_xlabel("Intensity")
            axs[2].set_ylabel("Pixel Count")
            axs[2].legend()

            fig.suptitle(f"h5_idx={h5_idx}, fov={fov}", fontsize=14)
            plt.tight_layout()
            plt.show()