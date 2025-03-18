"""
Analysis_DUSP1.py

This module contains classes for DUSP1 data analysis and visualization:

1. DUSP1AnalysisManager:
   - Manages HDF5 file access.
   - Extracts file paths from a log directory (if no direct locations are provided).
   - Provides methods to select an analysis (by name) and load datasets from HDF5 files.
   - Saves datasets as CSV.

2. DUSP1Measurement:
   - Processes spot, cluster, and cell properties data to produce cell-level measurements.
   - Supports optional SNR filtering on spots using three methods: 
       "weighted", "absolute", and "mg".
   - Adds metadata column 'unique_cell_id' unique for each (fov, NAS_location, h5_idx) for later inspection.

3. SNRAnalysis:
   - Performs three SNR-based analyses:
       • Weighted thresholding (using the 'snr' column with a 20th percentile cutoff between 2 and 5).
       • MG SNR calculation: MG_SNR = (signal - cell_intensity_mean-0) / cell_intensity_std-0.
       • Absolute thresholding (if an absolute threshold is provided).
   - Also compares the original 'snr' with MG_SNR by adding comparison columns.

4. DisplayManager:
   - Provides visualization routines.
   - Safely retrieves images and masks from HDF5 files.
   - Displays gating overlays (optionally highlighting a specific cell).
   - Displays cell crops filtered by an assigned total expression group.
   - Separately displays random cell crops for cells with transcription sites (TS) and with foci.
   - Contains a method to assign expression groups based on total mRNA (num_nuc_spots + num_cyto_spots) using the same bounds as for nuclear.

Author: Eric Ron
Date: March 12 2025
"""

import os
import h5py
import pandas as pd
import numpy as np
import sys
import traceback
import random
from typing import Union
from datetime import datetime
import dask.array as da
import matplotlib.pyplot as plt

#############################
# DUSP1AnalysisManager Class
#############################

class DUSP1AnalysisManager:
    """
    Analysis manager for DUSP1 experiments.
    
    This class handles:
      - Extracting HDF5 file paths from a log directory.
      - Safely opening and closing HDF5 files.
      - Loading key datasets (e.g., 'spotresults', 'cell_properties', 'clusterresults')
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
        self.open()
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
        if is_df:
            print(f"Successfully loaded {len(list_df)} DataFrames. Merging...")
            self.close()
            return pd.concat(list_df, axis=0, ignore_index=True)
        if is_array:
            print(f"Successfully loaded {len(list_arrays)} arrays.")
            self.close()
            return list_arrays
        print(f"ERROR: No valid data found for dataset: {dataset_name}.")
        self.close()
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

    def _handle_duplicates(self):
        # Stub for duplicate handling.
        pass

    def open(self):
        self.h5_files = []
        for l in self.location:
            if l not in [h.filename for h in self.h5_files]:
                self.h5_files.append(h5py.File(l, self.mode))
    
    def close(self):
        for h in self.h5_files:
            h.flush()
            h.close()
    
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


#############################
# DUSP1Measurement Class
#############################

class DUSP1Measurement:
    """
    Processes spot, cluster, and cell property data to produce cell-level measurements for DUSP1.
    
    Computes three SNR metrics for each spot:
      1) Weighted: Uses a weighted cutoff for spots with snr between 2 and 5.
      2) Absolute: Boolean flag indicating if snr >= snr_threshold.
      3) MG: Computed as (signal - cell_intensity_mean-0) / cell_intensity_std-0,
         and kept (MG_valid) only if MG_SNR >= the spot's original snr.
      
    Each spot (and thus each cell after aggregation) gets a unique_cell_id from cellprops.
    """
    def __init__(self, spots: pd.DataFrame, clusters: pd.DataFrame, cellprops: pd.DataFrame):
        self.spots = spots.copy()
        self.clusters = clusters.copy()
        self.cellprops = cellprops.copy()
        # Assign a unique cell ID to each row in cellprops.
        self.cellprops['unique_cell_id'] = np.arange(len(self.cellprops))
    
    def merge_data(self):
        # Ensure consistent types for the merge keys.
        for df in [self.spots, self.clusters, self.cellprops]:
            df['NAS_location'] = df['NAS_location'].astype(str)
            df['cell_label'] = df['cell_label'].astype(int)
            df['fov'] = df['fov'].astype(int)
        
        # Merge the spots and clusters dataframes by the unique cell ID
            self.spots = self.spots.merge(
                self.cellprops[['NAS_location', 'cell_label', 'fov', 'unique_cell_id']], 
                on=['NAS_location', 'cell_label', 'fov'], 
                how='left'
            )
            self.clusters = self.clusters.merge(
                self.cellprops[['NAS_location', 'cell_label', 'fov', 'unique_cell_id']], 
                on=['NAS_location', 'cell_label', 'fov'], 
                how='left'
            )        
        
        return self.cellprops, self.spots, self.clusters
    
    @staticmethod
    def second_largest(series):
        """Return the second-largest unique value in the series; if not available, return 0."""
        unique_vals = series.dropna().unique()
        if len(unique_vals) < 2:
            return 0
        return np.sort(unique_vals)[-2]
    
    def measure(self, snr_threshold: float) -> pd.DataFrame:
        """
        Processes the data to produce cell-level measurements.
        
        For each spot, the following SNR-related metrics are computed:
          - weighted_pass: Boolean flag based on a weighted cutoff (for snr between 2 and 5).
          - absolute_pass: Boolean flag if snr >= snr_threshold.
          - MG_SNR: Computed as (signal - cell_intensity_mean-0) / cell_intensity_std-0.
                    Then MG_valid is kept only if MG_SNR >= the spot's original snr.
        
        At the cell level, the metrics are aggregated as follows:
          - weighted_count: Count of spots passing the weighted filter.
          - absolute_pass: Binary flag (1 if any spot passes the absolute threshold).
          - mg_snr: Average of valid MG_SNR values among spots (MG_valid).
        
        In addition, other cell-level metrics (from clusters and cellprops) and metadata are included.
        """
        # Merge data so that every spot has a unique_cell_id and necessary cell properties.
        props, spots, clusters = self.merge_data()
        
        # --- Merge props columns needed for MG calculation into spots ---
        # "signal" is only in spots, while "cell_intensity_mean-0" and "cell_intensity_std-0" are in props.
        spots = spots.merge(props[['unique_cell_id', 'cell_intensity_mean-0', 'cell_intensity_std-0']],
                            on='unique_cell_id', how='left')

        # --- Compute Weighted SNR Flag ---
        mask_2_5 = (spots['snr'] >= 2) & (spots['snr'] < 5)
        weighted_cutoff = np.percentile(spots.loc[mask_2_5, 'snr'], 20) if mask_2_5.sum() > 0 else 2.0
        
        def weighted_keep_func(x):
            if x < 2:
                return False
            elif 2 <= x < 5:
                return x >= weighted_cutoff
            else:
                return True
        
        spots['weighted_pass'] = spots['snr'].apply(weighted_keep_func)
        
        # --- Compute Absolute SNR Flag ---
        spots['absolute_pass'] = spots['snr'] >= snr_threshold
        
        # --- Compute MG SNR ---
        spots['MG_SNR'] = (spots['signal'] - props['cell_intensity_mean-0']) / \
                           props['cell_intensity_std-0'].replace(0, np.nan)
        spots['MG_valid'] = spots['MG_SNR'].where(spots['MG_SNR'] >= spots['snr'], np.nan)
        
        # --- Compute Cell-level Metrics ---
        spots = spots.sort_values(by='unique_cell_id')
        clusters = clusters.sort_values(by='unique_cell_id')
        props = props.sort_values(by='unique_cell_id')
        cell_ids = props['unique_cell_id']
        
        # Metrics from clusters:
        num_ts = clusters[clusters['is_nuc'] == 1].groupby('unique_cell_id').size().reindex(cell_ids, fill_value=0)
        num_foci = clusters[clusters['is_nuc'] == 0].groupby('unique_cell_id').size().reindex(cell_ids, fill_value=0)
        num_spots_ts = clusters[clusters['is_nuc'] == 1].groupby('unique_cell_id')['nb_spots'].sum().reindex(cell_ids, fill_value=0)
        largest_ts = clusters[clusters['is_nuc'] == 1].groupby('unique_cell_id')['nb_spots'].max().reindex(cell_ids, fill_value=0)
        second_largest_ts = clusters[clusters['is_nuc'] == 1].groupby('unique_cell_id')['nb_spots']\
                              .apply(DUSP1Measurement.second_largest).reindex(cell_ids, fill_value=0)
        num_spots_foci = clusters[clusters['is_nuc'] == 0].groupby('unique_cell_id')['nb_spots'].sum().reindex(cell_ids, fill_value=0)
        
        # Metrics from spots:
        num_spots = spots.groupby('unique_cell_id').size().reindex(cell_ids, fill_value=0)
        num_nuc_spots = spots[spots['is_nuc'] == 1].groupby('unique_cell_id').size().reindex(cell_ids, fill_value=0)
        num_cyto_spots = spots[spots['is_nuc'] == 0].groupby('unique_cell_id').size().reindex(cell_ids, fill_value=0)
        
        # Aggregate SNR Metrics:
        weighted_count = spots.groupby('unique_cell_id')['weighted_pass'].sum().reindex(cell_ids, fill_value=0)
        absolute_flag = spots.groupby('unique_cell_id')['absolute_pass'].any().astype(int).reindex(cell_ids, fill_value=0)
        mg_snr = spots.groupby('unique_cell_id')['MG_valid'].mean().reindex(cell_ids, fill_value=np.nan)
        
        # Other cell properties from props:
        nuc_area = props['nuc_area']
        cyto_area = props['cyto_area']
        avg_nuc_int = props['nuc_intensity_mean-0']
        avg_cyto_int = props['cyto_intensity_mean-0']
        time = props['time']
        dex_conc = props['Dex_Conc']
        replica = spots.groupby('unique_cell_id')['replica'].first().reindex(cell_ids, fill_value=np.nan)
        fov = props['fov']
        nas_location = props['NAS_location']
        h5_idx = props['h5_idx']
        
        results = pd.DataFrame({
            'cell_id': cell_ids,
            'weighted_count': weighted_count.values,
            'absolute_pass': absolute_flag.values,
            'mg_snr': mg_snr.values,
            'num_ts': num_ts.values,
            'num_spots_ts': num_spots_ts.values,
            'largest_ts': largest_ts.values,
            'second_largest_ts': second_largest_ts.values,
            'num_foci': num_foci.values,
            'num_spots_foci': num_spots_foci.values,
            'num_spots': num_spots.values,
            'num_nuc_spots': num_nuc_spots.values,
            'num_cyto_spots': num_cyto_spots.values,
            'nuc_area_px': nuc_area.values,
            'cyto_area_px': cyto_area.values,
            'avg_nuc_int': avg_nuc_int.values,
            'avg_cyto_int': avg_cyto_int.values,
            'time': time.values,
            'dex_conc': dex_conc.values,
            'replica': replica.values,
            'fov': fov.values,
            'nas_location': nas_location.values,
            'h5_idx': h5_idx.values
        })
        
        return results

#############################
# SNRAnalysis Class
#############################

class SNRAnalysis:
    """
    This class performs three SNR-based analyses:
      1. Weighted Thresholding: Uses the 'snr' column from spots.
         - If snr < 2 → False.
         - If 2 ≤ snr < 5 → True if snr is at or above the 20th percentile in this range.
         - If snr ≥ 5 → True.
      2. MG SNR Calculation: Computed as:
             MG_SNR = (signal - cell_intensity_mean-0) / cell_intensity_std-0
         via merging the spots and cell properties DataFrames.
      3. Absolute Thresholding: If an absolute threshold value is provided,
         a boolean column is added that is True when the 'snr' is greater than or equal to that value.
         
    In addition, the class compares the original 'snr' value to the computed MG SNR.
    The weighted and MG SNR analyses are always done; the absolute thresholding is optional.
    """
    def __init__(self, spots, cellprops, clusters, abs_threshold=None, mg_snr_threshold=None):
        self.spots = spots.copy()
        self.clusters = clusters.copy()
        self.cellprops = cellprops.copy()
        self.abs_threshold = abs_threshold
        self.mg_snr_threshold = mg_snr_threshold
        
        # 0. Merge the data
        self.merged_df = self.merge_data()
        # 1. Compute MG SNR.
        self.merged_df = self._calculate_mg_snr()
        # 2. Compute weighted threshold based on the original 'snr' values.
        self.weighted_cutoff = self._apply_weighted_threshold()
        # 3. Apply absolute threshold if provided.
        if self.abs_threshold is not None:
            self._apply_absolute_threshold(self.abs_threshold)
        else:
            self.merged_df['absolute'] = False
        # 4. Compare the original 'snr' to the computed MG SNR.
        self._compare_snr_to_mg()

    def merge_data(self):
        self.cellprops['unique_cell_id'] = np.arange(len(self.cellprops))
        # Ensure consistent types for the merge keys.
        for df in [self.spots, self.clusters, self.cellprops]:
            df['NAS_location'] = df['NAS_location'].astype(str)
            df['cell_label'] = df['cell_label'].astype(int)
            df['fov'] = df['fov'].astype(int)
        
        # Merge the spots and clusters dataframes by the unique cell ID
            self.spots = self.spots.merge(
                self.cellprops[['NAS_location', 'cell_label', 'fov', 'unique_cell_id', 'cell_intensity_mean-0', 'cell_intensity_std-0']], 
                on=['NAS_location', 'cell_label', 'fov'], 
                how='left'
            )
            self.clusters = self.clusters.merge(
                self.cellprops[['NAS_location', 'cell_label', 'fov', 'unique_cell_id', 'cell_intensity_mean-0', 'cell_intensity_std-0']], 
                on=['NAS_location', 'cell_label', 'fov'], 
                how='left'
            )        
        return self.cellprops, self.spots, self.clusters

    def _calculate_mg_snr(self):
        merged_df = self.merged_df
        if 'signal' not in merged_df.keys():
            raise ValueError("Column 'signal' not found in spots DataFrame.")
        if ('cell_intensity_mean-0' not in merged_df.keys or 
            'cell_intensity_std-0' not in merged_df.keys):
            raise ValueError("Required intensity columns not found in cell properties DataFrame.")
        merged_df['MG_SNR'] = (merged_df['signal'] - merged_df['cell_intensity_mean-0']) / \
                              merged_df['cell_intensity_std-0'].replace(0, np.nan)
        return merged_df

    def _apply_weighted_threshold(self):
        if 'snr' not in self.merged_df.keys():
            raise ValueError("Merged DataFrame must contain a 'snr' column for weighted thresholding.")
        mask_2_5 = (self.merged_df['snr'] >= 2) & (self.merged_df['snr'] < 5)
        if mask_2_5.sum() > 0:
            weighted_cutoff = np.percentile(self.merged_df.loc[mask_2_5, 'snr'], 20)
        else:
            weighted_cutoff = 2.0
        def weighted_decision(snr):
            if snr < 2:
                return False
            elif 2 <= snr < 5:
                return snr >= weighted_cutoff
            else:
                return True
        self.merged_df['weighted'] = self.merged_df['snr'].apply(weighted_decision)
        return weighted_cutoff

    def _apply_absolute_threshold(self, abs_threshold):
        self.merged_df['absolute'] = self.merged_df['snr'] >= abs_threshold

    def _compare_snr_to_mg(self):
        if 'snr' not in self.merged_df.columns or 'MG_SNR' not in self.merged_df.columns:
            raise ValueError("Both 'snr' and 'MG_SNR' must exist in the merged DataFrame.")
        self.merged_df['snr_vs_mg'] = self.merged_df['snr'] - self.merged_df['MG_SNR']
        self.merged_df['snr_gt_mg'] = self.merged_df['snr'] > self.merged_df['MG_SNR']

    def plot_mg_snr_distribution(self, threshold=None, bins=50):
        if 'MG_SNR' not in self.merged_df.columns:
            raise ValueError("MG_SNR column not found in merged DataFrame.")
        plt.figure(figsize=(8, 6))
        plt.hist(self.merged_df['MG_SNR'].dropna(), bins=bins, alpha=0.7, edgecolor='black')
        plt.xlabel("MG SNR")
        plt.ylabel("Count")
        plt.title("Distribution of MG SNR")
        if threshold is not None:
            plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f"Threshold = {threshold}")
            plt.legend()
        plt.show()

    def get_merged_df(self):
        return self.merged_df

#############################
# DisplayManager Class
#############################

class DisplayManager:
    """
    DisplayManager encapsulates visualization routines for DUSP1 analysis.
    
    It is designed to:
      - Safely open HDF5 files to retrieve image and mask data.
      - Visualize gating overlays and display cell crops.
      - Assign expression groups based on total mRNA (num_nuc_spots + num_cyto_spots) using the same bounds as for nuclear.
      - Display random cell crops for cells with transcription sites (TS) or with foci (separately).
    
    Parameters:
        h5_file_paths (list of str): List of HDF5 file paths.
        measurement_data (dict): Dictionary with cell metadata under the key 'cells'. Each cell record should include:
             'unique_cell_id', 'h5_index', 'fov', 'bbox', 'num_nuc_spots', 'num_cyto_spots'.
        clusters_df (pd.DataFrame): DataFrame containing cluster data (with 'unique_cell_id' and 'is_nuc').
    """
    def __init__(self, h5_file_paths, measurement_data, clusters_df):
        self.h5_file_paths = h5_file_paths
        self.measurement_data = measurement_data  # expects a dict with key 'cells'
        self.clusters_df = clusters_df

    def get_images_and_masks(self, h5_index):
        file_path = self.h5_file_paths[h5_index]
        try:
            with h5py.File(file_path, 'r') as h5_file:
                images = np.array(h5_file['raw_images'])
                masks = np.array(h5_file['masks'])
            return images, masks
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            traceback.print_exc()
            return None, None

    def display_gating_overlay(self, h5_index=None, cell_id=None):
        if h5_index is None:
            h5_index = random.choice(range(len(self.h5_file_paths)))
        images, masks = self.get_images_and_masks(h5_index)
        if images is None or masks is None:
            return
        fov = 0
        img = images[fov, 0, :, :]
        mask = masks[fov, 0, :, :]
        plt.figure(figsize=(10, 10))
        plt.imshow(img, cmap='gray')
        plt.imshow(mask, cmap='jet', alpha=0.3)
        if cell_id is not None:
            cell = next((c for c in self.measurement_data.get('cells', []) 
                         if c.get('unique_cell_id') == cell_id), None)
            if cell and 'bbox' in cell:
                row_min, col_min, row_max, col_max = cell['bbox']
                plt.gca().add_patch(plt.Rectangle((col_min, row_min),
                                                  col_max - col_min,
                                                  row_max - row_min,
                                                  edgecolor='yellow',
                                                  facecolor='none',
                                                  linewidth=2))
                plt.title(f"Gating Overlay (h5_index={h5_index}, cell_id={cell_id})")
            else:
                plt.title(f"Gating Overlay (h5_index={h5_index}) - cell_id not found")
        else:
            plt.title(f"Gating Overlay (h5_index={h5_index})")
        plt.axis('off')
        plt.show()

    def display_cell_crops_by_group(self, group, n_cells=5, random_selection=True):
        cells = [cell for cell in self.measurement_data.get('cells', []) if cell.get('group_total') == group]
        if not cells:
            print(f"No cells found for total expression group '{group}'.")
            return
        if random_selection:
            selected_cells = random.sample(cells, min(n_cells, len(cells)))
        else:
            selected_cells = cells[:n_cells]
        for cell in selected_cells:
            h5_index = cell.get('h5_index')
            fov = cell.get('fov', 0)
            bbox = cell.get('bbox')
            if not bbox:
                continue
            images, _ = self.get_images_and_masks(h5_index)
            if images is None:
                continue
            img = images[fov, 0, :, :]
            row_min, col_min, row_max, col_max = bbox
            cell_crop = img[row_min:row_max, col_min:col_max]
            plt.figure(figsize=(6, 6))
            plt.imshow(cell_crop, cmap='gray')
            plt.title(f"Cell {cell.get('unique_cell_id')} (total = {group})")
            plt.axis('off')
            plt.show()

    def display_random_cells_with_TS(self, n_cells=5):
        ts_counts = self.clusters_df[self.clusters_df['is_nuc'] == 1].groupby('unique_cell_id').size()
        valid_ts_ids = set(ts_counts.index)
        cells = [cell for cell in self.measurement_data.get('cells', []) if cell.get('unique_cell_id') in valid_ts_ids]
        if not cells:
            print("No cells with TS found.")
            return
        selected_cells = random.sample(cells, min(n_cells, len(cells)))
        for cell in selected_cells:
            h5_index = cell.get('h5_index')
            fov = cell.get('fov', 0)
            bbox = cell.get('bbox')
            if not bbox:
                continue
            images, _ = self.get_images_and_masks(h5_index)
            if images is None:
                continue
            img = images[fov, 0, :, :]
            row_min, col_min, row_max, col_max = bbox
            cell_crop = img[row_min:row_max, col_min:col_max]
            plt.figure(figsize=(6, 6))
            plt.imshow(cell_crop, cmap='gray')
            plt.title(f"Cell {cell.get('unique_cell_id')} with TS")
            plt.axis('off')
            plt.show()

    def display_random_cells_with_foci(self, n_cells=5):
        foci_counts = self.clusters_df[self.clusters_df['is_nuc'] == 0].groupby('unique_cell_id').size()
        valid_foci_ids = set(foci_counts.index)
        cells = [cell for cell in self.measurement_data.get('cells', []) if cell.get('unique_cell_id') in valid_foci_ids]
        if not cells:
            print("No cells with foci found.")
            return
        selected_cells = random.sample(cells, min(n_cells, len(cells)))
        for cell in selected_cells:
            h5_index = cell.get('h5_index')
            fov = cell.get('fov', 0)
            bbox = cell.get('bbox')
            if not bbox:
                continue
            images, _ = self.get_images_and_masks(h5_index)
            if images is None:
                continue
            img = images[fov, 0, :, :]
            row_min, col_min, row_max, col_max = bbox
            cell_crop = img[row_min:row_max, col_min:col_max]
            plt.figure(figsize=(6, 6))
            plt.imshow(cell_crop, cmap='gray')
            plt.title(f"Cell {cell.get('unique_cell_id')} with foci")
            plt.axis('off')
            plt.show()

    def assign_expression_groups(self):
        cells = self.measurement_data.get('cells', [])
        if not cells:
            print("No cell data available to assign expression groups.")
            return
        total_values = []
        for cell in cells:
            if cell.get('num_nuc_spots') is not None and cell.get('num_cyto_spots') is not None:
                total = cell['num_nuc_spots'] + cell['num_cyto_spots']
                cell['total_mRNA'] = total
                total_values.append(total)
            else:
                cell['total_mRNA'] = None
        if not total_values:
            print("No valid total mRNA counts available.")
            return
        total_cutoff = np.percentile(total_values, 0.1)
        def total_group(x):
            if x is None:
                return "unknown"
            if x < 25 or x <= total_cutoff:
                return "low"
            elif 50 <= x <= 100:
                return "mid"
            elif x > 100 and x < 300:
                return "high"
            else:
                return "other"
        for cell in cells:
            cell['group_total'] = total_group(cell.get('total_mRNA'))
        print("Expression groups assigned based on total mRNA.")

#############################
# End of Module
#############################