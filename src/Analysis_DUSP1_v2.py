#!/usr/bin/env python3
"""
Combined Analysis Pipeline for DUSP1

Steps:
1. DUSP1AnalysisManager:
   - Manages HDF5 file access.
   - Extracts file paths from a log directory (if no direct locations are provided).
   - Provides methods to select an analysis (by name) and load datasets from HDF5 files.
   - Saves datasets as CSV.

2. SNRAnalysis:
   - Performs three SNR-based analyses:
       • Weighted thresholding (using the 'snr' column with a 20th percentile cutoff between 2 and 5).
       • MG SNR calculation: MG_SNR = (signal - cell_intensity_mean-0) / cell_intensity_std-0.
       • Absolute thresholding (if an absolute threshold is provided).
   - Also compares the original 'snr' with MG_SNR by adding comparison columns.

3. Compute statistics by time and concentration.
4. Visualize the MG SNR distribution.
5. Generate final cell-level measurements using DUSP1Measurement.
6. Optionally, visualize results with DisplayManager.
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
# SNRAnalysis Class
#############################
class SNRAnalysis:
    """
    This class performs three SNR-based analyses:
      1. Weighted Thresholding: Uses the 'snr' column from spots.
         - If snr < 2 → False.
         - If 2 <= snr < 5 → True if snr is at or above the 20th percentile in this range.
         - If snr >= 5 → True.
      2. MG SNR Calculation: Computed as:
             MG_SNR = (signal - cell_intensity_mean-0) / cell_intensity_std-0
         via merging the spots and cell properties DataFrames.
      3. Absolute Thresholding: If an absolute threshold value is provided,
         a boolean column is added that is True when the 'snr' is greater than or equal to that value.
         
    The output is the spots DataFrame with the following columns added:
        - 'MG_SNR': The computed MG SNR.
        - 'weighted': Whether the spot passes the weighted threshold.
        - 'absolute': Whether the spot passes the absolute threshold (if provided).
        - 'snr_vs_mg': The difference between the original 'snr' and the computed MG SNR.
        - 'mg_gt_snr': Whether the MG SNR is greater than the original 'snr'.
    """
    def __init__(self, spots, cellprops, clusters, abs_threshold=None, mg_snr_threshold=None):
        self.spots = spots.copy()
        self.clusters = clusters.copy()
        self.cellprops = cellprops.copy()
        self.abs_threshold = abs_threshold
        self.mg_snr_threshold = mg_snr_threshold
        
        # Merge the data into spots (and clusters if needed)
        self._merge_data()
        
        # Compute MG SNR.
        self._calculate_mg_snr()
        
        # Compute weighted threshold based on the original 'snr' values.
        self.weighted_cutoff = self._apply_weighted_threshold()
        
        # Apply absolute threshold if provided.
        if self.abs_threshold is not None:
            self._apply_absolute_threshold(self.abs_threshold)
        else:
            self.spots['absolute'] = False
        
        # Compare the original 'snr' to the computed MG SNR.
        self._compare_snr_to_mg()
        
    def _merge_data(self):
        # Add a unique cell id to cellprops.
        self.cellprops['unique_cell_id'] = np.arange(len(self.cellprops))
        
        # Merge the spots DataFrame with the relevant cell properties.
        self.spots = self.spots.merge(
            self.cellprops[['NAS_location', 'cell_label', 'fov', 'unique_cell_id', 
                             'cell_intensity_mean-0', 'cell_intensity_std-0']], 
            on=['NAS_location', 'cell_label', 'fov'], 
            how='left'
        )
        
        # Optionally, you can merge clusters as well if needed.
        self.clusters = self.clusters.merge(
            self.cellprops[['NAS_location', 'cell_label', 'fov', 'unique_cell_id', 
                             'cell_intensity_mean-0', 'cell_intensity_std-0']], 
            on=['NAS_location', 'cell_label', 'fov'], 
            how='left'
        )
    
    def _calculate_mg_snr(self):
        # Compute MG SNR and add as a new column in spots.
        self.spots['MG_SNR'] = (
            self.spots['signal'] - self.spots['cell_intensity_mean-0']
        ) / self.spots['cell_intensity_std-0'].replace(0, np.nan)
    
    def _apply_weighted_threshold(self):
        if 'snr' not in self.spots.columns:
            raise ValueError("Spots DataFrame must contain a 'snr' column for weighted thresholding.")
        # Identify spots in the 2 to 5 range.
        mask_2_5 = (self.spots['snr'] >= 2) & (self.spots['snr'] < 5)
        if mask_2_5.sum() > 0:
            weighted_cutoff = np.percentile(self.spots.loc[mask_2_5, 'snr'], 20)
        else:
            weighted_cutoff = 2.0
        
        # Apply the weighted threshold decision.
        def weighted_decision(snr):
            if snr < 2:
                return False
            elif 2 <= snr < 5:
                return snr >= weighted_cutoff
            else:
                return True
            
        self.spots['weighted'] = self.spots['snr'].apply(weighted_decision)
        return weighted_cutoff

    def _apply_absolute_threshold(self, abs_threshold):
        # Create a boolean column based on the provided absolute threshold.
        self.spots['absolute'] = self.spots['snr'] >= abs_threshold

    def _compare_snr_to_mg(self):
        # Compute the difference between the original snr and the MG_SNR.
        self.spots['snr_vs_mg'] = self.spots['snr'] - self.spots['MG_SNR']
        # Determine if MG_SNR is greater than the original snr.
        self.spots['mg_gt_snr'] = (self.spots['snr'] < self.spots['MG_SNR']).astype(int)

    def get_results(self):
        """Return the spots DataFrame with all computed columns."""
        return self.spots

    def plot_mg_snr_distribution(self, threshold=None, bins=50):
        if 'MG_SNR' not in self.spots.columns:
            raise ValueError("MG_SNR column not found in spots DataFrame.")
        plt.figure(figsize=(8, 6))
        plt.hist(self.spots['MG_SNR'].dropna(), bins=bins, alpha=0.7, edgecolor='black')
        plt.xlabel("MG SNR")
        plt.ylabel("Count")
        plt.title("Distribution of MG SNR")
        if threshold is not None:
            plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f"Threshold = {threshold}")
            plt.legend()
        plt.show()

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

class DUSP1Measurement:
    """
    Processes spot, cluster, and cell property data to produce cell-level measurements for DUSP1.
    
    Computes three SNR metrics for each spot:
      1) Weighted: Uses a weighted cutoff for spots with snr between 2 and 5.
      2) Absolute: Boolean flag indicating if snr >= snr_threshold.
      3) MG: Computed as (signal - cell_intensity_mean-0) / cell_intensity_std-0,
         and MG_count is the number of spots for which MG_SNR is greater than the weighted snr threshold.
      
    Each spot (and thus each cell after aggregation) gets a unique_cell_id from cellprops.
    """
    def __init__(self, spots: pd.DataFrame, clusters: pd.DataFrame, cellprops: pd.DataFrame):
        self.spots = spots.copy()
        self.clusters = clusters.copy()
        self.cellprops = cellprops.copy()
        # Assign a unique cell ID to each row in cellprops.
        self.cellprops['unique_cell_id'] = np.arange(len(self.cellprops))
    
    def merge_data(self):
        # Ensure consistent data types for the merge keys.
        for df in [self.spots, self.clusters, self.cellprops]:
            df['NAS_location'] = df['NAS_location'].astype(str)
            df['cell_label'] = df['cell_label'].astype(int)
            df['fov'] = df['fov'].astype(int)
        
        # Merge spots and clusters with cellprops to add the unique_cell_id.
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
        unique_vals = np.sort(series.dropna().unique())
        if len(unique_vals) < 2:
            return 0
        return unique_vals[-2]
    
    def measure(self, snr_threshold: float) -> pd.DataFrame:
        """
        Processes the data to produce cell-level measurements.
        
        For each spot:
          - weighted_pass: Boolean flag using a weighted cutoff (for snr between 2 and 5).
          - absolute_pass: Boolean flag if snr >= snr_threshold.
          - MG_SNR: Computed as (signal - cell_intensity_mean-0) / cell_intensity_std-0.
          
        At the cell level, the following are aggregated:
          - weighted_count: Count of spots passing the weighted filter.
          - absolute_count: Count of spots passing the absolute threshold.
          - MG_count: Count of spots where MG_SNR > weighted snr threshold.
          
        Other cell-level metrics (from clusters and cellprops) are also included.
        """
        # Merge data so that every spot gets a unique_cell_id.
        props, spots, clusters = self.merge_data()
        
        # Merge cell intensity columns into spots so that MG SNR can be computed directly.
        spots = spots.merge(
            props[['unique_cell_id', 'cell_intensity_mean-0', 'cell_intensity_std-0']],
            on='unique_cell_id', how='left'
        )
        
        # --- Compute Weighted SNR Flag ---
        mask_2_5 = (spots['snr'] >= 2) & (spots['snr'] < 5)
        weighted_cutoff = (np.percentile(spots.loc[mask_2_5, 'snr'], 20)
                           if mask_2_5.sum() > 0 else 2.0)
        
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
        epsilon = 1e-6
        spots['MG_SNR'] = (spots['signal'] - spots['cell_intensity_mean-0']) / \
                        (spots['cell_intensity_std-0'] + epsilon)
        
        # --- Aggregate to Cell-Level Metrics ---
        # Sort dataframes by unique_cell_id for consistent aggregation.
        spots = spots.sort_values(by='unique_cell_id')
        clusters = clusters.sort_values(by='unique_cell_id')
        props = props.sort_values(by='unique_cell_id')
        cell_ids = props['unique_cell_id']
        
        # Calculate counts from spots:
        weighted_count = spots.groupby('unique_cell_id')['weighted_pass'].sum().reindex(cell_ids, fill_value=0)
        absolute_count = spots.groupby('unique_cell_id')['absolute_pass'].sum().reindex(cell_ids, fill_value=0)
        mg_count = spots.groupby('unique_cell_id')['MG_SNR'].apply(lambda x: (x > weighted_cutoff).sum())\
                        .reindex(cell_ids, fill_value=0)
        
        # (Optional) You may keep additional aggregation from clusters or other metrics here.
        # For example, the following are left from your original design:
        num_spots = spots.groupby('unique_cell_id').size().reindex(cell_ids, fill_value=0)
        num_nuc_spots = spots[spots['is_nuc'] == 1].groupby('unique_cell_id').size()\
                            .reindex(cell_ids, fill_value=0)
        num_cyto_spots = spots[spots['is_nuc'] == 0].groupby('unique_cell_id').size()\
                             .reindex(cell_ids, fill_value=0)
        
        # Other cell properties from props:
        nuc_area = props['nuc_area']
        cyto_area = props['cyto_area']
        avg_nuc_int = props['nuc_intensity_mean-0']
        avg_cyto_int = props['cyto_intensity_mean-0']
        avg_cell_int = props['cell_intensity_mean-0']
        std_cell_int = props['cell_intensity_std-0']
        time = props['time']
        dex_conc = props['Dex_Conc']
        replica = spots.groupby('unique_cell_id')['replica'].first().reindex(cell_ids, fill_value=np.nan)
        fov = props['fov']
        nas_location = props['NAS_location']
        h5_idx = props['h5_idx']
        
        # Assemble the cell-level results data frame.
        results = pd.DataFrame({
            'cell_id': cell_ids,
            'weighted_count': weighted_count.values,
            'absolute_count': absolute_count.values,
            'MG_count': mg_count.values,
            'num_spots': num_spots.values,
            'num_nuc_spots': num_nuc_spots.values,
            'num_cyto_spots': num_cyto_spots.values,
            'nuc_area_px': nuc_area.values,
            'cyto_area_px': cyto_area.values,
            'avg_nuc_int': avg_nuc_int.values,
            'avg_cyto_int': avg_cyto_int.values,
            'avg_cell_int': avg_cell_int.values,
            'std_cell_int': std_cell_int.values,
            'time': time.values,
            'dex_conc': dex_conc.values,
            'replica': replica.values,
            'fov': fov.values,
            'nas_location': nas_location.values,
            'h5_idx': h5_idx.values
        })
        
        return results
    
    def save_results(self, results: pd.DataFrame, csv_path: str):
        """Save the cell-level results to a CSV file."""
        results.to_csv(csv_path, index=False)
        print(f"Saved cell-level results to {csv_path}")

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