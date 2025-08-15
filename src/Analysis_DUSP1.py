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
import seaborn as sns
from skimage import exposure
import matplotlib.colors as mcolors
from skimage.measure import find_contours
from tqdm import tqdm
import joblib
from sklearn.ensemble import RandomForestClassifier
import time
from typing import List, Optional, Union
from scipy.stats import gaussian_kde

#############################
# DUSP1AnalysisManager Class
#############################

# Utility function to safely open HDF5 files with retries
def safe_open_h5(file_path, retries=3, delay=1.0):
    for attempt in range(1, retries + 1):
        try:
            return h5py.File(file_path, 'r')
        except OSError as e:
            print(f"[Retry {attempt}] Failed to open {file_path}: {e}")
            time.sleep(delay)
    raise RuntimeError(f"Failed to open file after {retries} retries: {file_path}")

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

        for obj in list(self.analysis_names):  # Make a copy to avoid modification during iteration
            parent = obj.parent
            name = obj.name.split('/')[-1]  # Get the local name within the parent group
            try:
                del parent[name]
                print(f"Deleted '{name}' from {parent.file.filename}")
            except Exception as e:
                print(f"Failed to delete '{name}' from {parent.file.filename}: {e}")

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
        - 'MG_pass': Whether the spot passes the MG SNR threshold (if provided).
        - 'snr_vs_mg': The difference between the original 'snr' and the computed MG SNR.
        - 'mg_gt_snr': Whether the MG SNR is greater than the original 'snr'.
    """
    def __init__(self, spots, cellprops, clusters, abs_threshold=None, mg_threshold=None):
        self.spots = spots.copy()
        self.clusters = clusters.copy()
        self.cellprops = cellprops.copy()
        self.abs_threshold = abs_threshold
        self.mg_threshold = mg_threshold
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

        # Apply MG SNR threshold.
        self._apply_mg_threshold(self.mg_threshold)

        # Optionally, you can return the results here.
        self.get_results()
        
    def _merge_data(self):
        # Add a unique cell id to cellprops.
        self.cellprops['unique_cell_id'] = np.arange(len(self.cellprops))
        # Add a unique spot id to spots.
        self.spots['unique_spot_id'] = np.arange(len(self.spots))
        # Add a unique spot id to clusters.
        self.clusters['unique_cluster_id'] = np.arange(len(self.clusters))
        
        # Merge the spots DataFrame with the relevant cell properties.
        self.spots = self.spots.merge(
            self.cellprops[['NAS_location', 'cell_label', 'fov', 'unique_cell_id',
                            'cell_intensity_mean-0', 'cell_intensity_std-0', 'nuc_intensity_mean-0', 'nuc_intensity_std-0',
                            'cyto_intensity_mean-0', 'cyto_intensity_std-0']], 
            on=['NAS_location', 'cell_label', 'fov'], 
            how='left'
        )
        
        # Optionally, merge clusters if needed.
        self.clusters = self.clusters.merge(
            self.cellprops[['NAS_location', 'cell_label', 'fov', 'unique_cell_id',
                            'cell_intensity_mean-0', 'cell_intensity_std-0', 'nuc_intensity_mean-0', 'nuc_intensity_std-0',
                            'cyto_intensity_mean-0', 'cyto_intensity_std-0']], 
            on=['NAS_location', 'cell_label', 'fov'], 
            how='left'
        )
    
    def _calculate_mg_snr(self):
        epsilon = 1e-6
        # Initialize new columns with NaN
        self.spots['MG_SNR'] = np.nan

        # Compute nuclear MG_SNR for nuclear spots.
        nuc_mask = self.spots['is_nuc'] == 1
        self.spots.loc[nuc_mask, 'MG_SNR'] = (
            self.spots.loc[nuc_mask, 'signal'] - self.spots.loc[nuc_mask, 'nuc_intensity_mean-0']
        ) / (self.spots.loc[nuc_mask, 'nuc_intensity_std-0'] + epsilon)

        # Compute cytoplasmic MG_SNR for cytoplasmic spots.
        cyto_mask = self.spots['is_nuc'] == 0
        self.spots.loc[cyto_mask, 'MG_SNR'] = (
            self.spots.loc[cyto_mask, 'signal'] - self.spots.loc[cyto_mask, 'cyto_intensity_mean-0']
        ) / (self.spots.loc[cyto_mask, 'cyto_intensity_std-0'] + epsilon)
    
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
        # Determine if MG_SNR is less than the original snr.
        self.spots['mg_lt_snr'] = (self.spots['snr'] > self.spots['MG_SNR']).astype(int)

    def _apply_mg_threshold(self, mg_threshold):
        # Create a boolean column based on the provided MG SNR threshold.
        self.spots['MG_pass'] = self.spots['MG_SNR'] >= mg_threshold

    def get_results(self):
        """Return the spots DataFrame with all computed columns."""
        return self.spots, self.clusters, self.cellprops

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

#############################
# DUSP1Measurement Class
#############################

class DUSP1Measurement:
    """
    Processes smFISH spot, cluster, and cell property data to compute cell-level measurements
    for DUSP1 expression.

    This class aggregates spot-level metrics using three signal quality filters:
      - weighted: Custom boolean filter (e.g., based on snr range or heuristic).
      - absolute: Spots with SNR >= snr_threshold.
      - MG_SNR: Spots with modified Gaussian SNR,
        defined as (signal - cell_intensity_mean-0) / cell_intensity_std-0,
        where MG_count counts spots exceeding mg_threshold.

    Aggregates these metrics by cell (using 'unique_cell_id') with separate counts
    for nuclear and cytoplasmic compartments. Also summarizes nuclear transcriptional
    sites (TS), cytoplasmic foci, and relevant cell properties from image-derived data.

    Outputs include:
      - Spot counts passing each filter (total, nuclear, cytoplasmic).
      - Counts and sizes of clusters (TS and foci).
      - Cell morphology and intensity statistics.
      - Metadata (time, Dex concentration, replica, FOV, etc.).
    """

    def __init__(
        self,
        spots: pd.DataFrame,
        clusters: pd.DataFrame,
        cellprops: pd.DataFrame,
        model: Union[str, RandomForestClassifier] = None,
        is_tpl: bool = False,
        tpl_time_list: List[int] = None,
        dex_rel2_tpl_time: List[int] = None,
    ):
        self.spots = spots.copy()
        self.clusters = clusters.copy()
        self.cellprops = cellprops.copy()
        self.is_tpl = is_tpl
        self.tpl_time_list = tpl_time_list
        self.dex_rel2_tpl_time = dex_rel2_tpl_time

        # Optional Random Forest classifier:
        # - If you pass in an estimator, just grab it
        # - If you pass in a path, try to load it
        self.clf = None
        if isinstance(model, RandomForestClassifier):
            self.clf = model
            print("Using provided RandomForestClassifier instance.")
        elif isinstance(model, str):
            if os.path.isfile(model):
                try:
                    self.clf = joblib.load(model)
                    print(f"Loaded RF model from: {model}")
                except Exception as e:
                    print(f"Warning: could not load model at {model}: {e}")
            else:
                print(f"Model file not found at: {model}. Continuing without RF.")    

    @staticmethod
    def second_largest(series):
        """Return the second-largest unique value in the series; if not available, return 0."""
        unique_vals = np.sort(series.dropna().unique())
        if len(unique_vals) < 2:
            return 0
        return unique_vals[-2]
    
    def measure(self, abs_threshold: float, mg_threshold: float) -> pd.DataFrame:
        """
        Processes the data to produce cell-level measurements.
        
        For each spot:
          - weighted_pass: Boolean flag using a weighted cutoff (for snr between 2 and 5).
          - absolute_pass: Boolean flag if snr >= snr_threshold.
          - MG_SNR: Computed as (signal - nuc or cyto_intensity_mean-0) / nuc or cyto_intensity_std-0.
          
        At the cell level, the following are aggregated:
          - weighted_count: Count of spots passing the weighted filter.
          - absolute_count: Count of spots passing the snr_threshold.
          - MG_count: Count of spots where MG_SNR >= mg_threshold .
          
        Other cell-level metrics (from clusters and cellprops) are also included.
        """
        # — Predict on every spot, regardless of manual_label —
        # — RF prediction on all spots, using exactly the features the model expects —
        if getattr(self, 'clf', None) is not None:
            feature_names = list(self.clf.feature_names_in_)
            # build X_rf with the right columns in the right order
            X_rf = pd.DataFrame(
                {f: (self.spots[f] if f in self.spots.columns 
                    else np.zeros(len(self.spots), dtype=float))
                for f in feature_names},
                index=self.spots.index
            )
            # now run predict
            self.spots['rf_prediction'] = self.clf.predict(X_rf)

        # --- Aggregate to Cell-Level Metrics ---
        # Sort dataframes by unique_cell_id for consistent aggregation.
        self.spots = self.spots.sort_values(by='unique_cell_id')
        self.clusters = self.clusters.sort_values(by='unique_cell_id')
        self.cellprops = self.cellprops.sort_values(by='unique_cell_id')
        cell_ids = self.cellprops['unique_cell_id']
        
        # Calculate counts from spots:
        weighted_count = self.spots.groupby('unique_cell_id')['weighted'].sum().reindex(cell_ids, fill_value=0)
        nuc_weighted_count = self.spots[self.spots['is_nuc'] == 1].groupby('unique_cell_id')['weighted'].sum()\
                            .reindex(cell_ids, fill_value=0)
        cyto_weighted_count = self.spots[self.spots['is_nuc'] == 0].groupby('unique_cell_id')['weighted'].sum()\
                            .reindex(cell_ids, fill_value=0)
        absolute_count = self.spots.groupby('unique_cell_id')['absolute'].sum().reindex(cell_ids, fill_value=0)
        nuc_absolute_count = self.spots[self.spots['is_nuc'] == 1].groupby('unique_cell_id')['absolute'].sum()\
                            .reindex(cell_ids, fill_value=0)
        cyto_absolute_count = self.spots[self.spots['is_nuc'] == 0].groupby('unique_cell_id')['absolute'].sum()\
                            .reindex(cell_ids, fill_value=0)
        mg_count = self.spots.groupby('unique_cell_id')['MG_SNR'].apply(lambda x: (x >= mg_threshold).sum())\
                        .reindex(cell_ids, fill_value=0)
        nuc_mg_count = self.spots[self.spots['is_nuc'] == 1].groupby('unique_cell_id')['MG_SNR']\
                            .apply(lambda x: (x >= mg_threshold).sum()).reindex(cell_ids, fill_value=0)
        cyto_mg_count = self.spots[self.spots['is_nuc'] == 0].groupby('unique_cell_id')['MG_SNR']\
                            .apply(lambda x: (x >= mg_threshold).sum()).reindex(cell_ids, fill_value=0)
        
        # Metrics from clusters:
        num_ts = self.clusters[self.clusters['is_nuc'] == 1].groupby('unique_cell_id').size().reindex(cell_ids, fill_value=0)
        num_foci = self.clusters[self.clusters['is_nuc'] == 0].groupby('unique_cell_id').size().reindex(cell_ids, fill_value=0)
        num_spots_ts = self.clusters[self.clusters['is_nuc'] == 1].groupby('unique_cell_id')['nb_spots'].sum().reindex(cell_ids, fill_value=0)
        largest_ts = self.clusters[self.clusters['is_nuc'] == 1].groupby('unique_cell_id')['nb_spots'].max().reindex(cell_ids, fill_value=0)
        second_largest_ts = self.clusters[self.clusters['is_nuc'] == 1].groupby('unique_cell_id')['nb_spots']\
                              .apply(DUSP1Measurement.second_largest).reindex(cell_ids, fill_value=0)
        num_spots_foci = self.clusters[self.clusters['is_nuc'] == 0].groupby('unique_cell_id')['nb_spots'].sum().reindex(cell_ids, fill_value=0)

        # Left from original method:
        num_spots = self.spots.groupby('unique_cell_id').size().reindex(cell_ids, fill_value=0)
        num_nuc_spots = self.spots[self.spots['is_nuc'] == 1].groupby('unique_cell_id').size()\
                            .reindex(cell_ids, fill_value=0)
        num_cyto_spots = self.spots[self.spots['is_nuc'] == 0].groupby('unique_cell_id').size()\
                             .reindex(cell_ids, fill_value=0)
        
        # Random Forest prediction count:
        if 'rf_prediction' in self.spots.columns:
            RF_count = self.spots.groupby('unique_cell_id')['rf_prediction'] \
                                .sum().reindex(cell_ids, fill_value=0)
            nuc_RF_count = self.spots[self.spots['is_nuc'] == 1] \
                                .groupby('unique_cell_id')['rf_prediction'] \
                                .sum().reindex(cell_ids, fill_value=0)
            cyto_RF_count = self.spots[self.spots['is_nuc'] == 0] \
                                .groupby('unique_cell_id')['rf_prediction'] \
                                .sum().reindex(cell_ids, fill_value=0)
        else:
            RF_count = pd.Series(index=cell_ids, data=np.nan)
            nuc_RF_count = pd.Series(index=cell_ids, data=np.nan)
            cyto_RF_count = pd.Series(index=cell_ids, data=np.nan)

        # Other cell properties from props:
        nuc_area = self.cellprops['nuc_area']
        cyto_area = self.cellprops['cyto_area']
        avg_nuc_int = self.cellprops['nuc_intensity_mean-0']
        avg_cyto_int = self.cellprops['cyto_intensity_mean-0']
        avg_cell_int = self.cellprops['cell_intensity_mean-0']
        std_cell_int = self.cellprops['cell_intensity_std-0']
        time = self.cellprops['time']
        dex_conc = self.cellprops['Dex_Conc']
        replica = self.spots.groupby('unique_cell_id')['replica'].first().reindex(cell_ids, fill_value=np.nan)
        fov = self.cellprops['fov']
        nas_location = self.cellprops['NAS_location']
        h5_idx = self.cellprops['h5_idx']
        touching_border = self.cellprops['touching_border']
        
        # Assemble the cell-level results data frame.
        results = pd.DataFrame({
            'unique_cell_id': cell_ids,
            'weighted_count': weighted_count.values,
            'nuc_weighted_count': nuc_weighted_count.values,
            'cyto_weighted_count': cyto_weighted_count.values,
            'absolute_count': absolute_count.values,
            'nuc_absolute_count': nuc_absolute_count.values,
            'cyto_absolute_count': cyto_absolute_count.values,
            'MG_count': mg_count.values,
            'nuc_MG_count': nuc_mg_count.values,
            'cyto_MG_count': cyto_mg_count.values,
            'RF_count': RF_count.values,
            'nuc_RF_count': nuc_RF_count.values,
            'cyto_RF_count': cyto_RF_count.values,
            'num_spots': num_spots.values,
            'num_nuc_spots': num_nuc_spots.values,
            'num_cyto_spots': num_cyto_spots.values,
            'num_ts': num_ts.values,
            'num_foci': num_foci.values,
            'num_spots_foci': num_spots_foci.values,
            'num_spots_ts': num_spots_ts.values,
            'largest_ts': largest_ts.values,
            'second_largest_ts': second_largest_ts.values,
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
            'h5_idx': h5_idx.values,
            'touching_border': touching_border.values
        })
        # ── Now tack on the TPL columns if requested ───────────────────────────────
        if self.is_tpl:
            # 1) decide how to get time_tpl:
            if 'time_TPL' in self.cellprops.columns:
                # pull directly from cellprops
                tpl_ser = (
                    self.cellprops
                        .set_index('unique_cell_id')['time_TPL']
                        .reindex(results['unique_cell_id'])
                )
            else:
                # fallback: match via the three lists you provided
                if not (self.tpl_time_list and self.tpl_dex_list and self.tpl_time_tpl_list):
                    raise ValueError(
                        "When no cellprops['time_TPL'], you must pass tpl_time_list, "
                        "tpl_dex_list, and tpl_time_tpl_list to __init__"
                    )
                # build a lookup DataFrame
                tpl_map = pd.DataFrame({
                    'time':   self.tpl_time_list,
                    'dex':    self.tpl_dex_list,
                    'time_tpl': self.tpl_time_tpl_list
                }).drop_duplicates()
                # merge on (time, dex)
                merged = (
                    results[['unique_cell_id', 'time', 'dex_conc']]
                        .merge(tpl_map, left_on=['time','dex_conc'],
                                        right_on=['time','dex'],
                                        how='left')
                )
                tpl_ser = merged['time_tpl'].values

            # 2) assign it and build your five flags
            results['time_tpl'] = tpl_ser
            for idx, t in enumerate((0, 20, 75, 150, 180), start=1):
                results[f'tryptCond{idx}'] = (results['time_tpl'] == t).astype(int)

        return results
    
    def save_results(self, results: pd.DataFrame, csv_path: str):
        """Save the cell-level results to a CSV file."""
        results.to_csv(csv_path, index=False)
        print(f"Saved cell-level results to {csv_path}")

    def summarize_filtered_cells(self) -> pd.DataFrame:
        """
        Generate cell-level summary metrics based on already-filtered spots, clusters, and cellprops.
        Assumes all inputs have already been filtered for a specific thresholding method.

        Returns:
            pd.DataFrame: One row per cell with counts of nuclear/cytoplasmic spots,
                        clusters, intensity stats, and metadata.
        """
        # deterministic order
        self.spots     = self.spots.sort_values(by='unique_cell_id')
        self.clusters  = self.clusters.sort_values(by='unique_cell_id')
        self.cellprops = self.cellprops.sort_values(by='unique_cell_id')
        cell_ids = self.cellprops['unique_cell_id']

        # Spot-based counts
        num_spots      = self.spots.groupby('unique_cell_id').size().reindex(cell_ids, fill_value=0)
        num_nuc_spots  = (self.spots[self.spots['is_nuc'] == 1]
                        .groupby('unique_cell_id').size().reindex(cell_ids, fill_value=0))
        num_cyto_spots = (self.spots[self.spots['is_nuc'] == 0]
                        .groupby('unique_cell_id').size().reindex(cell_ids, fill_value=0))

        # Cluster-based counts
        ts = self.clusters[self.clusters['is_nuc'] == 1]
        fc = self.clusters[self.clusters['is_nuc'] == 0]
        num_ts         = ts.groupby('unique_cell_id').size().reindex(cell_ids, fill_value=0)
        num_foci       = fc.groupby('unique_cell_id').size().reindex(cell_ids, fill_value=0)
        num_spots_ts   = ts.groupby('unique_cell_id')['nb_spots'].sum().reindex(cell_ids, fill_value=0)
        largest_ts     = ts.groupby('unique_cell_id')['nb_spots'].max().reindex(cell_ids, fill_value=0)
        second_largest_ts = (ts.groupby('unique_cell_id')['nb_spots']
                            .apply(self.second_largest).reindex(cell_ids, fill_value=0))
        num_spots_foci = fc.groupby('unique_cell_id')['nb_spots'].sum().reindex(cell_ids, fill_value=0)

        # Assemble (keep your existing names for areas)
        data = {
            'unique_cell_id': cell_ids.values,
            'num_spots': num_spots.values,
            'num_nuc_spots': num_nuc_spots.values,
            'num_cyto_spots': num_cyto_spots.values,
            'num_ts': num_ts.values,
            'num_foci': num_foci.values,
            'num_spots_ts': num_spots_ts.values,
            'largest_ts': largest_ts.values,
            'second_largest_ts': second_largest_ts.values,
            'num_spots_foci': num_spots_foci.values,

            # areas 
            'nuc_area_px':  self.cellprops['nuc_area'].values,
            'cyto_area_px': self.cellprops['cyto_area'].values,

            # intensity stats (channel “-0” as in your measure())
            'avg_nuc_int':  self.cellprops['nuc_intensity_mean-0'].values,
            'avg_cyto_int': self.cellprops['cyto_intensity_mean-0'].values,
            'avg_cell_int': self.cellprops['cell_intensity_mean-0'].values,
            'std_cell_int': self.cellprops['cell_intensity_std-0'].values,

            # metadata
            'time': self.cellprops['time'].values,
            'dex_conc': self.cellprops['Dex_Conc'].values,
            'replica': self.cellprops['replica'].values if 'replica' in self.cellprops.columns
                    else self.spots.groupby('unique_cell_id')['replica'].first().reindex(cell_ids).values,
            'fov': self.cellprops['fov'].values,
            'nas_location': self.cellprops['NAS_location'].values if 'NAS_location' in self.cellprops.columns else np.nan,
            'h5_idx': self.cellprops['h5_idx'].values,
            'touching_border': self.cellprops['touching_border'].values,
        }

        # Only for TPL experiments, inject the extra columns
        if self.is_tpl and 'time_TPL' in self.cellprops.columns:
            time_tpl = (self.cellprops.set_index('unique_cell_id')['time_TPL']
                                .reindex(cell_ids))
            data['time_tpl'] = time_tpl.values
            for idx, t in enumerate((0, 20, 75, 150, 180), start=1):
                data[f'tryptCond{idx}'] = (time_tpl.values == t).astype(int)

        results = pd.DataFrame(data)
        return results        

#############################
# DUSP1_filtering Class
#############################

class DUSP1_filtering:
    def __init__(self, 
                method: str = 'MG',
                abs_threshold: float = 4.0,
                mg_threshold: float = 3.0,
                is_tpl: bool = False):
        self.method = method.lower()
        self.abs_threshold = abs_threshold
        self.mg_threshold = mg_threshold
        self.is_tpl = is_tpl

    def apply(self, results: pd.DataFrame) -> pd.DataFrame:
        filtered = results.loc[results['touching_border'] == False].copy()
        method_columns = {
            'weighted': ['weighted_count', 'nuc_weighted_count', 'cyto_weighted_count'],
            'absolute': ['absolute_count', 'nuc_absolute_count', 'cyto_absolute_count'],
            'mg':       ['MG_count', 'nuc_MG_count', 'cyto_MG_count'],
            'rf':       ['RF_count', 'nuc_RF_count', 'cyto_RF_count'],
            'none':     ['num_spots', 'num_nuc_spots', 'num_cyto_spots'],
            'mg_abs':   ['MG_count', 'nuc_MG_count', 'cyto_MG_count'],  # fallback to MG counts
        }
        method_columns['all'] = sum(method_columns.values(), [])

        if self.method not in method_columns:
            raise ValueError("Invalid method. Choose from 'weighted','absolute','mg','rf','none','mg_abs','all'.")

        common_columns = [
            'unique_cell_id', 'touching_border', 'num_ts', 'num_foci',
            'num_spots_foci', 'num_spots_ts', 'largest_ts', 'second_largest_ts',
            'nuc_area_px', 'cyto_area_px', 'avg_nuc_int', 'avg_cyto_int',
            'avg_cell_int', 'std_cell_int', 'time', 'dex_conc',
            'replica', 'fov', 'nas_location', 'h5_idx'
        ]

        cols = ['unique_cell_id', 'touching_border'] + method_columns[self.method]
        cols += [c for c in common_columns if c not in ('unique_cell_id', 'touching_border')]
        cols = [c for c in cols if c in filtered.columns]
        return filtered[cols]

    def apply_spots(self, spots: pd.DataFrame, results: pd.DataFrame = None, method: str = None) -> tuple:
        method = (method or self.method).lower()

        desired_columns = [
            'z_px','y_px','x_px','is_nuc','cluster_index','cell_label','snr','signal',
            'fov','FISH_Channel','condition','replica','time','Dex_Conc','NAS_location',
            'h5_idx','unique_spot_id','unique_cell_id','cell_intensity_mean-0','cell_intensity_std-0',
            'nuc_intensity_mean-0','nuc_intensity_std-0','cyto_intensity_mean-0','cyto_intensity_std-0',
            'MG_SNR'
        ]

        # Optional RF prediction
        if 'rf_prediction' in spots.columns:
            desired_columns.append('rf_prediction')

        if results is not None:
            valid = results.loc[results['touching_border'] == False, 'unique_cell_id']
            spots = spots[spots['unique_cell_id'].isin(valid)].copy()

        # Compute per-method keep flags
        spots['keep_mg']       = spots.get('MG_pass', False).astype(bool)
        spots['keep_absolute'] = spots.get('absolute', False).astype(bool)
        spots['keep_weighted'] = spots.get('weighted', False).astype(bool)

        if 'rf_prediction' in spots.columns:
            spots['keep_rf'] = spots['rf_prediction'].astype(bool)
        else:
            spots['keep_rf'] = False

        # mg_abs: MG_pass OR (snr > abs_threshold)
        spots['keep_mg_abs'] = spots['keep_mg'] | (
            ~spots['keep_mg'] &
            (spots['MG_SNR'] >= 1) &
            (spots['MG_SNR'] <= self.mg_threshold) &
            (spots['snr'] > self.abs_threshold)
)

        # Determine mask
        if method == 'mg':
            mask = spots['keep_mg']
        elif method == 'absolute':
            mask = spots['keep_absolute']
        elif method == 'weighted':
            mask = spots['keep_weighted']
        elif method == 'rf':
            mask = spots['keep_rf']
        elif method == 'none':
            mask = pd.Series(True, index=spots.index)
        elif method == 'mg_abs':
            mask = spots['keep_mg_abs']
        else:
            raise ValueError("Invalid method. Choose from 'mg','absolute','weighted','rf','none','mg_abs','all'.")

        # Subset filtered and removed
        filtered = spots[mask].copy()
        removed  = spots[~mask].copy()

        # Ensure expected columns exist for downstream measurement
        if method == 'weighted':
            filtered['weighted'] = True
            removed['weighted'] = False
        elif method == 'absolute':
            filtered['absolute'] = True
            removed['absolute'] = False
        elif method in ['mg', 'mg_abs']:
            # MG_SNR is already present
            pass
        elif method == 'rf':
            if 'rf_prediction' not in filtered.columns:
                filtered['rf_prediction'] = 0
            if 'rf_prediction' not in removed.columns:
                removed['rf_prediction'] = 0

        filtered = filtered[[c for c in desired_columns if c in filtered.columns]]
        removed  = removed[[c for c in desired_columns if c in removed.columns]]

        return filtered, removed

    def remove_partial_cells(self, clusters: pd.DataFrame, cellprops: pd.DataFrame) -> tuple:
        pcs = cellprops.loc[cellprops['touching_border'] == True, 'unique_cell_id']
        return (
            clusters.loc[~clusters['unique_cell_id'].isin(pcs)].copy(),
            cellprops.loc[~cellprops['unique_cell_id'].isin(pcs)].copy()
        )
    
    def apply_all(
        self,
        spots: pd.DataFrame,
        clusters: pd.DataFrame,
        cellprops: pd.DataFrame,
        results: pd.DataFrame = None
    ) -> tuple:
        """
        Filters spots, clusters, and cellprops according to the selected method,
        removes border-touching cells, and returns:
        - filtered_spots
        - filtered_clusters
        - filtered_cellprops
        - SSITcellresults (cell-level summary from DUSP1Measurement)
        - removed_spots (spots excluded by the filter)
        """

        # 1. Apply spot filtering
        filtered_spots, removed_spots = self.apply_spots(spots, results)

        # 2. Get valid (non-border, passing-filter) cell IDs
        valid_cell_ids = set(filtered_spots['unique_cell_id'])

        # 3. Filter clusters and cellprops
        filtered_clusters = clusters[clusters['unique_cell_id'].isin(valid_cell_ids)].copy()
        filtered_cellprops = cellprops[
            (cellprops['unique_cell_id'].isin(valid_cell_ids)) &
            (cellprops['touching_border'] == False)
        ].copy()

        # 4. Compute SSITcellresults using local DUSP1Measurement class
        measurer = DUSP1Measurement(
            spots=filtered_spots,
            clusters=filtered_clusters,
            cellprops=filtered_cellprops,
            is_tpl=self.is_tpl,
        )
        SSITcellresults = measurer.summarize_filtered_cells()

        # 5. Inject TPL columns into SSITcellresults if requested
        if self.is_tpl:
            # pull the TPL-addition time from filtered_cellprops
            tpl_times = (
                filtered_cellprops
                    .set_index('unique_cell_id')['time_TPL']
                    .reindex(SSITcellresults['unique_cell_id'])
            )
            SSITcellresults['time_tpl'] = tpl_times.values
            # build the five flags
            for idx, t in enumerate((0, 20, 75, 150, 180), start=1):
                SSITcellresults[f'tryptCond{idx}'] = (tpl_times.values == t).astype(int)

        return filtered_spots, filtered_clusters, filtered_cellprops, SSITcellresults, removed_spots

    
#############################
# DisplayManager Class
#############################

# # Global matplotlib settings for publication quality
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['axes.titlesize'] = 14
# plt.rcParams['axes.labelsize'] = 12

# Clean, minimal setup for fast and legible previews
plt.rcParams.update({
    'figure.dpi': 150,               # Slightly lower DPI for speed
    'font.family': 'Times New Roman',     # Easier to read on screen
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.edgecolor': 'gray',
    'axes.linewidth': 0.8,
    'figure.autolayout': True,       # Prevent cutoff
    'savefig.bbox': 'tight'
})

# Utility functions


def adjust_contrast(image, lower=2, upper=98):
    image = np.array(image)  # ensure it's a NumPy array
    if image.size > 0:
        p_low, p_high = np.percentile(image, (lower, upper))
        return exposure.rescale_intensity(image, in_range=(p_low, p_high), out_range=(0, 1))
    return image

def draw_spot_circle(ax, x, y, radius=4, color='gold', linewidth=2):
    """Draw a circle around a spot."""
    circle = plt.Circle((x, y), radius, edgecolor=color, facecolor='none', linewidth=linewidth)
    ax.add_patch(circle)  

def draw_spot_circle(ax, x, y, radius=4, color='gold', linewidth=2, alpha=1.0, zorder=3):
    """Draw a circle around a spot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x, y : float
        Center coordinates in pixels.
    radius : float
        Circle radius in pixels.
    color : str
        Edge color.
    linewidth : float
        Circle line width.
    alpha : float
        Opacity in [0, 1]; default 1 (opaque).
    zorder : int
        Draw order (higher = on top).
    """
    circle = plt.Circle(
        (x, y),
        radius,
        edgecolor=color,
        facecolor='none',
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(circle)
    return circle

def lowercase_columns(*dfs):
    """
    Convert all column names of provided DataFrames to lowercase.
    Returns a list of transformed DataFrames.
    """
    return [df.rename(columns=str.lower) for df in dfs]



class DUSP1DisplayManager_RemovalPercentage(DUSP1AnalysisManager):
    def __init__(self, analysis_manager, cell_level_results=None,
                 spots=None, clusters=None, cellprops=None, removed_spots= None,
                 h5_idx=None, fov=None, unique_cell_id=None):
        """
        Initialize the display manager.
        
        Parameters:
          - analysis_manager: instance of DUSP1AnalysisManager.
          - cell_level_results: DataFrame with cell-level measurements.
          - spots: DataFrame with spot-level data.
          - clusters: DataFrame with cluster-level data.
          - cellprops: DataFrame with cell property data.
          - removed_spots: DataFrame with removed spots.
          - h5_idx: (Optional)index of the HDF5 file to load for display.
          - fov: (Optional) field-of-view index within the HDF5 file for display.
          - unique_cell_id: (Optional) unique cell ID within the FOV for display.

          
        Note: This class does not load all images/masks into memory.
              Instead, it only loads the specific FOV on demand.
        """
        self.h5_file_paths = analysis_manager.location
        self.cellprops = (analysis_manager.select_datasets("cell_properties", dtype="dataframe")
                          if cellprops is None else cellprops)
        self.spots = (analysis_manager.select_datasets("spotresults", dtype="dataframe")
                      if spots is None else spots)
        self.clusters_df = (analysis_manager.select_datasets("clusterresults", dtype="dataframe")
                            if clusters is None else clusters)
        self.cell_level_results = cell_level_results
        self.removed_spots = removed_spots

        # Set default values for h5_idx, fov, and unique_cell_id.
        # If not provided, they will be set to None.
        if h5_idx is not None:
            self.h5_idx = h5_idx
        else:
            self.h5_idx = None
        if fov is not None:
                self.fov = fov
        else:
            self.fov = None
        if unique_cell_id is not None:
            self.unique_cell_id = unique_cell_id
        else:
            self.unique_cell_id = None
        # Ensure the analysis manager is initialized.
    
    def get_images_and_masks(self, h5_idx=None, fov=None):
        """
        Lazy load images and masks from a specific HDF5 file and FOV.

        Parameters:
        - h5_idx: index of the HDF5 file in self.h5_file_paths. If None, a random one is chosen.
        - fov: field-of-view index within the file. If None, a random FOV is chosen.

        Returns:
        - images: NumPy array for the specified FOV.
                Expected shape: (3, 27, 936, 640)
        - masks: NumPy array for the specified FOV.
                Expected shape: (3, 27, 936, 640)
        - h5_idx: the index of the HDF5 file used.
        - fov: the FOV index used.
        """
        if h5_idx is None:
            h5_idx = random.choice(range(len(self.h5_file_paths)))
        file_path = self.h5_file_paths[h5_idx]

        h5_file = None
        try:
            h5_file = safe_open_h5(file_path)  # Retry-safe wrapper
            num_fov = h5_file['raw_images'].shape[0]
            if fov is None:
                fov = random.choice(range(num_fov))

            images = np.array(h5_file['raw_images'][fov])
            masks = np.array(h5_file['masks'][fov])

            if images.shape[0] == 1:
                images = np.squeeze(images, axis=0)
            if masks.shape[0] == 1:
                masks = np.squeeze(masks, axis=0)

        finally:
            if h5_file is not None:
                try:
                    h5_file.close()
                except Exception as close_err:
                    print(f"WARNING: Error closing file {file_path}: {close_err}")

        return images, masks, h5_idx, fov

    def display_gating_overlay(self):
        """
        Display a full-FOV image with segmentation mask overlay.
        """
        Cyto_Channel = 1

        # Select a random h5_idx from the spots DataFrame.
        idx = np.random.choice(len(self.spots))
        spot_row = self.spots.iloc[idx]
        h5_idx, fov, nas_location = spot_row['h5_idx'], spot_row['fov'], spot_row['NAS_location']
        self.h5_idx = h5_idx
        self.fov = fov
        print(f"Selected H5 Index: {h5_idx}")
        print(f"Nas Location: {nas_location}")
        print(f"FOV: {fov}")

        # Load images and masks using lazy loading.
        images, masks, h5_idx, fov = self.get_images_and_masks(self.h5_idx, self.fov)
        # For gating, choose a color channel (here, channel 0) and perform max projection over the z axis.
        img = np.max(images[1], axis=0)  # images[0] has shape (27, 936, 640) -> (936, 640)
        mask = np.max(masks[1], axis=0)

        # Rescale intensity.
        p1, p99 = np.percentile(img, (1, 99))
        img = exposure.rescale_intensity(img, in_range=(p1, p99), out_range=(0, 1))

        # Filter DataFrames based on h5_idx, fov, and NAS location.
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
        clusters_frame = self.clusters_df[
            (self.clusters_df['h5_idx'] == h5_idx) &
            (self.clusters_df['fov'] == fov) &
            (self.clusters_df['NAS_location'] == nas_location)
        ]

        print(f"Spots DataFrame: {spots_frame.shape}")
        print(f"Cell Properties DataFrame: {cellprops_frame.shape}")
        print(f"Clusters DataFrame: {clusters_frame.shape}")

        # Identify kept cell labels from cellprops and those present in the mask.
        cell_labels_in_df = set(cellprops_frame['cell_label'].unique())
        cell_labels_in_mask = set(np.unique(mask))
        cell_labels_in_mask.discard(0)

        kept_colors = list(mcolors.TABLEAU_COLORS.values())
        removed_color = 'red'

        fig, ax = plt.subplots(figsize=(10, 10))
        # Display the cyto channel from the image (if needed, you can adjust which channel to display)
        ax.imshow(img, cmap='gray')

        # Plot kept cell outlines.
        color_map = {cell_label: kept_colors[i % len(kept_colors)]
                    for i, cell_label in enumerate(cell_labels_in_df)}

        for cell_label in cell_labels_in_df:
            cell_mask = (mask == cell_label)
            # Find contours in the 2D mask.
            contours = find_contours(cell_mask, 0.5)
            if contours:
                largest_contour = max(contours, key=lambda x: x.shape[0])
                ax.plot(largest_contour[:, 1], largest_contour[:, 0],
                        linewidth=2, color=color_map[cell_label],
                        label=f'Cell {cell_label}')

        # Plot discarded cell outlines.
        cell_labels_not_in_df = cell_labels_in_mask - cell_labels_in_df
        for cell_label in cell_labels_not_in_df:
            cell_mask = (mask == cell_label)
            contours = find_contours(cell_mask, 0.5)
            if contours:
                largest_contour = max(contours, key=lambda x: x.shape[0])
                ax.plot(largest_contour[:, 1], largest_contour[:, 0],
                        color=removed_color, linewidth=2, linestyle='dashed',
                        label=f'Removed: Cell {cell_label}')

        plt.legend()
        plt.show()
        return self.h5_idx, self.fov, nas_location

    def _display_zoom_on_cell(self, spotChannel, cytoChannel, nucChannel):
        """
        Zoom on bounding box, compute percentile-based contrast,
        then overlay:
          - gold circles for all 'regular' spots
          - chosen_spot in blue circle
          - TS in magenta arrows (with cluster size)
          - foci in cyan arrows
        """
        self.h5_idx = self.h5_idx
        self.fov = self.fov
        spotChannel = 0
        cytoChannel = 1
        nucChannel = 2

        valid_cells = self.cellprops[
            (self.cellprops['h5_idx'] == self.h5_idx) &
            (self.cellprops['fov'] == self.fov) &
            (self.cellprops['unique_cell_id'] != 0)
        ]['unique_cell_id'].unique()

        if len(valid_cells) == 0:
            print(f"No valid cells in FOV={self.fov}. Aborting.")
            return

        # Step 3: Find cells with TS (is_nuc == 1)
        cells_with_TS = self.clusters_df[
            (self.clusters_df['h5_idx'] == self.h5_idx) &
            (self.clusters_df['fov'] == self.fov) &
            (self.clusters_df['is_nuc'] == 1)
        ]['unique_cell_id'].unique()

        # Step 4: Find cells with Foci (is_nuc == -1)
        cells_with_Foci = self.clusters_df[
            (self.clusters_df['h5_idx'] == self.h5_idx) &
            (self.clusters_df['fov'] == self.fov) &
            (self.clusters_df['is_nuc'] == 0)
        ]['unique_cell_id'].unique()

        # Step 5: Find cells with Spot
        cells_with_Spot = self.spots[
            (self.spots['h5_idx'] == self.h5_idx) &
            (self.spots['fov'] == self.fov)
        ]['unique_cell_id'].unique()

        # Step 6: Apply selection logic
        # Priority: TS + Foci + Spot > TS + Spot > Spot > Any valid cell
        cells_with_TS_Foci_Spot = np.intersect1d(np.intersect1d(cells_with_TS, cells_with_Foci), cells_with_Spot)
        cells_with_TS_Spot = np.intersect1d(cells_with_TS, cells_with_Spot)
        cells_with_Spot_only = cells_with_Spot

        if len(cells_with_TS_Foci_Spot) > 0:
            self.cell_label = np.random.choice(cells_with_TS_Foci_Spot)
            print(f"Chose unique_cell_id={self.cell_label} with TS, Foci, and Spot (FOV={self.fov}).")
        elif len(cells_with_TS_Spot) > 0:
            self.cell_label = np.random.choice(cells_with_TS_Spot)
            print(f"Chose unique_cell_id={self.cell_label} with TS and Spot (FOV={self.fov}).")
        elif len(cells_with_Spot_only) > 0:
            self.cell_label = np.random.choice(cells_with_Spot_only)
            print(f"Chose unique_cell_id={self.cell_label} with Spot only (FOV={self.fov}).")
        else:
            # Final fallback: pick any valid cell
            self.cell_label = np.random.choice(valid_cells)
            print(f"No cell with TS, Foci, or Spot => picked random unique_cell_id={self.cell_label} (FOV={self.fov}).")        

        cdf = self.cellprops[
            (self.cellprops['h5_idx'] == self.h5_idx) &
            (self.cellprops['fov'] == self.fov) &
            (self.cellprops['unique_cell_id'] == self.cell_label)
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

        # Load images and masks using lazy loading.
        images, masks, h5_idx, fov = self.get_images_and_masks(self.h5_idx, self.fov)

        # Spot channel -> 3D -> convert to 2D by max-projection
        img_spot_3d = images[spotChannel, :, :, :]
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
        mask_nuc_3d = masks[nucChannel, :, :, :]
        mask_cyto_3d = masks[cytoChannel, :, :, :]
        mask_nuc_2d = np.max(mask_nuc_3d, axis=0)
        mask_cyto_2d = np.max(mask_cyto_3d, axis=0)

        crop_nucmask = mask_nuc_2d[row_min:row_max, col_min:col_max]
        crop_cytomask = mask_cyto_2d[row_min:row_max, col_min:col_max]

        # All spots in this cell
        cell_spots = self.spots[
            (self.spots['h5_idx'] == self.h5_idx) &
            (self.spots['fov'] == self.fov) &
            (self.spots['unique_cell_id'] == self.cell_label)
        ]

        # Transcription sites in this cell
        cell_TS = self.clusters_df[
            (self.clusters_df['h5_idx'] == self.h5_idx) &
            (self.clusters_df['fov'] == self.fov) &
            (self.clusters_df['unique_cell_id'] == self.cell_label) &
            (self.clusters_df['is_nuc'] == 1)
        ]

        # Foci in this cell
        cell_foci = self.clusters_df[
            (self.clusters_df['h5_idx'] == self.h5_idx) &
            (self.clusters_df['fov'] == self.fov) &
            (self.clusters_df['unique_cell_id'] == self.cell_label) &
            (self.clusters_df['is_nuc'] == 0)
        ]

        if cell_TS.empty:
            print('No TS in this cell')

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
            draw_spot_circle(axs[1], sx, sy, radius=4, color='magenta')
            cluster_size = getattr(tsrow, 'nb_spots', 1)  # fallback
            axs[1].text(sx - 12, sy, f"{cluster_size}", color='magenta', fontsize=10)

        # Mark foci in cyan arrows
        for _, frow in cell_foci.iterrows():
            sx = frow['x_px'] - dx
            sy = frow['y_px'] - dy
            draw_spot_circle(axs[1], sx, sy, radius=4, color='cyan')
            foci_size = getattr(frow, 'nb_spots', 1)
            axs[1].text(sx - 12, sy, f'{foci_size}', color='cyan', fontsize=10)

        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

        return self.chosen_spot, self.chosen_ts, self.chosen_foci

    def _display_zoom_on_one_spot(self, spotChannel):
        """
        Further zoom on the same chosen_spot. Also mark the chosen TS (magenta arrow).
        """
        if self.chosen_spot is None:
            print("No chosen_spot available for further zoom.")
            return

        spotChannel = 0
        self.h5_idx = self.h5_idx
        self.fov = self.fov
        self.chosen_spot = self.chosen_spot

        chosen_spot = self.chosen_spot
        sx = int(chosen_spot['x_px'])
        sy = int(chosen_spot['y_px'])

        pad = 15
        x1 = max(sx - pad, 0)
        x2 = sx + pad
        y1 = max(sy - pad, 0)
        y2 = sy + pad

        # Load images and masks using lazy loading.
        images, masks, h5_idx, fov = self.get_images_and_masks(self.h5_idx, self.fov)

        img_spot_3d = images[spotChannel, :, :, :]
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
                draw_spot_circle(ax, rx, ry, radius=4, color='magenta')
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
                draw_spot_circle(ax, rx, ry, radius=4, color='cyan')
                # Optionally annotate TS size:
                csize = getattr(self.chosen_foci, 'nb_spots', 1)
                ax.text(rx - 12, ry, f"{csize}", color='cyan', fontsize=10)

        # Mark other spots in gold if they are inside the patch
        cell_spots = self.spots[
            (self.spots['h5_idx'] == self.h5_idx) &
            (self.spots['fov'] == self.fov) &
            (self.spots['unique_cell_id'] == self.cell_label)
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

    def calculate_removal_percentage(self):
        """
        input 
        - spots DataFrame with kept spots
        - removed_spots DataFrame with removed spots
        - cellprops DataFrame with cell properties

        Calculates the percentage of spots removed for each cell.
        Calculates summary stats for each 'h5_idx', 'fov', and 'cell'.
            -  High: >= 80% removed
            -  Medium: 40% <= removed < 80%
            -  Low: < 40% removed

        Returns:
            - cellprops DataFrame with removal percentage and summary stats for each cell.
        """
        if self.removed_spots is None:
            print("No removed spots DataFrame provided.")
            return

        # Calculate the number of spots in each cell
        num_spots_per_cell = self.spots.groupby('unique_cell_id').size().reset_index(name='num_spots')

        # Merge with removed spots to get the number of removed spots per cell
        removed_spots_per_cell = self.removed_spots.groupby('unique_cell_id').size().reset_index(name='num_removed_spots')
        merged_df = pd.merge(num_spots_per_cell, removed_spots_per_cell, on='unique_cell_id', how='left')
        merged_df['num_removed_spots'] = merged_df['num_removed_spots'].fillna(0)

        # Calculate removal percentage
        merged_df['removal_percentage'] = (merged_df['num_removed_spots'] / merged_df['num_spots']) * 100

        # Add summary stats to the cellprops DataFrame
        self.cellprops = pd.merge(self.cellprops, merged_df[['unique_cell_id', 'removal_percentage']], on='unique_cell_id', how='left')

        # Categorize removal percentage into High, Medium, Low
        conditions = [
            (self.cellprops['removal_percentage'] >= 80),
            (self.cellprops['removal_percentage'] >= 40) & (self.cellprops['removal_percentage'] < 80),
            (self.cellprops['removal_percentage'] < 40)
        ]
        choices = ['High', 'Medium', 'Low']
        self.cellprops['summary_stats'] = np.select(conditions, choices, default='Unknown')

        return self.cellprops
    
    def display_representative_cells(self):
        """
        For each HDF5 index (h5_idx), display up to three representative cells – one for each removal category
        (preferably High, Medium, and Low). For each cell, the following will be displayed:
        1. A zoomed-in view of the cell (via _display_zoom_on_cell), with a title that includes additional info
            in the format: "conc_time_FOV{fov}_ID{unique_cell_id}".
        2. A zoomed-in view of one representative spot from that cell (via _display_zoom_on_one_spot),
            showing detected (gold) spots (and TS in magenta, foci in cyan).
        
        If a cell's metadata lacks a 'conc' or 'time' field, "NA" is shown.
        """
        if self.cellprops is None:
            print("No cell properties DataFrame provided.")
            return

        # Update cellprops with removal percentage and summary stats.
        self.calculate_removal_percentage()

        # Get unique h5_idx values.
        unique_h5_idx = self.cellprops['h5_idx'].unique()

        # Desired removal categories – you might prioritize High over Low.
        desired_categories = ["High", "Medium", "Low"]

        for h5 in unique_h5_idx:
            print(f"\nProcessing h5_idx {h5}")
            # Get cell properties for this h5_idx.
            cells_this_h5 = self.cellprops[self.cellprops['h5_idx'] == h5]
            for cat in desired_categories:
                # Filter for cells in the current removal category.
                cat_cells = cells_this_h5[cells_this_h5['summary_stats'] == cat]
                if cat_cells.empty:
                    print(f"No cells with removal category '{cat}' in h5_idx {h5}.")
                    continue

                # Select one representative cell at random.
                rep_cell = cat_cells.sample(1).iloc[0]
                print(f"Selected cell with removal '{cat}' from h5_idx {h5}.")

                # Set instance variables so that the zoom functions use the same FOV.
                self.h5_idx = rep_cell['h5_idx']
                self.fov = rep_cell['fov']
                # Here, we use the 'unique_cell_id' from the cellprops.
                self.cell_label = rep_cell['unique_cell_id']

                # Compose a title in the format "conc_time_FOV{fov}_ID{unique_cell_id}".
                conc = rep_cell.get('Dex_Conc', "NA")
                time_val = rep_cell.get('time', "NA")
                title_str = f"{conc}_{time_val}_FOV{rep_cell['fov']}_ID{rep_cell['unique_cell_id']}"

                # --- Plot 1: Zoomed-in view of the representative cell ---
                print(f"Displaying cell plot: {title_str}")
                # _display_zoom_on_cell() should generate its own figure.
                self._display_zoom_on_cell(spotChannel=0, cytoChannel=1, nucChannel=2)
                plt.gcf().suptitle(title_str)
                plt.show()

                # --- Prepare for Plot 2: Select a representative spot from this cell ---
                # Ensure the chosen spot is set by filtering for spots in this cell.
                cell_spots = self.spots[
                    (self.spots['h5_idx'] == self.h5_idx) &
                    (self.spots['fov'] == self.fov) &
                    (self.spots['unique_cell_id'] == self.cell_label)
                ]
                if cell_spots.empty:
                    print(f"No spots found for cell {self.cell_label} in FOV {self.fov}.")
                    continue
                # Randomly select one spot from the cell and assign it to self.chosen_spot.
                # self.chosen_spot = cell_spots.sample(1).iloc[0]
                self.chosen_spot = self.chosen_spot

                # Display TS and Foci for this cell.
                cell_clusters = self.clusters_df[
                    (self.clusters_df['h5_idx'] == self.h5_idx) &
                    (self.clusters_df['fov'] == self.fov) &
                    (self.clusters_df['unique_cell_id'] == self.cell_label)
                ]
                if not cell_clusters.empty:
                    # For demonstration, pick one TS if available.
                    ts_candidates = cell_clusters[cell_clusters['is_nuc'] == 1]
                    self.chosen_ts = ts_candidates.sample(1).iloc[0] if not ts_candidates.empty else None
                    # Pick one foci if available.
                    foci_candidates = cell_clusters[cell_clusters['is_nuc'] == 0]
                    self.chosen_foci = foci_candidates.sample(1).iloc[0] if not foci_candidates.empty else None
                else:
                    self.chosen_ts = None
                    self.chosen_foci = None

                # --- Plot 2: Zoom in on a representative spot in that cell ---
                print(f" Displaying spot zoom for cell ID {rep_cell['unique_cell_id']}.")
                self._display_zoom_on_one_spot(spotChannel=0)
        return self.cellprops

    def main_display(self):
        """
        Main display function that sequentially runs various display routines.
        """
        print("Running display_gating_overlay...")
        self.display_gating_overlay()
        
        print("Running _display_zoom_on_cell...")
        cell = self._display_zoom_on_cell(spotChannel=0, cytoChannel=1, nucChannel=2)
        if cell is not None:
            print("Running _display_zoom_on_one_spot...")
            self._display_zoom_on_one_spot(spotChannel=0)
        
        print("Running display_representative_cells...")
        self.display_representative_cells()

        print("All display routines completed.")


class DUSP1DisplayManager(DUSP1AnalysisManager):
    def __init__(self,
                 analysis_manager,
                 cell_level_results=None,
                 spots=None,
                 clusters=None,
                 cellprops=None,
                 removed_spots=None,
                 h5_idx=None,
                 fov=None,
                 unique_cell_id=None,
                 spot_channel=0,
                 cyto_channel=1,
                 nuc_channel=2,
                 rng_seed=None,
                 align_h5_to_manager=True, 
                 store_original_h5=True):
        """
        Construct a display helper on top of an already-initialized analysis.

        Purpose
        -------
        - Provide **read-only, on-demand** access to images/masks for specific FOVs.
        - Centralize state for display routines (current h5_idx/fov/cell, chosen TS/foci/spot).
        - Keep memory + file IO safe: **no preloading of entire HDF5 files**.

        Inputs
        ------
        analysis_manager : DUSP1AnalysisManager
            The analysis object you already used to select datasets.
            Used here ONLY to read table paths and (if needed) fetch DataFrames.
        cell_level_results : pd.DataFrame | None
            Optional cell-level metrics (not required for displays).
        spots, clusters, cellprops, removed_spots : pd.DataFrame | None
            If None, they are loaded via:
              - "spotresults", "clusterresults", "cell_properties" from `analysis_manager`.
            Assumes **consistent upstream dtypes** for keys (h5_idx, fov, unique_cell_id, etc.).
        h5_idx, fov, unique_cell_id : int | None
            Optional initial selection context for displays.
        spot_channel, cyto_channel, nuc_channel : int
            Channel indices used throughout displays for consistency.
        rng_seed : int | None
            Seed for deterministic sampling when selecting example FOVs/cells.

        Behavior & Assumptions
        ----------------------
        - **No dtype coercion** is performed here; we trust upstream processing.
        - No persistent HDF5 handles are kept; callers must use helpers that open/close safely.
        - This class stores **current selection state** and **picked rows** (TS/foci/spot)
          so downstream zoom/crop functions can reuse them in the same context.

        Side Effects
        ------------
        - Sets properties like `self.h5_idx`, `self.fov`, `self.cell_label`, etc.
        - Initializes placeholders: `chosen_spot`, `chosen_ts`, `chosen_foci`,
          `regular_nuc_spot`, `regular_cyto_spot` to `None`.
        """

        # ---- dataset locations / tables (lazy-loaded via analysis_manager if not provided) ----
        self.h5_file_paths = analysis_manager.location

        self.cellprops = (analysis_manager.select_datasets("cell_properties", dtype="dataframe")
                          if cellprops is None else cellprops)
        self.spots = (analysis_manager.select_datasets("spotresults", dtype="dataframe")
                      if spots is None else spots)
        self.clusters_df = (analysis_manager.select_datasets("clusterresults", dtype="dataframe")
                            if clusters is None else clusters)
        self.cell_level_results = cell_level_results
        self.removed_spots = removed_spots

        # ---- channel configuration (used consistently across displays) ----
        self.SPOT_CH = int(spot_channel)
        self.CYTO_CH = int(cyto_channel)
        self.NUC_CH  = int(nuc_channel)

        # ---- current selection context (may be overridden per call) ----
        self.h5_idx = int(h5_idx) if h5_idx is not None else None
        self.fov = int(fov) if fov is not None else None
        self.unique_cell_id = int(unique_cell_id) if unique_cell_id is not None else None

        # ---- placeholders for selections made by display routines ----
        self.cell_label = None
        self.chosen_spot = None
        self.chosen_ts = None
        self.chosen_foci = None
        self.regular_nuc_spot = None   # non-TS/foci spot inside nucleus
        self.regular_cyto_spot = None  # non-TS/foci spot inside cytoplasm

        # ---- reproducible sampling RNG ----
        self.rng = np.random.default_rng(rng_seed)

        # ---- flags for HDF5 alignment and storage behavior ----
        self.align_h5_to_manager = bool(align_h5_to_manager)
        self.store_original_h5   = bool(store_original_h5)

        # ---- convenience flags (used by some display paths) ----
        self.has_nb_spots_in_clusters = (self.clusters_df is not None) and ('nb_spots' in self.clusters_df.columns)
        
        # ---- detect NAS columns ----
        self.has_NAS = any([
            (df is not None and 'NAS_location' in df.columns)
            for df in (self.spots, self.cellprops, self.clusters_df)
        ])

        # ---- align DataFrame h5_idx to the manager's file order (fixes mismatch) ----
        if self.align_h5_to_manager:
            self._align_h5_indices_to_manager_inplace(keep_original=self.store_original_h5, verbose=True)

    def _safe_slug(self, s, maxlen=80):
        import os, re
        if s is None:
            return ""
        s = str(s)
        s = s.replace(os.sep, "_").replace("/", "_").replace("\\", "_").replace(":", "-")
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
        if len(s) > maxlen:
            s = s[-maxlen:]
        return s

    def _basename(self, s):
        try:
            return os.path.basename(str(s)).lower()
        except Exception:
            return None

    def _align_h5_indices_to_manager_inplace(self, keep_original=True, verbose=True):
        """
        Remap h5_idx in self.spots / self.clusters_df / self.cellprops so that they
        match the index order of self.h5_file_paths. Uses NAS_location basename as identity.
        """
        mgr_bases = [self._basename(p) for p in self.h5_file_paths]
        base_to_new = {b: i for i, b in enumerate(mgr_bases)}

        def infer_old_to_new(df):
            """
            For a given DF, return a dict old_idx -> new_idx by majority NAS_location basename.
            """
            mapping = {}
            if df is None or df.empty:
                return mapping
            if not {'h5_idx', 'NAS_location'}.issubset(df.columns):
                return mapping

            for old_idx, g in df.groupby('h5_idx', observed=True):
                bases = g['NAS_location'].dropna().map(self._basename)
                if bases.empty:
                    continue
                vals, counts = np.unique(bases.values, return_counts=True)
                picked = vals[np.argmax(counts)]
                if picked in base_to_new:
                    mapping[int(old_idx)] = int(base_to_new[picked])
            return mapping

        # gather votes from all tables
        votes = {}
        for df in (self.spots, self.clusters_df, self.cellprops):
            m = infer_old_to_new(df)
            for old_idx, new_idx in m.items():
                votes.setdefault(int(old_idx), []).append(int(new_idx))

        if not votes:
            if verbose:
                print("[align] No NAS_location info found; leaving h5_idx unchanged.")
            self.h5_idx_map = {}
            return

        # majority vote per old_idx
        old_to_new = {}
        for old_idx, arr in votes.items():
            vals, counts = np.unique(arr, return_counts=True)
            old_to_new[int(old_idx)] = int(vals[np.argmax(counts)])

        if verbose:
            print("[align] Remapping h5_idx to match manager order:")
            for k in sorted(old_to_new):
                new = old_to_new[k]
                print(f"    {k} -> {new}  ({mgr_bases[new]})")

        def apply_map(df):
            if df is None or 'h5_idx' not in getattr(df, 'columns', []):
                return df
            if keep_original and 'h5_idx_orig' not in df.columns:
                df = df.copy()
                df['h5_idx_orig'] = df['h5_idx']
            else:
                df = df.copy()
            df['h5_idx'] = df['h5_idx'].map(lambda x: old_to_new.get(int(x), int(x))).astype(int)
            return df

        # apply in place (reassign attributes)
        self.spots       = apply_map(self.spots)
        self.clusters_df = apply_map(self.clusters_df)
        self.cellprops   = apply_map(self.cellprops)

        # stash mapping for debugging
        self.h5_idx_map = old_to_new

    def _to_int_series(self, s):
        # Coerce a Series to nullable Int64 (tolerates NaN) for reliable equality
        import pandas as pd
        if s is None:
            return s
        return pd.to_numeric(s, errors="coerce").astype("Int64")

    def _mask_by_keys(self, df, h5_idx=None, fov=None, unique_cell_id=None, nas=None):
        # Build a boolean mask with safe integer coercion for key columns
        import numpy as np
        if df is None or df.empty:
            return None
        m = np.ones(len(df), dtype=bool)
        if ('h5_idx' in df.columns) and (h5_idx is not None):
            m &= (self._to_int_series(df['h5_idx']) == int(h5_idx))
        if ('fov' in df.columns) and (fov is not None):
            m &= (self._to_int_series(df['fov']) == int(fov))
        if ('unique_cell_id' in df.columns) and (unique_cell_id is not None):
            m &= (self._to_int_series(df['unique_cell_id']) == int(unique_cell_id))
        if (nas is not None) and ('NAS_location' in df.columns):
            m &= (df['NAS_location'].astype(str) == str(nas))
        return m

    def get_images_and_masks(self, h5_idx=None, fov=None):
        """
        Lazy-load images and masks for a specific HDF5 file and FOV.

        Behavior
        --------
        - If `h5_idx`/`fov` are None, uses `self.h5_idx`/`self.fov`.
        - If still None, samples with self.rng from available options.
        - Returns NumPy arrays for the selected FOV and updates `self.h5_idx/self.fov`.

        Returns
        -------
        images : np.ndarray
            Shape (C, Z, Y, X) after optional squeeze.
        masks  : np.ndarray
            Shape (C, Z, Y, X) after optional squeeze.
        used_h5_idx : int
        used_fov    : int
        """
        # ---- decide which h5/fov to use ----
        used_h5_idx = h5_idx if h5_idx is not None else self.h5_idx
        if used_h5_idx is None:
            if not self.h5_file_paths or len(self.h5_file_paths) == 0:
                raise ValueError("No HDF5 file paths available in `self.h5_file_paths`.")
            used_h5_idx = int(self.rng.integers(0, len(self.h5_file_paths)))

        file_path = self.h5_file_paths[used_h5_idx]

        h5_file = None
        try:
            h5_file = safe_open_h5(file_path)  # your retry-safe wrapper

            # basic dataset presence checks
            if 'raw_images' not in h5_file or 'masks' not in h5_file:
                raise KeyError(f"Expected groups 'raw_images' and 'masks' in {file_path}.")

            num_fov = h5_file['raw_images'].shape[0]

            used_fov = fov if fov is not None else self.fov
            if used_fov is None:
                if num_fov <= 0:
                    raise ValueError(f"No FOVs found in 'raw_images' for file: {file_path}")
                used_fov = int(self.rng.integers(0, num_fov))

            if not (0 <= used_fov < num_fov):
                raise IndexError(f"FOV {used_fov} out of range [0, {num_fov-1}] for file: {file_path}")

            # read exactly one FOV and materialize as NumPy arrays
            images = np.array(h5_file['raw_images'][used_fov])
            masks  = np.array(h5_file['masks'][used_fov])

            # handle shapes like (1, C, Z, Y, X) -> (C, Z, Y, X)
            if images.ndim >= 5 and images.shape[0] == 1:
                images = np.squeeze(images, axis=0)
            if masks.ndim >= 5 and masks.shape[0] == 1:
                masks = np.squeeze(masks, axis=0)

            # minimal sanity check
            if images.ndim != 4 or masks.ndim != 4:
                raise ValueError(f"Expected 4D arrays (C,Z,Y,X). Got images {images.shape}, masks {masks.shape}.")

        finally:
            if h5_file is not None:
                try:
                    h5_file.close()
                except Exception as close_err:
                    print(f"WARNING: Error closing file {file_path}: {close_err}")

        # update current state to what we actually used
        self.h5_idx = used_h5_idx
        self.fov = used_fov

        return images, masks, used_h5_idx, used_fov
    
    def display_gating_overlay(self,
                            h5_idx=None,
                            fov=None,
                            image_channel=None,
                            label_mask_channel=None,
                            show=True,
                            save=False,
                            outdir=None,
                            fname_prefix="gating",
                            dpi_notebook=120,
                            dpi_save=600,
                            with_legend=False,
                            nas_strategy='from_file',
                            nas_location_override=None):
        """
        Display a full-FOV image with segmentation overlays, marking:
        • kept cells  : IDs present in cellprops for this FOV (and NAS if applicable)
        • removed cells: IDs present in the label mask but NOT in cellprops

        NAS handling
        -----------
        The tables are subset to a single NAS_location chosen by:
        nas_strategy in {'from_file','exact','majority','none'} (default 'from_file').
        If 'exact', provide nas_location_override.

        Returns
        -------
        used_h5_idx : int
        used_fov    : int
        nas_location : str | None
        """
        import os

        # ---- resolve channels ----
        image_channel = self.CYTO_CH if image_channel is None else int(image_channel)
        label_mask_channel = self.CYTO_CH if label_mask_channel is None else int(label_mask_channel)

        # ---- load data for this FOV (also updates self.h5_idx/self.fov) ----
        images, masks, used_h5_idx, used_fov = self.get_images_and_masks(h5_idx=h5_idx, fov=fov)

        # ---- build 2D underlay image and labeled mask ----
        img_2d = np.max(images[image_channel], axis=0)        # (Y, X)
        label_2d = np.max(masks[label_mask_channel], axis=0)  # labeled cells

        # contrast rescale
        if img_2d.size > 0:
            p1, p99 = np.percentile(img_2d, (1, 99))
            img_2d = exposure.rescale_intensity(img_2d, in_range=(p1, p99), out_range=(0, 1))

        # ---- choose NAS_location per strategy ----
        nas_location = None
        if self.has_NAS:
            nas_location = self._select_nas_location(used_h5_idx, used_fov,
                                                    strategy=nas_strategy,
                                                    override=nas_location_override)

        # ---- subset metadata to this FOV (and NAS if available) ----
        def _subset(df):
            if df is None:
                return None
            sub = df[(df['h5_idx'] == used_h5_idx) & (df['fov'] == used_fov)]
            if (nas_location is not None) and ('NAS_location' in sub.columns):
                sub = sub[sub['NAS_location'] == nas_location]
            return sub

        spots_frame     = _subset(self.spots)
        cellprops_frame = _subset(self.cellprops)
        _ = _subset(self.clusters_df)  # not used for drawing here, kept for symmetry/logging if needed

        if cellprops_frame is None or cellprops_frame.empty:
            print(f"[INFO] No cellprops for (h5_idx={used_h5_idx}, fov={used_fov}"
                f"{', NAS='+nas_location if nas_location else ''}). "
                "Overlay will show all mask labels as 'removed'.")

        # ---- derive kept vs removed sets ----
        # prefer 'cell_label' (mask ID), else try 'unique_cell_id'
        kept_ids = set()
        if (cellprops_frame is not None) and (not cellprops_frame.empty):
            if 'cell_label' in cellprops_frame.columns:
                kept_ids = set(cellprops_frame['cell_label'].dropna().astype(int).unique().tolist())
            elif 'unique_cell_id' in cellprops_frame.columns:
                kept_ids = set(cellprops_frame['unique_cell_id'].dropna().astype(int).unique().tolist())

        mask_ids = set(np.unique(label_2d).tolist())
        mask_ids.discard(0)  # background
        removed_ids = mask_ids - kept_ids

        # ---- colors ----
        kept_palette = list(mcolors.TABLEAU_COLORS.values())
        removed_color = 'red'
        id_to_color = {cid: kept_palette[i % len(kept_palette)] for i, cid in enumerate(sorted(kept_ids))}

        # ---- plot ----
        fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi_notebook)
        ax.imshow(img_2d, cmap='gray')

        # draw kept contours
        for cid in sorted(kept_ids):
            cmask = (label_2d == cid)
            contours = find_contours(cmask, 0.5)
            if not contours:
                continue
            cnt = max(contours, key=lambda x: x.shape[0])  # largest contour
            ax.plot(cnt[:, 1], cnt[:, 0], linewidth=2, color=id_to_color[cid],
                    label=(f'Cell {cid}' if with_legend else None))

        # draw removed contours (dashed red)
        for cid in sorted(removed_ids):
            cmask = (label_2d == cid)
            contours = find_contours(cmask, 0.5)
            if not contours:
                continue
            cnt = max(contours, key=lambda x: x.shape[0])
            ax.plot(cnt[:, 1], cnt[:, 0], linewidth=2, linestyle='dashed', color=removed_color,
                    label=(f'Removed {cid}' if with_legend else None))

        title = f"h5_idx={used_h5_idx}, FOV={used_fov}"
        if nas_location:
            title += f", NAS={nas_location}"
        ax.set_title(title)
        ax.axis('off')
        if with_legend:
            ax.legend(fontsize=9, frameon=False, loc='upper right')

        # save if requested
        if save:
            import os
            os.makedirs(outdir or "figs", exist_ok=True)
            nas_tag = ""
            if self.has_NAS:
                nas = self._select_nas_location(used_h5_idx, used_fov, strategy='from_file')
                nas_tag = f"_NAS-{self._safe_slug(nas)}" if nas else ""
            fname = f"{fname_prefix}_h{used_h5_idx}_f{used_fov}{nas_tag}"
            fpath = os.path.join(outdir or "figs", f"{fname}.png")
            fig.savefig(fpath, dpi=dpi_save, bbox_inches='tight', pad_inches=0.02)
            print(f"[saved] {fpath}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return used_h5_idx, used_fov, nas_location
    
    def _display_zoom_on_cell(self,
                            unique_cell_id,
                            spotChannel=None,
                            cytoChannel=None,
                            nucChannel=None,
                            show=True,
                            save=False,
                            outdir=None,
                            fname_prefix="cell",
                            dpi_notebook=120,
                            dpi_save=600):
        if self.h5_idx is None or self.fov is None:
            raise ValueError("Set self.h5_idx and self.fov (or call display_gating_overlay/get_images_and_masks first).")

        spotChannel = self.SPOT_CH if spotChannel is None else int(spotChannel)
        cytoChannel = self.CYTO_CH if cytoChannel is None else int(cytoChannel)
        nucChannel  = self.NUC_CH  if nucChannel  is None else int(nucChannel)

        # --- metadata for this cell ---
        cell_id = int(unique_cell_id)
        mask = self._mask_by_keys(self.cellprops, h5_idx=self.h5_idx, fov=self.fov, unique_cell_id=cell_id)
        cdf = self.cellprops.loc[mask] if mask is not None else self.cellprops.iloc[0:0]
        if cdf.empty:
            print(f"[WARN] Cell {cell_id} not found in (h5_idx={self.h5_idx}, fov={self.fov}).")
            # quick one-line aid: show a few available IDs in this FOV (post-coercion)
            m2 = self._mask_by_keys(self.cellprops, h5_idx=self.h5_idx, fov=self.fov)
            if m2 is not None and m2.any():
                try:
                    avail = self._to_int_series(self.cellprops.loc[m2, 'unique_cell_id']).dropna().astype(int).unique().tolist()
                    print(f"       Available unique_cell_id (sample): {sorted(avail)[:12]}{' ...' if len(avail)>12 else ''}")
                except Exception:
                    pass
            self.cell_label = None
            self.chosen_spot = self.chosen_ts = self.chosen_foci = None
            self.regular_nuc_spot = self.regular_cyto_spot = None
            return None, None, None

        self.cell_label = cell_id
        mask_label_id = int(cdf['cell_label'].iloc[0]) if 'cell_label' in cdf.columns else cell_id

        # bbox (clip to image later)
        row_min = int(cdf['cell_bbox-0'].iloc[0])
        col_min = int(cdf['cell_bbox-1'].iloc[0])
        row_max = int(cdf['cell_bbox-2'].iloc[0])
        col_max = int(cdf['cell_bbox-3'].iloc[0])

        # --- load images & masks for this FOV ---
        images, masks, _, _ = self.get_images_and_masks(self.h5_idx, self.fov)

        # 2D intensity (max-projection)
        img2d = np.max(images[spotChannel], axis=0)

        # clip bbox to image bounds
        row_min = max(0, row_min); col_min = max(0, col_min)
        row_max = min(img2d.shape[0], row_max); col_max = min(img2d.shape[1], col_max)

        crop = img2d[row_min:row_max, col_min:col_max]
        if crop.size > 0:
            p1, p99 = np.percentile(crop, (1, 99))
            crop_stretched = exposure.rescale_intensity(crop, in_range=(p1, p99), out_range=(0, 1))
        else:
            crop_stretched = crop

        # --- labeled masks -> boolean maps for overlay only ---
        nuc2d  = np.max(masks[nucChannel],  axis=0)
        cyto2d = np.max(masks[cytoChannel], axis=0)

        # detect whether masks are per-cell labeled; build cell/nucleus binary maps
        cyto_is_labeled = (mask_label_id in np.unique(cyto2d))
        nuc_is_labeled  = (mask_label_id in np.unique(nuc2d))

        cell_bin = (cyto2d == mask_label_id) if cyto_is_labeled else (cyto2d > 0)
        nuc_bin  = (nuc2d  == mask_label_id) if nuc_is_labeled  else ((nuc2d > 0) & cell_bin)
        cyto_only_bin = cell_bin & (~nuc_bin)  # for overlay (visualization)

        # crops of the boolean masks for overlay
        crop_nuc = nuc_bin[row_min:row_max, col_min:col_max]
        crop_cyt = cyto_only_bin[row_min:row_max, col_min:col_max]

        # --- gather detections for this cell ---
        cell_spots = self.spots[(self.spots['h5_idx'] == self.h5_idx) &
                                (self.spots['fov'] == self.fov) &
                                (self.spots['unique_cell_id'] == cell_id)]
        # deterministic scan order (top-to-bottom, left-to-right)
        if {'y_px', 'x_px'}.issubset(cell_spots.columns):
            cell_spots = cell_spots.sort_values(['y_px', 'x_px'], kind='mergesort')

        cell_clusters = self.clusters_df[(self.clusters_df['h5_idx'] == self.h5_idx) &
                                        (self.clusters_df['fov'] == self.fov) &
                                        (self.clusters_df['unique_cell_id'] == cell_id)]

        ts_df   = cell_clusters[cell_clusters['is_nuc'] == 1]
        foci_df = cell_clusters[cell_clusters['is_nuc'] == 0]

        def _pick_largest(df):
            if df is None or df.empty: return None
            return df.sort_values('nb_spots', ascending=False).iloc[0] if 'nb_spots' in df.columns else df.iloc[0]

        self.chosen_ts   = _pick_largest(ts_df)
        self.chosen_foci = _pick_largest(foci_df)

        # cluster IDs to exclude from "regular" spots
        exclude_ids = set()
        if not ts_df.empty and 'cluster_index' in ts_df.columns:
            exclude_ids.update(ts_df['cluster_index'].dropna().unique().tolist())
        if not foci_df.empty and 'cluster_index' in foci_df.columns:
            exclude_ids.update(foci_df['cluster_index'].dropna().unique().tolist())

        regular_spots = (cell_spots[~cell_spots['cluster_index'].isin(exclude_ids)]
                        if 'cluster_index' in cell_spots.columns else cell_spots).copy()

        # --- PICK regular nuc / cyto BY is_nuc FLAG (preferred path) ---
        self.regular_nuc_spot = None
        self.regular_cyto_spot = None
        if 'is_nuc' in regular_spots.columns:
            rs = regular_spots.copy()
            # be robust to floats/bools
            isn = rs['is_nuc'].astype(int)
            nuc_candidates  = rs[isn == 1]
            cyto_candidates = rs[isn == 0]
            if not nuc_candidates.empty:
                self.regular_nuc_spot = nuc_candidates.iloc[0]
            if not cyto_candidates.empty:
                self.regular_cyto_spot = cyto_candidates.iloc[0]
        else:
            # --- Fallback: geometry by masks if is_nuc is missing ---
            def _in_bin(bin2d, srow):
                x = int(round(srow['x_px'])); y = int(round(srow['y_px']))
                return (0 <= x < bin2d.shape[1]) and (0 <= y < bin2d.shape[0]) and bool(bin2d[y, x])
            for _, s in regular_spots.iterrows():
                if (self.regular_nuc_spot is None) and _in_bin(nuc_bin, s):
                    self.regular_nuc_spot = s; continue
                if (self.regular_cyto_spot is None) and _in_bin(cyto_only_bin, s):
                    self.regular_cyto_spot = s
                if (self.regular_nuc_spot is not None) and (self.regular_cyto_spot is not None):
                    break

        # backward compatibility with the spot-zoom wrapper
        self.chosen_spot = self.regular_nuc_spot

        # --- render ---
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=(dpi_notebook if show else dpi_save))
        axs[0].imshow(crop_stretched, cmap='gray')
        axs[0].imshow(crop_nuc.astype(float), cmap='Blues', alpha=0.30)
        axs[0].imshow(crop_cyt.astype(float), cmap='Reds',  alpha=0.30)
        axs[0].set_title(f"Cell {cell_id} — masks")

        axs[1].imshow(crop_stretched, cmap='gray')
        axs[1].imshow(crop_nuc.astype(float), cmap='Blues', alpha=0.20)
        axs[1].imshow(crop_cyt.astype(float), cmap='Reds',  alpha=0.20)
        axs[1].set_title(f"Cell {cell_id} — detections")

        dx, dy = col_min, row_min

        # draw all cell spots (gold)
        for _, s in cell_spots.iterrows():
            sx = s['x_px'] - dx
            sy = s['y_px'] - dy
            draw_spot_circle(axs[1], sx, sy, radius=3, color='gold', linewidth=2.5)

        # TS (magenta) with size annotation if available
        if self.chosen_ts is not None:
            sx = self.chosen_ts['x_px'] - dx; sy = self.chosen_ts['y_px'] - dy
            draw_spot_circle(axs[1], sx, sy, radius=4, color='magenta', linewidth=2.5)
            if 'nb_spots' in self.chosen_ts.index:
                axs[1].text(sx - 12, sy, f"{int(self.chosen_ts['nb_spots'])}", color='magenta', fontsize=15)

        # Foci (cyan)
        if self.chosen_foci is not None:
            sx = self.chosen_foci['x_px'] - dx; sy = self.chosen_foci['y_px'] - dy
            draw_spot_circle(axs[1], sx, sy, radius=4, color='cyan', linewidth=2.5)
            if 'nb_spots' in self.chosen_foci.index:
                axs[1].text(sx - 12, sy, f"{int(self.chosen_foci['nb_spots'])}", color='cyan', fontsize=15)

        # highlight the two "regular" picks for clarity (blue=nuc, darkorange=cyto)
        if self.regular_nuc_spot is not None:
            sx = self.regular_nuc_spot['x_px'] - dx; sy = self.regular_nuc_spot['y_px'] - dy
            draw_spot_circle(axs[1], sx, sy, radius=4, color='blue', linewidth=2.5)
        if self.regular_cyto_spot is not None:
            sx = self.regular_cyto_spot['x_px'] - dx; sy = self.regular_cyto_spot['y_px'] - dy
            draw_spot_circle(axs[1], sx, sy, radius=4, color='darkorange', linewidth=2.5)

        for ax in axs: ax.axis('off')
        plt.tight_layout()

        # save if requested
        if save:
            import os
            os.makedirs(outdir or "figs", exist_ok=True)
            nas_tag = ""
            if self.has_NAS:
                nas = self._select_nas_location(self.h5_idx, self.fov, strategy='from_file')
                nas_tag = f"_NAS-{self._safe_slug(nas)}" if nas else ""
            fname = f"{fname_prefix}_h{self.h5_idx}_f{self.fov}_cell{cell_id}{nas_tag}"
            fpath = os.path.join(outdir or "figs", f"{fname}.png")
            fig = axs[0].get_figure()
            fig.savefig(fpath, dpi=dpi_save, bbox_inches='tight', pad_inches=0.02)
            print(f"[saved] {fpath}")

        if show:
            plt.show()
        else:
            plt.close(axs[0].get_figure())

        return self.chosen_spot, self.chosen_ts, self.chosen_foci

    def _display_spot_crop(self,
                        row,
                        kind,
                        spotChannel=None,
                        pad=15,
                        show=True,
                        save=False,
                        outdir=None,
                        fname_prefix="crop",
                        dpi_notebook=120,
                        dpi_save=600,
                        draw_context=True,
                        show_removed=False,
                        removed_color="red"):
        """
        Crop around a single target (TS, foci, or regular spot) and render a zoom.
        Avoids duplicate labels by skipping re-draw of the same TS/FOCI in context.
        Applies NAS-location subsetting when available to keep overlays consistent.
        """
        import os
        import pandas as pd

        if row is None:
            print("[INFO] _display_spot_crop called with row=None; nothing to draw.")
            return None
        if self.h5_idx is None or self.fov is None:
            raise ValueError("Set self.h5_idx and self.fov before cropping (call display_gating_overlay or get_images_and_masks).")

        # resolve channel
        spotChannel = self.SPOT_CH if spotChannel is None else int(spotChannel)

        # current NAS (only used for subsetting overlays / filename tag)
        nas = None
        if self.has_NAS:
            try:
                nas = self._select_nas_location(self.h5_idx, self.fov, strategy='from_file')
            except Exception:
                nas = None

        # helper: robust key-subset (tolerates dtype drift)
        def _safe_subset(df, h5, fv, uid=None, nas_val=None):
            if df is None or df.empty:
                return df.iloc[0:0]
            m = pd.Series(True, index=df.index)
            if 'h5_idx' in df.columns:
                m &= pd.to_numeric(df['h5_idx'], errors='coerce').astype('Int64') == int(h5)
            if 'fov' in df.columns:
                m &= pd.to_numeric(df['fov'], errors='coerce').astype('Int64') == int(fv)
            if (uid is not None) and ('unique_cell_id' in df.columns):
                m &= pd.to_numeric(df['unique_cell_id'], errors='coerce').astype('Int64') == int(uid)
            if (nas_val is not None) and ('NAS_location' in df.columns):
                m &= (df['NAS_location'].astype(str) == str(nas_val))
            return df.loc[m]

        # helper: determine if two rows refer to the same object (for dedup)
        def _same_object(a, b):
            try:
                if ('cluster_index' in a.index and 'cluster_index' in b.index and
                    pd.notna(a['cluster_index']) and pd.notna(b['cluster_index'])):
                    return int(a['cluster_index']) == int(b['cluster_index'])
            except Exception:
                pass
            try:
                return (int(round(a['x_px'])) == int(round(b['x_px'])) and
                        int(round(a['y_px'])) == int(round(b['y_px'])))
            except Exception:
                return False

        # --- center & bounds ---
        try:
            sx = int(round(row['x_px'])); sy = int(round(row['y_px']))
        except KeyError as e:
            raise KeyError(f"row is missing coordinate column: {e}")

        images, masks, _, _ = self.get_images_and_masks(self.h5_idx, self.fov)
        img2d = np.max(images[spotChannel], axis=0)

        H, W = img2d.shape
        x1 = max(sx - pad, 0); x2 = min(sx + pad, W)
        y1 = max(sy - pad, 0); y2 = min(sy + pad, H)

        sub = img2d[y1:y2, x1:x2]
        if sub.size > 0:
            p1, p99 = np.percentile(sub, (1, 99))
            sub = exposure.rescale_intensity(sub, in_range=(p1, p99), out_range=(0, 1))

        # --- figure ---
        fig, ax = plt.subplots(figsize=(5, 5), dpi=(dpi_notebook if show else dpi_save))
        ax.imshow(sub, cmap='gray')
        ax.set_title(f"{kind} (cell {self.cell_label if self.cell_label is not None else 'NA'})")

        # color map (CYTO_REG = 'darkorange' for improved contrast vs gold)
        color_map = {'TS': 'magenta', 'FOCI': 'cyan', 'NUC_REG': 'blue', 'CYTO_REG': 'darkorange', 'SPOT': 'blue'}
        tcolor = color_map.get(kind, 'white')

        # draw target marker at its actual offset in the crop
        cx, cy = sx - x1, sy - y1
        draw_spot_circle(ax, cx, cy, radius=3, color=tcolor, linewidth=2.5)

        # a single annotation for TS/FOCI near the target (no duplicate)
        if kind in ('TS', 'FOCI') and isinstance(row, pd.Series) and ('nb_spots' in row.index) and pd.notna(row['nb_spots']):
            try:
                ax.text(cx - 7, cy, f"{int(row['nb_spots'])}", color=tcolor, fontsize=15)
            except Exception:
                pass

        # optional context overlays, NAS-aware and key-safe
        if draw_context and (self.cell_label is not None):
            cell_spots = _safe_subset(self.spots, self.h5_idx, self.fov, uid=self.cell_label, nas_val=nas)
            for _, s in cell_spots.iterrows():
                rx = int(round(s['x_px'])) - x1
                ry = int(round(s['y_px'])) - y1
                if 0 <= rx < (x2 - x1) and 0 <= ry < (y2 - y1):
                    if (int(round(s['x_px'])) == sx) and (int(round(s['y_px'])) == sy):
                        continue  # skip the target itself
                    draw_spot_circle(ax, rx, ry, radius=3, color='gold', linewidth=2)

            # chosen TS within patch? (skip when the target IS that TS)
            if self.chosen_ts is not None:
                tx = int(round(self.chosen_ts['x_px'])); ty = int(round(self.chosen_ts['y_px']))
                rx, ry = tx - x1, ty - y1
                if 0 <= rx < (x2 - x1) and 0 <= ry < (y2 - y1):
                    if not (kind == 'TS' and _same_object(self.chosen_ts, row)):
                        draw_spot_circle(ax, rx, ry, radius=4, color='magenta', linewidth=2.5)
                        if 'nb_spots' in self.chosen_ts.index and pd.notna(self.chosen_ts['nb_spots']):
                            ax.text(rx - 7, ry, f"{int(self.chosen_ts['nb_spots'])}", color='magenta', fontsize=15)

            # chosen foci within patch? (skip when the target IS that FOCI)
            if self.chosen_foci is not None:
                tx = int(round(self.chosen_foci['x_px'])); ty = int(round(self.chosen_foci['y_px']))
                rx, ry = tx - x1, ty - y1
                if 0 <= rx < (x2 - x1) and 0 <= ry < (y2 - y1):
                    if not (kind == 'FOCI' and _same_object(self.chosen_foci, row)):
                        draw_spot_circle(ax, rx, ry, radius=4, color='cyan', linewidth=2.5)
                        if 'nb_spots' in self.chosen_foci.index and pd.notna(self.chosen_foci['nb_spots']):
                            ax.text(rx - 7, ry, f"{int(self.chosen_foci['nb_spots'])}", color='cyan', fontsize=15)

        # optional: overlay removed spots (NAS-aware)
        if show_removed and (self.removed_spots is not None) and (self.cell_label is not None):
            needed_cols = {'h5_idx', 'fov', 'unique_cell_id', 'x_px', 'y_px'}
            if needed_cols.issubset(set(self.removed_spots.columns)):
                rem = _safe_subset(self.removed_spots, self.h5_idx, self.fov, uid=self.cell_label, nas_val=nas)
                for _, s in rem.iterrows():
                    rx = int(round(s['x_px'])) - x1
                    ry = int(round(s['y_px'])) - y1
                    if 0 <= rx < (x2 - x1) and 0 <= ry < (y2 - y1):
                        draw_spot_circle(ax, rx, ry, radius=3, color=removed_color, linewidth=2, alpha=0.5)

        ax.axis('off')
        plt.tight_layout()

        # save if requested
        if save:
            os.makedirs(outdir or "figs", exist_ok=True)
            nas_tag = f"_NAS-{self._safe_slug(nas)}" if nas else ""
            cell_id_str = self.cell_label if self.cell_label is not None else getattr(row, 'unique_cell_id', 'NA')
            base = f"{fname_prefix}_h{self.h5_idx}_f{self.fov}_cell{cell_id_str}_{kind.lower()}_x{sx}_y{sy}{nas_tag}"
            fpath = os.path.join(outdir or "figs", f"{base}.png")
            fig.savefig(fpath, dpi=dpi_save, bbox_inches='tight', pad_inches=0.02)
            print(f"[saved] {fpath}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return (x1, y1, x2, y2)

    def _display_zoom_on_one_spot(self,
                                spotChannel=None,
                                row=None,
                                kind='SPOT',
                                pad=15,
                                show=True,
                                save=False,
                                outdir=None,
                                fname_prefix="crop",
                                dpi_notebook=120,
                                dpi_save=600,
                                draw_context=True,
                                show_removed=False,
                                removed_color="red"):
        """
        Backward-compatible wrapper around `_display_spot_crop`.

        - If `row` is None, uses `self.chosen_spot` (set by `_display_zoom_on_cell`).
        - `kind`: 'TS' | 'FOCI' | 'NUC_REG' | 'CYTO_REG' | 'SPOT'
        - `show_removed=True` overlays removed spots (if available) for this cell.

        Returns the crop bounds (x1, y1, x2, y2).
        """
        if row is None:
            row = self.chosen_spot
        return self._display_spot_crop(row=row,
                                    kind=kind,
                                    spotChannel=spotChannel,
                                    pad=pad,
                                    show=show,
                                    save=save,
                                    outdir=outdir,
                                    fname_prefix=fname_prefix,
                                    dpi_notebook=dpi_notebook,
                                    dpi_save=dpi_save,
                                    draw_context=draw_context,
                                    show_removed=show_removed,
                                    removed_color=removed_color)

    def _select_nas_location(self, h5_idx, fov, strategy='from_file', override=None):
        """
        Decide which NAS_location to use for (h5_idx, fov).

        strategy:
        - 'from_file' : infer from the actual H5 file path used; fallback to 'majority'
        - 'exact'     : use `override` exactly (if present for this pair), else None
        - 'majority'  : pick the most frequent NAS_location across tables
        - 'none'      : don't choose (return None)
        """
        # collect candidates for this (h5,fov)
        candidates = []
        for df in (self.spots, self.cellprops, self.clusters_df):
            if df is None or 'NAS_location' not in getattr(df, 'columns', []):
                continue
            vals = df[(df['h5_idx'] == h5_idx) & (df['fov'] == fov)]['NAS_location'].dropna().astype(str).values
            if len(vals) > 0:
                candidates.extend(vals.tolist())

        if len(candidates) == 0:
            return None

        uniq = np.unique(candidates)

        if strategy == 'none':
            return None

        if strategy == 'exact':
            return override if (override is not None and override in set(uniq)) else None

        if strategy == 'majority':
            vals, counts = np.unique(candidates, return_counts=True)
            return vals[np.argmax(counts)]

        if strategy == 'from_file':
            # try to match the actual file path string with NAS strings
            try:
                fp = str(self.h5_file_paths[h5_idx]).lower()
                best = None
                best_len = -1
                for nas in uniq:
                    s = str(nas).lower()
                    if s and s in fp and len(s) > best_len:
                        best = nas
                        best_len = len(s)
                if best is not None:
                    return best
            except Exception:
                pass
            # fallback: majority
            vals, counts = np.unique(candidates, return_counts=True)
            return vals[np.argmax(counts)]

        # default fallback
        vals, counts = np.unique(candidates, return_counts=True)
        return vals[np.argmax(counts)]

    def classify_cells_in_fov(self, h5_idx, fov, return_nas=False,
                            nas_strategy='from_file', nas_location_override=None):
        """
        Partition cells in (h5_idx, fov) into:
            'TS+FOCI', 'TS_only', 'FOCI_only', 'Neither'

        NAS handling matches display_gating_overlay via nas_strategy/override.
        Uses robust key subsetting to tolerate dtype drift (e.g., after ID prefixing).
        """
        import pandas as pd
        import numpy as np

        nas = self._select_nas_location(h5_idx, fov,
                                        strategy=nas_strategy,
                                        override=nas_location_override)

        # robust subsetting
        def _safe_subset(df):
            if df is None:
                return None
            if df.empty:
                return df.iloc[0:0]
            m = pd.Series(True, index=df.index)
            if 'h5_idx' in df.columns:
                m &= pd.to_numeric(df['h5_idx'], errors='coerce').astype('Int64') == int(h5_idx)
            if 'fov' in df.columns:
                m &= pd.to_numeric(df['fov'], errors='coerce').astype('Int64') == int(fov)
            if (nas is not None) and ('NAS_location' in df.columns):
                m &= (df['NAS_location'].astype(str) == str(nas))
            return df.loc[m]

        cp = _safe_subset(self.cellprops)
        cl = _safe_subset(self.clusters_df)

        # valid cells from cellprops
        valid = set()
        if cp is not None and not cp.empty:
            if 'unique_cell_id' not in cp.columns:
                raise KeyError("cellprops must contain 'unique_cell_id'.")
            uids = pd.to_numeric(cp['unique_cell_id'], errors='coerce').astype('Int64')
            valid = set(uids.dropna().astype(int).tolist())
            valid.discard(0)

        # cells with TS and foci from clusters
        ts_cells = set()
        foci_cells = set()
        if cl is not None and not cl.empty:
            if 'unique_cell_id' not in cl.columns or 'is_nuc' not in cl.columns:
                raise KeyError("clusters_df must contain 'unique_cell_id' and 'is_nuc'.")
            uids_cl = pd.to_numeric(cl['unique_cell_id'], errors='coerce').astype('Int64')
            is_nuc_series = pd.to_numeric(cl['is_nuc'], errors='coerce').astype('Int64')

            ts_cells = set(uids_cl[is_nuc_series == 1].dropna().astype(int).unique().tolist())
            foci_cells = set(uids_cl[is_nuc_series == 0].dropna().astype(int).unique().tolist())

        # restrict to valid cells
        ts_cells &= valid
        foci_cells &= valid

        cats = {
            'TS+FOCI':   sorted(ts_cells & foci_cells),
            'TS_only':   sorted(ts_cells - foci_cells),
            'FOCI_only': sorted(foci_cells - ts_cells),
            'Neither':   sorted(valid - (ts_cells | foci_cells)),
        }
        return (cats, nas) if return_nas else cats   

    def pick_fovs(self, h5_indices=None, fov_list=None, per_h5=1, seed=None):
        """
        Decide which FOV(s) to display per h5_idx.

        Parameters
        ----------
        h5_indices : list[int] | None
            If None, inferred from cellprops['h5_idx'] (else spots/clusters as fallback).
        fov_list : dict[int, list[int]] | list[int] | None
            - dict: {h5_idx: [fov, ...]} to explicitly choose per file.
            - list: only valid when exactly one h5_idx is specified; applies that list to it.
            - None: sample up to `per_h5` FOVs observed in metadata for each h5.
        per_h5 : int
            How many FOVs to sample per h5 when fov_list is None.
        seed : int | None
            Optional override for sampling seed (defaults to self.rng).

        Returns
        -------
        pairs : list[tuple[int, int]]
            List of (h5_idx, fov) tuples.
        """
        rng = np.random.default_rng(seed) if seed is not None else self.rng

        # derive available h5 indices
        if h5_indices is None:
            sources = []
            for df in (self.cellprops, self.spots, self.clusters_df):
                if df is not None and 'h5_idx' in df.columns:
                    sources.append(np.unique(df['h5_idx'].dropna().astype(int).values))
            if len(sources) == 0:
                raise ValueError("Could not infer h5_indices from any table.")
            h5_indices = sorted(np.unique(np.concatenate(sources)).tolist())
        else:
            h5_indices = [int(h) for h in h5_indices]

        pairs = []
        for h5 in h5_indices:
            # explicit mapping provided
            if isinstance(fov_list, dict) and (h5 in fov_list):
                for f in fov_list[h5]:
                    pairs.append((int(h5), int(f)))
                continue

            # single-list form (only valid if one h5_idx given)
            if isinstance(fov_list, list):
                if len(h5_indices) != 1:
                    raise ValueError("fov_list as a list is only allowed when exactly one h5_idx is provided.")
                for f in fov_list:
                    pairs.append((int(h5), int(f)))
                continue

            # default: sample from available FOVs observed in metadata for this h5
            fovs_available = []
            for df in (self.cellprops, self.spots, self.clusters_df):
                if df is not None and {'h5_idx','fov'}.issubset(df.columns):
                    fovs_available.append(np.unique(df.loc[df['h5_idx'] == h5, 'fov'].dropna().astype(int).values))
            if len(fovs_available) == 0:
                continue
            fovs_available = sorted(np.unique(np.concatenate(fovs_available)).tolist())

            k = min(int(per_h5), len(fovs_available))
            if k == 0:
                continue
            chosen = rng.choice(fovs_available, size=k, replace=False)
            for f in chosen:
                pairs.append((int(h5), int(f)))

        return pairs

    def default_display(self,
                        h5_indices=None,
                        fov_list=None,
                        per_h5=1,
                        seed=None,
                        # per-category selection
                        max_cells_per_category=1,
                        categories_order=('TS+FOCI', 'TS_only', 'FOCI_only', 'Neither'),
                        # rendering / saving
                        show=True,
                        save=False,
                        outdir=None,
                        dpi_notebook=120,
                        dpi_save=600,
                        gating_fname_prefix="gating",
                        cell_fname_prefix="cell",
                        crop_fname_prefix="crop",
                        crop_pad=15,
                        draw_context=True,
                        show_removed=False,
                        removed_color="red",
                        # NAS handling
                        nas_strategy='from_file',
                        nas_location_override=None,
                        # channels (override if needed)
                        spotChannel=None,
                        cytoChannel=None,
                        nucChannel=None,
                        with_legend=False):
        """
        Default end-to-end display for smiFISH mRNA detection figures.

        For each selected (h5_idx, fov):
        1) Display gating overlay (kept vs removed cells).
        2) Classify cells into categories and pick up to `max_cells_per_category` per category.
        3) For each picked cell:
                - Show cell crop with detections.
                - Emit spot crops according to category:
                    TS+FOCI  -> TS, FOCI, NUC_REG, CYTO_REG
                    TS_only  -> TS, NUC_REG, CYTO_REG
                    FOCI_only-> FOCI, NUC_REG, CYTO_REG
                    Neither  -> NUC_REG, CYTO_REG

        Saving
        ------
        If `save=True`, all figures are saved **directly into** `outdir` (or 'figs' if None)
        with filenames that embed the h5/fov/cell identifiers. No per-FOV subfolders.
        Notebook display uses `dpi_notebook`; saved figures use `dpi_save` for publication.
        """
        import os
        rng = np.random.default_rng(seed) if seed is not None else self.rng

        # flat output directory
        out_root = outdir or "figs"
        if save:
            os.makedirs(out_root, exist_ok=True)

        # choose FOVs to process
        pairs = self.pick_fovs(h5_indices=h5_indices, fov_list=fov_list, per_h5=per_h5, seed=seed)

        # resolve channels
        spotChannel = self.SPOT_CH if spotChannel is None else int(spotChannel)
        cytoChannel = self.CYTO_CH if cytoChannel is None else int(cytoChannel)
        nucChannel  = self.NUC_CH  if nucChannel  is None else int(nucChannel)

        for (h5, fv) in pairs:
            save_dir = out_root  # flat saving into the provided outdir

            # 1) gating overlay (also sets self.h5_idx/self.fov)
            used_h, used_f, nas = self.display_gating_overlay(
                h5_idx=h5, fov=fv,
                image_channel=cytoChannel,
                label_mask_channel=cytoChannel,
                show=show,
                save=save,
                outdir=save_dir,
                fname_prefix=gating_fname_prefix,
                dpi_notebook=dpi_notebook,
                dpi_save=dpi_save,
                with_legend=with_legend,
                nas_strategy=nas_strategy,
                nas_location_override=nas_location_override
            )

            # 2) classify cells for this FOV (with the same NAS policy)
            cats, _ = self.classify_cells_in_fov(
                used_h, used_f, return_nas=True,
                nas_strategy=nas_strategy,
                nas_location_override=nas_location_override
            )

            # ensure consistent category loop order
            for cat in categories_order:
                cell_list = cats.get(cat, [])
                if not cell_list:
                    continue

                # pick up to N cells from this category (reproducibly)
                k = min(max_cells_per_category, len(cell_list))
                chosen_cells = rng.choice(np.array(cell_list, dtype=int), size=k, replace=False)

                for cell_id in np.atleast_1d(chosen_cells):
                    # 3) cell display: sets chosen_ts/foci and regular nuc/cyto picks
                    self.h5_idx = used_h
                    self.fov = used_f
                    self._display_zoom_on_cell(
                        unique_cell_id=int(cell_id),
                        spotChannel=spotChannel,
                        cytoChannel=cytoChannel,
                        nucChannel=nucChannel,
                        show=show,
                        save=save,
                        outdir=save_dir,
                        fname_prefix=f"{cell_fname_prefix}_{cat.replace('+','plus')}"
                                    .replace('/', '-').replace(' ', '')
                                    .lower(),
                        dpi_notebook=dpi_notebook,
                        dpi_save=dpi_save
                    )

                    # emit spot crops as required by category
                    def _maybe_crop(row, kind):
                        if row is not None:
                            self._display_zoom_on_one_spot(
                                spotChannel=spotChannel,
                                row=row,
                                kind=kind,
                                pad=crop_pad,
                                show=show,
                                save=save,
                                outdir=save_dir,
                                fname_prefix=f"{crop_fname_prefix}_{cat.replace('+','plus')}"
                                            .replace('/', '-').replace(' ', '')
                                            .lower(),
                                dpi_notebook=dpi_notebook,
                                dpi_save=dpi_save,
                                draw_context=draw_context,
                                show_removed=show_removed,
                                removed_color=removed_color
                            )

                    if cat == 'TS+FOCI':
                        _maybe_crop(self.chosen_ts, 'TS')
                        _maybe_crop(self.chosen_foci, 'FOCI')
                        _maybe_crop(self.regular_nuc_spot, 'NUC_REG')
                        _maybe_crop(self.regular_cyto_spot, 'CYTO_REG')

                    elif cat == 'TS_only':
                        _maybe_crop(self.chosen_ts, 'TS')
                        _maybe_crop(self.regular_nuc_spot, 'NUC_REG')
                        _maybe_crop(self.regular_cyto_spot, 'CYTO_REG')

                    elif cat == 'FOCI_only':
                        _maybe_crop(self.chosen_foci, 'FOCI')
                        _maybe_crop(self.regular_nuc_spot, 'NUC_REG')
                        _maybe_crop(self.regular_cyto_spot, 'CYTO_REG')

                    elif cat == 'Neither':
                        _maybe_crop(self.regular_nuc_spot, 'NUC_REG')
                        _maybe_crop(self.regular_cyto_spot, 'CYTO_REG')

                    else:
                        # if a custom category slips in, default to the two regular crops
                        _maybe_crop(self.regular_nuc_spot, 'NUC_REG')
                        _maybe_crop(self.regular_cyto_spot, 'CYTO_REG')


class PostProcessingPlotter:
    """
    Plotting utilities for DUSP1 post-processing across multiple days.

    Methods:
      - plot_time_sweep(dex_conc, save_dir=None, display=True)
      - plot_conc_sweep(timepoint, save_dir=None, display=True)
      - plot_time_conc_sweep(conc_list, save_dir=None, display=True)
    """
    def __init__(self, clusters_df, cellprops_df, ssit_df, is_tpl: bool):
        # rename to lowercase for consistency
        self.clusters = clusters_df.rename(columns=str.lower).copy()
        self.cellprops = cellprops_df.rename(columns=str.lower).copy()
        self.ssit = ssit_df.rename(columns=str.lower).copy()
        self.is_tpl = is_tpl
        # metrics to plot
        self.metrics = ['num_nuc_spots', 'num_cyto_spots', 'num_spots']
        # set global style
        sns.set_theme(style='whitegrid', context='paper')
        plt.rcParams['font.family'] = 'Times New Roman'

    def plot_time_sweep(self, dex_conc, save_dir=None, display=True):
        """
        1) TS bar at dex=0 (0 min control) + dex_conc across times
        2) Ridge plots of metrics vs time (including 0 min control)
        3) Line plot of mean±SD nuc/cyto spot counts vs time
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # 1) TS bar
        clusters_nuc = self.clusters[self.clusters['is_nuc'] == 1]
        ts_count = (
            clusters_nuc
            .groupby(['replica','h5_idx','time','fov','cell_label'])
            .size().reset_index(name='ts_count')
        )
        cells = (
            self.cellprops[['replica','h5_idx','time','fov','cell_label','dex_conc']]
            .drop_duplicates()
        )
        subset_cells = cells[
            (cells['dex_conc'] == dex_conc) |
            ((cells['dex_conc'] == 0) & (cells['time'] == 0))
        ]
        merged = pd.merge(subset_cells, ts_count,
                          on=['replica','h5_idx','time','fov','cell_label'],
                          how='left').fillna({'ts_count':0})
        tmp = merged.copy()
        tmp['ts_cat'] = tmp['ts_count'].apply(lambda x: '>=4' if x>=4 else str(int(x)))
        grp = (
            tmp[tmp['ts_cat']!='0']
            .groupby(['time','ts_cat'])
            .size().reset_index(name='count')
        )
        tot = tmp.groupby('time').size().reset_index(name='total')
        grp = pd.merge(grp, tot, on='time')
        grp['fraction'] = grp['count'] / grp['total']
        pivot = (
            grp
            .pivot(index='time', columns='ts_cat', values='fraction')
            .reindex(columns=['1','2','3','>=4'], fill_value=0)
        )

        fig, ax = plt.subplots(figsize=(8,4))
        pal = sns.color_palette("tab10", n_colors=4)
        pivot.plot(kind='bar', ax=ax, color=pal, edgecolor='k')
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Fraction of Cells")
        ax.set_title(f"TS categories at {dex_conc} nM Dex")
        ax.legend(title="TS count", bbox_to_anchor=(1.02,1), loc='upper left')
        plt.tight_layout()
        if save_dir:
            fig.savefig(os.path.join(save_dir, f"ts_time_{dex_conc}.png"), dpi=300)
        if display:
            plt.show()
        plt.close(fig)

        # 2) Ridge plots (include 0 min control)
        control = self.ssit[
            (self.ssit['dex_conc']==0) & (self.ssit['time']==0)
        ]
        df_conc = self.ssit[self.ssit['dex_conc']==dex_conc]
        df = pd.concat([control, df_conc], ignore_index=True)
        times = sorted(df['time'].unique())
        n = len(times)

        for m in self.metrics:
            xs = np.linspace(df[m].min(), df[m].max(), 200)
            # thresholds from control
            ref = control[m].dropna()
            if len(ref) >= 2:
                cdf  = np.arange(1,len(ref)+1)/len(ref)
                thr0 = np.interp(0.5,  cdf, np.sort(ref))
                thr1 = np.interp(0.95, cdf, np.sort(ref))
            else:
                thr0 = thr1 = None

            fig, ax = plt.subplots(figsize=(8, n*1.2))
            fig.suptitle(f"Time sweep — {dex_conc} nM Dex: {m}")
            cmap = sns.color_palette('rocket_r', n_colors=n)[::-1]

            for i, t in enumerate(times):
                data = df[df['time']==t][m].dropna()
                if len(data) >= 2:
                    kde = gaussian_kde(data)
                    y = kde(xs); y = y/y.max()*0.8
                else:
                    y = np.zeros_like(xs)
                y_off = n-1-i
                ax.fill_between(xs, y_off, y_off+y, color=cmap[i], alpha=0.7)

            if thr0 is not None:
                ax.axvline(thr0, linestyle='--', color='red')
                ax.axvline(thr1, linestyle='-',  color='red')

            ax.set_yticks([n-1-i for i in range(n)])
            ax.set_yticklabels([f"{t} min" for t in times])
            ax.set_xlabel("mRNA Count")
            plt.tight_layout(rect=[0,0,1,0.95])
            if save_dir:
                fig.savefig(os.path.join(save_dir, f"ridge_time_{dex_conc}_{m}.png"), dpi=300)
            if display:
                plt.show()
            plt.close(fig)

        # 3) Line plot vs time (with control)
        stats = (
            df.groupby('time')
              .agg(
                mean_n=('num_nuc_spots','mean'),
                sd_n  =('num_nuc_spots','std'),
                mean_c=('num_cyto_spots','mean'),
                sd_c  =('num_cyto_spots','std')
              ).reset_index()
        )
        fig, ax = plt.subplots(figsize=(8,4))
        ax.errorbar(stats['time'], stats['mean_n'], yerr=stats['sd_n'],
                    fmt='-o', color='blue', label='Nuc')
        ax.errorbar(stats['time'], stats['mean_c'], yerr=stats['sd_c'],
                    fmt='-o', color='darkorange', label='Cyto')

        # annotate TPL addition times
        if self.is_tpl and 'time_tpl' in self.ssit.columns:
            # unique TPL addition times for this dex_conc
            tpls = self.ssit[
                (self.ssit['dex_conc']==dex_conc) &
                self.ssit['time_tpl'].notnull()
            ]['time_tpl'].unique()
            for idx, t in enumerate(sorted(tpls)):
                ax.axvline(t, linestyle='--', color='red', alpha=0.5,
                           label='TPL addition' if idx==0 else None)

        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Mean ± SD")
        ax.set_title(f"Time sweep — {dex_conc} nM Dex")
        ax.legend()
        plt.tight_layout()
        if save_dir:
            fig.savefig(os.path.join(save_dir, f"line_time_{dex_conc}.png"), dpi=300)
        if display:
            plt.show()
        plt.close(fig)

    def plot_conc_sweep(self, timepoint, save_dir=None, display=True):
        """
        1) TS bar vs dex_conc at fixed timepoint
        2) Ridge plots of metrics vs dex_conc (including control at dex=0)
        3) Line plot of mean±SD nuc/cyto spot counts vs dex_conc
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # 1) TS bar
        clusters_nuc = self.clusters[self.clusters['is_nuc']==1]
        ts_count = (
            clusters_nuc
            .groupby(['replica','h5_idx','time','fov','cell_label'])
            .size().reset_index(name='ts_count')
        )
        cells = (
            self.cellprops[['replica','h5_idx','time','fov','cell_label','dex_conc']]
            .drop_duplicates()
        )
        subset_cells = cells[cells['time']==timepoint]
        merged = pd.merge(subset_cells, ts_count,
                          on=['replica','h5_idx','time','fov','cell_label'],
                          how='left').fillna({'ts_count':0})
        tmp = merged.copy()
        tmp['ts_cat'] = tmp['ts_count'].apply(lambda x: '>=4' if x>=4 else str(int(x)))
        grp = (
            tmp[tmp['ts_cat']!='0']
            .groupby(['dex_conc','ts_cat'])
            .size().reset_index(name='count')
        )
        tot = tmp.groupby('dex_conc').size().reset_index(name='total')
        grp = pd.merge(grp, tot, on='dex_conc')
        grp['fraction'] = grp['count']/grp['total']
        concs = sorted(tmp['dex_conc'].unique())
        pivot = (
            grp
            .pivot(index='dex_conc', columns='ts_cat', values='fraction')
            .reindex(index=concs, columns=['1','2','3','>=4'], fill_value=0)
        )

        fig, ax = plt.subplots(figsize=(8,4))
        pal = sns.color_palette("tab10", n_colors=4)
        pivot.plot(kind='bar', ax=ax, color=pal, edgecolor='k')
        ax.set_xlabel("Dex_Conc (nM)")
        ax.set_ylabel("Fraction of Cells")
        ax.set_title(f"TS categories at t={timepoint} min")
        plt.tight_layout()
        if save_dir:
            fig.savefig(os.path.join(save_dir, f"ts_conc_{timepoint}.png"), dpi=300)
        if display:
            plt.show()
        plt.close(fig)

        # 2) Ridge vs dex_conc (include control at dex=0)
        control = self.ssit[
            (self.ssit['dex_conc']==0) & (self.ssit['time']==timepoint)
        ]
        df_other = self.ssit[self.ssit['time']==timepoint]
        df = pd.concat([control, df_other], ignore_index=True)
        concs = sorted(df['dex_conc'].unique())
        n = len(concs)

        for m in self.metrics:
            xs = np.linspace(df[m].min(), df[m].max(), 200)
            ref = control[m].dropna()
            if len(ref) >= 2:
                cdf  = np.arange(1,len(ref)+1)/len(ref)
                thr0 = np.interp(0.5,  cdf, np.sort(ref))
                thr1 = np.interp(0.95, cdf, np.sort(ref))
            else:
                thr0 = thr1 = None

            fig, ax = plt.subplots(figsize=(8, n*1.2))
            fig.suptitle(f"Conc sweep at t={timepoint} min: {m}")
            cmap = sns.color_palette('rocket_r', n_colors=n)[::-1]

            for i, c in enumerate(concs):
                data = df[df['dex_conc']==c][m].dropna()
                if len(data) >= 2:
                    kde = gaussian_kde(data)
                    y = kde(xs); y = y/y.max()*0.8
                else:
                    y = np.zeros_like(xs)
                y_off = n-1-i
                ax.fill_between(xs, y_off, y_off+y, color=cmap[i], alpha=0.7)

            if thr0 is not None:
                ax.axvline(thr0, linestyle='--', color='red')
                ax.axvline(thr1, linestyle='-',  color='red')

            ax.set_yticks([n-1-i for i in range(n)])
            ax.set_yticklabels([f"{c} nM" for c in concs])
            ax.set_xlabel("mRNA Count")
            plt.tight_layout(rect=[0,0,1,0.95])
            if save_dir:
                fig.savefig(os.path.join(save_dir, f"ridge_conc_{timepoint}_{m}.png"), dpi=300)
            if display:
                plt.show()
            plt.close(fig)

        # 3) Line vs dex_conc
        stats = (
            df.groupby('dex_conc')
              .agg(mean_n=('num_nuc_spots','mean'),
                   sd_n  =('num_nuc_spots','std'),
                   mean_c=('num_cyto_spots','mean'),
                   sd_c  =('num_cyto_spots','std'))
              .reset_index()
        )
        fig, ax = plt.subplots(figsize=(8,4))
        ax.errorbar(stats['dex_conc'], stats['mean_n'], yerr=stats['sd_n'],
                    fmt='-o', color='blue', label='Nuc')
        ax.errorbar(stats['dex_conc'], stats['mean_c'], yerr=stats['sd_c'],
                    fmt='-o', color='darkorange', label='Cyto')
        ax.set_xscale('symlog', linthresh=1e-3)
        ax.set_xticks([0, 1e-2,1e-1,1,10,100,1000,10000])
        ax.set_xticklabels(['0','0.01','0.1','1','10','100','1000','10000'])
        ax.set_xlabel("Dex_Conc (nM)")
        ax.set_ylabel("Mean ± SD")
        ax.set_title(f"Conc sweep at t={timepoint} min")
        ax.legend()
        plt.tight_layout()
        if save_dir:
            fig.savefig(os.path.join(save_dir, f"line_conc_{timepoint}.png"), dpi=300)
        if display:
            plt.show()
        plt.close(fig)

    def plot_time_conc_sweep(self, conc_list, save_dir=None, display=True):
        """
        Run a time sweep (as above) for each concentration in conc_list.
        """
        for dex_conc in conc_list:
            print(f"\n=== Time sweep for {dex_conc} nM Dex ===")
            self.plot_time_sweep(dex_conc, save_dir=save_dir, display=display)


class ExperimentPlotter:
    """
    Plot TS‐bar, ridge (joy), and line panels for:
      • Time sweep: multiple times at one concentration
      • Conc sweep: multiple concentrations at one time
      • Both‐varying: multiple times AND concentrations

    Inputs: DataFrame with lowercase columns including:
      ['replica','dex_conc','time',
       'num_ts','num_nuc_spots','num_cyto_spots','num_spots']
    """
    def __init__(self, ssit_df: pd.DataFrame):
        self.df = ssit_df.copy()
        self.df.columns = self.df.columns.str.lower()

    def plot_experiment(self,
                        replicas:   list[str],
                        times:      list[float],
                        concs:      list[float],
                        save_dir:   str    = None,
                        display:    bool   = True):
        # — prepare —
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", context="paper")
        plt.rcParams['font.family'] = 'Times New Roman'

        df = self.df[self.df['replica'].isin(replicas)]
        all_times = sorted(set(times)|{0})
        all_concs = sorted(set(concs)|{0})

        time_sweep = len(times)>1 and len(concs)==1
        conc_sweep = len(concs)>1 and len(times)==1
        both_vary  = len(times)>1 and len(concs)>1
        if not (time_sweep or conc_sweep or both_vary):
            raise ValueError(
                "Must supply either a time sweep (multi‐time, single‐conc), "
                "conc sweep (multi‐conc, single‐time), or both‐varying."
            )

        # precompute pooled‐control CDF thresholds
        thr = {}
        for m in ['num_nuc_spots','num_cyto_spots','num_spots']:
            ctrl = df[(df.dex_conc==0)&(df.time==0)][m].dropna().values
            if len(ctrl)>1:
                s = np.sort(ctrl); cdf = np.arange(1,len(s)+1)/len(s)
                thr[m] = (np.interp(0.5,cdf,s), np.interp(0.95,cdf,s))
            else:
                thr[m] = (0,0)

        if time_sweep:
            dex    = concs[0]
            suffix = f"{dex}nM_time-sweep"
            self._plot_time_sweep(df, replicas, all_times, dex, thr,
                                  save_dir, display,
                                  title_suffix=suffix)
            self._plot_ridge_time(df, replicas, all_times, dex, thr,
                                  save_dir, display,
                                  title_suffix=suffix)
            self._plot_line_time(df, replicas, all_times, dex,
                                 save_dir, display,
                                 title_suffix=suffix)

        elif conc_sweep:
            t0     = times[0]
            suffix = f"{t0}min_concentration-sweep"
            self._plot_conc_sweep(df, replicas, all_concs, t0, thr,
                                  save_dir, display,
                                  title_suffix=suffix)
            self._plot_ridge_conc(df, replicas, all_concs, t0, thr,
                                  save_dir, display,
                                  title_suffix=suffix)
            self._plot_line_conc(df, replicas, all_concs, t0,
                                 save_dir, display,
                                 title_suffix=suffix)

        else:  # both‐varying
            base   = "time-concentration-sweep"
            # time‐panels for each conc
            for dex in concs:
                suffix = f"{dex}nM_{base}"
                self._plot_time_sweep(df, replicas, all_times, dex, thr,
                                      save_dir, display,
                                      title_suffix=suffix)
                self._plot_ridge_time(df, replicas, all_times, dex, thr,
                                      save_dir, display,
                                      title_suffix=suffix)
                self._plot_line_time(df, replicas, all_times, dex,
                                     save_dir, display,
                                     title_suffix=suffix)
            # conc‐panels for each time
            for t0 in times:
                suffix = f"{t0}min_{base}"
                self._plot_conc_sweep(df, replicas, all_concs, t0, thr,
                                      save_dir, display,
                                      title_suffix=suffix)
                self._plot_ridge_conc(df, replicas, all_concs, t0, thr,
                                      save_dir, display,
                                      title_suffix=suffix)
                self._plot_line_conc(df, replicas, all_concs, t0,
                                     save_dir, display,
                                     title_suffix=suffix)


    def _plot_time_sweep(self, df, reps, times, dex, thr, save_dir, display, title_suffix=None):
        sub = df[(df.dex_conc==dex)|((df.dex_conc==0)&(df.time==0))].copy()
        sub['has_ts'] = (sub.num_ts>=1).astype(int)

        # replicate‐means
        rep_frac = (
            sub.groupby(['replica','time'])['has_ts']
               .mean().unstack('replica')
               .reindex(times, fill_value=0)
        )
        # overall
        ov = (
            rep_frac.stack().reset_index(name='frac')
                    .groupby('time')['frac']
                    .agg(['mean','std'])
                    .reindex(times, fill_value=0)
        )

        fig, ax = plt.subplots(figsize=(8,4))
        pal = sns.color_palette("rocket_r", len(reps))
        rep_frac.plot(kind='bar', ax=ax, color=pal, width=0.8)

        # overall bar
        ax.bar(
            np.arange(len(times)),
            ov['mean'],
            yerr=ov['std'],
            width=1.0,
            color='grey',
            alpha=0.3,
            capsize=5,
            label='Overall'
        )

        ax.set_xticks(np.arange(len(times)))
        ax.set_xticklabels([f"{t} min" for t in times])
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Fraction ≥1 TS")
        title = "TS fraction"
        if title_suffix:
            title += f" — {title_suffix}"
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.02,1), loc='upper left')
        plt.tight_layout()
        if save_dir:
            fn = f"tsbar_time_{dex}nM.png" if not title_suffix else \
                 f"tsbar_time_{title_suffix.replace(' ','_')}.png"
            fig.savefig(os.path.join(save_dir, fn), dpi=300)
        if display: plt.show()
        plt.close(fig)


    def _plot_ridge_time(self, df, reps, times, dex, thr, save_dir, display, title_suffix=None):
        sub     = df[(df.dex_conc==dex) | ((df.dex_conc==0)&(df.time==0))].copy()
        metrics = ['num_nuc_spots','num_cyto_spots','num_spots']
        H       = 0.9      # fixed ridge height
        xs_dict = {}       # cache xs per metric

        for m in metrics:
            # ── 1) SUMMARY TABLE ──────────────────────────────────────
            summary = (
                sub
                .groupby(['replica','time'])[m]
                .agg(count='count', mean='mean', std='std')
                .reset_index()
            )
            print(f"\nSummary for {m} at {dex} nM:")
            print(summary.to_string(index=False))

            # ── 2) PREPARE GRID & GLOBAL PEAK ─────────────────────────
            xs = np.linspace(sub[m].min(), sub[m].max(), 200)
            xs_dict[m] = xs
            global_max = 0.0

            # include each replica/time
            for rep in reps:
                for t in times:
                    data = sub[(sub.replica==rep)&(sub.time==t)][m].dropna().values
                    if len(data)>=2:
                        dens = gaussian_kde(data)(xs)
                        global_max = max(global_max, dens.max())

            # include the pooled distributions at each time
            for t in times:
                data = sub[sub.time==t][m].dropna().values
                if len(data)>=2:
                    dens = gaussian_kde(data)(xs)
                    global_max = max(global_max, dens.max())

            scale = H / global_max if global_max>0 else 1.0

            # ── 3) SET UP FIGURE ──────────────────────────────────────
            fig, ax = plt.subplots(figsize=(8, len(times)*1.2))
            title = f"Pooled — {dex} nM Dex: {m.replace('_',' ').title()}"
            if title_suffix:
                title += f" — {title_suffix}"
            fig.suptitle(title, fontsize=14)

            # ── 4) OVERLAY “OVERALL” ──────────────────────────────────
            overall_handle = None
            for ti, t in enumerate(times):
                data = sub[sub.time==t][m].dropna().values
                if len(data) < 2:
                    continue
                kde    = gaussian_kde(data)
                y_pool = kde(xs) * scale
                y0     = len(times)-1 - ti

                overall_handle = ax.fill_between(
                    xs, y0, y0 + y_pool,
                    color='black', alpha=0.7,
                    label='Overall' if overall_handle is None else None
                )

            # ── 5) EACH REPLICA ───────────────────────────────────────
            pal         = sns.color_palette("rocket_r", n_colors=len(reps))
            rep_handles = []
            for i, rep in enumerate(reps):
                color = pal[i]
                handle = None

                for ti, t in enumerate(times):
                    data = sub[(sub.replica==rep)&(sub.time==t)][m].dropna().values
                    if len(data) < 2:
                        y = np.zeros_like(xs)
                    else:
                        kde = gaussian_kde(data)
                        y   = kde(xs) * scale

                    y0 = len(times)-1 - ti
                    if ti==0:
                        handle, = ax.plot(
                            xs, y0 + y,
                            color=color, alpha=0.8,
                            label=rep
                        )
                    else:
                        ax.plot(xs, y0 + y, color=color, alpha=0.8)
                    ax.fill_between(xs, y0, y0 + y, color=color, alpha=0.1)

                rep_handles.append(handle)

            # ── 6) CDF THRESHOLDS ─────────────────────────────────────
            lo, hi = thr[m]
            ax.axvline(lo, color='red', linestyle='--', linewidth=1)
            ax.axvline(hi, color='red', linestyle='-',  linewidth=1)

            # ── 7) AXES & LEGEND ──────────────────────────────────────
            ax.set_yticks([len(times)-1 - i for i in range(len(times))])
            ax.set_yticklabels([f"{t} min" for t in times])
            ax.set_xlabel("mRNA Count")

            handles = [overall_handle] + rep_handles
            labels  = ['Overall'] + reps
            ax.legend(handles=handles, labels=labels,
                      title="Distribution",
                      bbox_to_anchor=(1.02,1), loc='upper left')

            plt.tight_layout(rect=[0,0,1,0.95])
            if save_dir:
                fn = f"ridge_time_reps_{dex}nM_{m}.png"
                fig.savefig(os.path.join(save_dir, fn), dpi=300, bbox_inches='tight')
            if display:
                plt.show()
            plt.close(fig)


    def _plot_line_time(self, df, reps, times, dex, save_dir, display, title_suffix=None):
        # subset to this concentration + control
        sub = df[(df.dex_conc == dex) | ((df.dex_conc == 0) & (df.time == 0))].copy()

        # prepare 3 panels: nuclear, cytoplasmic, total
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)
        panels = [
            ('num_nuc_spots',   'blue',   'Nuclear'),
            ('num_cyto_spots',  'darkorange', 'Cytoplasmic'),
            ('num_spots',       'black',  'Total'),
        ]

        for ax, (m, color, label) in zip(axes, panels):
            # 1) build rm: each replica's mean at each time
            rm = (
                sub
                .groupby(['replica', 'time'])[m]
                .mean()
                .reset_index(name='mean')   # now column is "mean"
            )

            # 2) summarize rm → sm (mean of means ± std of means)
            sm = (
                rm
                .groupby('time')['mean']
                .agg(['mean','std'])            # default columns: "mean","std"
                .rename(columns={'mean':'mean_m','std':'sd_m'})
                .reindex(times, fill_value=0)
            )

            # 3) overall pooled: mean ± std on the raw data
            ov = (
                sub
                .groupby('time')[m]
                .agg(['mean','std'])
                .rename(columns={'mean':'mean_o','std':'sd_o'})
                .reindex(times, fill_value=0)
            )

            # 4) plot replica‐means line + error bars
            ax.errorbar(
                sm.index, sm['mean_m'], yerr=sm['sd_m'],
                fmt='-o', color=color, capsize=5,
                label='Replicate means ± SD'
            )

            # 5) overall shaded band + dashed line + diamonds
            ax.fill_between(
                ov.index,
                ov['mean_o'] - ov['sd_o'],
                ov['mean_o'] + ov['sd_o'],
                color=color, alpha=0.2
            )
            ax.plot(
                ov.index, ov['mean_o'],
                '--D', color=color,
                label='Overall mean ± SD'
            )

            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Mean ± SD")
            ax.set_title(label)
            ax.legend(loc='upper left')

        # main title
        main_ttl = "Line plots"
        if title_suffix:
            main_ttl = f"{title_suffix} — {main_ttl}"
        fig.suptitle(main_ttl, y=1.02, fontsize=16, weight='bold')
        plt.tight_layout()

        if save_dir:
            fn = f"line_time_{dex}nM.png" if not title_suffix else \
                 f"line_time_{title_suffix.replace(' ','_')}.png"
            fig.savefig(os.path.join(save_dir, fn), dpi=300, bbox_inches='tight')
        if display:
            plt.show()
        plt.close(fig)


    def _plot_conc_sweep(self, df, reps, concs, time_pt, thr, save_dir, display, title_suffix=None):
        sub = df[(df.time==time_pt)|((df.dex_conc==0)&(df.time==0))].copy()
        sub['has_ts'] = (sub.num_ts>=1).astype(int)

        rep_frac = (
            sub.groupby(['replica','dex_conc'])['has_ts']
               .mean().unstack('replica')
               .reindex(concs, fill_value=0)
        )
        ov = (
            rep_frac.stack().reset_index(name='frac')
                    .groupby('dex_conc')['frac']
                    .agg(['mean','std'])
                    .reindex(concs, fill_value=0)
        )

        nonz = [c for c in concs if c>0]
        eps = min(nonz)*1e-2 if nonz else 1e-3
        xs = [eps if c==0 else c for c in concs]

        fig, ax = plt.subplots(figsize=(8,4))
        pal = sns.color_palette("rocket_r", len(reps))
        rep_frac.plot(kind='bar', ax=ax, color=pal, width=0.8)

        ax.bar(
            np.arange(len(concs)),
            ov['mean'], yerr=ov['std'],
            width=1.0, color='grey',
            alpha=0.3, capsize=5,
            label='Overall'
        )

        # ax.set_xscale('log')
        ax.set_xticks(range(len(concs)))
        ax.set_xticklabels(['0']+[str(c) for c in nonz])
        ax.set_xlabel("Dex_Conc (nM)")
        ax.set_ylabel("Fraction ≥1 TS")
        ttl = "TS fraction"
        if title_suffix:
            ttl += f" — {title_suffix}"
        ax.set_title(ttl)
        ax.legend(bbox_to_anchor=(1.02,1), loc='upper left')
        plt.tight_layout()
        if save_dir:
            fn = f"tsbar_conc_{time_pt}min.png" if not title_suffix else \
                 f"tsbar_conc_{title_suffix.replace(' ','_')}.png"
            fig.savefig(os.path.join(save_dir, fn), dpi=300)
        if display: plt.show()
        plt.close(fig)


    def _plot_ridge_conc(self, df, reps, concs, time_pt, thr, save_dir, display, title_suffix=None):
        sub   = df[(df.time==time_pt) | ((df.dex_conc==0)&(df.time==0))].copy()
        rows  = [0] + [c for c in concs if c>0]
        n     = len(rows)
        metrics = ['num_nuc_spots','num_cyto_spots','num_spots']
        H     = 0.8
        xs_dict = {}

        for m in metrics:
            # ── 1) SUMMARY TABLE ──────────────────────────────────
            summary = (
                sub
                .groupby(['replica','dex_conc'])[m]
                .agg(count='count', mean='mean', std='std')
                .reset_index()
            )
            print(f"\nSummary for {m} at t={time_pt} min:")
            print(summary.to_string(index=False))

            # ── 2) GRID + GLOBAL MAX ──────────────────────────────
            xs = np.linspace(sub[m].min(), sub[m].max(), 200)
            xs_dict[m] = xs
            global_max = 0.0

            # each rep/conc
            for rep in reps:
                for c in rows:
                    data = sub[(sub.replica==rep)&(sub.dex_conc==c)][m].dropna().values
                    if len(data)>=2:
                        dens = gaussian_kde(data)(xs)
                        global_max = max(global_max, dens.max())

            # pooled per conc
            for c in rows:
                data = sub[sub.dex_conc==c][m].dropna().values
                if len(data)>=2:
                    dens = gaussian_kde(data)(xs)
                    global_max = max(global_max, dens.max())

            scale = H / global_max if global_max>0 else 1.0

            # ── 3) SETUP FIGURE ────────────────────────────────────
            fig, ax = plt.subplots(figsize=(8, n*1.2))
            ttl   = m.replace('_',' ').title()
            title = f"Pooled — {time_pt} min: {ttl}"
            if title_suffix:
                title = f"{title_suffix} — {title}"
            fig.suptitle(title, fontsize=14)

            # ── 4) OVERLAY “OVERALL” ───────────────────────────────
            overall_handle = None
            for i, c in enumerate(rows):
                data = sub[sub.dex_conc==c][m].dropna().values
                if len(data)<2:
                    continue
                kde    = gaussian_kde(data)
                y_pool = kde(xs) * scale
                y0     = n-1 - i

                overall_handle = ax.fill_between(
                    xs, y0, y0 + y_pool,
                    color='black', alpha=0.7,
                    label='Overall' if overall_handle is None else None
                )

            # ── 5) PER-REPLICA ─────────────────────────────────────
            pal = dict(zip(reps, sns.color_palette("rocket_r", len(reps))))
            rep_handles = []
            for rep in reps:
                handle = None
                for i, c in enumerate(rows):
                    data = sub[(sub.replica==rep)&(sub.dex_conc==c)][m].dropna().values
                    if len(data)<2:
                        y = np.zeros_like(xs)
                    else:
                        kde = gaussian_kde(data)
                        y   = kde(xs) * scale
                    y0 = n-1 - i

                    if handle is None:
                        handle, = ax.plot(
                            xs, y0 + y,
                            color=pal[rep], alpha=0.8,
                            label=rep
                        )
                    else:
                        ax.plot(xs, y0 + y, color=pal[rep], alpha=0.8)
                    ax.fill_between(xs, y0, y0 + y, color=pal[rep], alpha=0.1)

                rep_handles.append(handle)

            # ── 6) CDF LINES & AXES ─────────────────────────────────
            lo, hi = thr[m]
            ax.axvline(lo, linestyle='--', color='red', linewidth=1)
            ax.axvline(hi, linestyle='-',  color='red', linewidth=1)

            ax.set_yticks([n-1 - i for i in range(n)])
            ax.set_yticklabels([f"{c} nM" for c in rows])
            ax.set_xlabel("mRNA Count")

            # ── 7) LEGEND ───────────────────────────────────────────
            handles = [overall_handle] + rep_handles
            labels  = ['Overall'] + reps
            ax.legend(handles=handles, labels=labels,
                      title="Distribution",
                      bbox_to_anchor=(1.02,1), loc='upper left')

            plt.tight_layout(rect=[0,0,1,0.95])
            if save_dir:
                fn = f"ridge_conc_{time_pt}min_{m}.png"
                fig.savefig(os.path.join(save_dir, fn), dpi=300, bbox_inches='tight')
            if display:
                plt.show()
            plt.close(fig)

    def _plot_line_conc(self, df, reps, concs, time_pt, save_dir, display, title_suffix=None):
        # subset to this time point + control
        sub = df[(df['time'] == time_pt) | ((df['dex_conc'] == 0) & (df['time'] == 0))].copy()

        # panels = nuclear, cytoplasmic, total
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)
        panels = [
            ('num_nuc_spots',   'blue',   'Nuclear'),
            ('num_cyto_spots',  'darkorange', 'Cytoplasmic'),
            ('num_spots',       'black',  'Total'),
        ]

        # compute a tiny positive x for the 0-conc control
        nonz = [c for c in concs if c > 0]
        eps  = (min(nonz) * 1e-2) if nonz else 1e-3

        for ax, (m, color, label) in zip(axes, panels):
            # 1) replicate-means
            rm = (
                sub
                .groupby(['replica','dex_conc'])[m]
                .mean()
                .reset_index(name='mean')
            )

            # 2) summarize replicate-means
            sm = (
                rm
                .groupby('dex_conc')['mean']
                .agg(['mean','std'])
                .rename(columns={'mean':'mean_m','std':'sd_m'})
                .reindex(concs, fill_value=0)
            )

            # 3) overall pooled
            ov = (
                sub
                .groupby('dex_conc')[m]
                .agg(['mean','std'])
                .rename(columns={'mean':'mean_o','std':'sd_o'})
                .reindex(concs, fill_value=0)
            )

            # map 0 → eps for plotting
            x_sm = [eps if c==0 else c for c in sm.index]
            x_ov = [eps if c==0 else c for c in ov.index]

            # 4) plot replicate-means line + error bars
            ax.errorbar(
                x_sm, sm['mean_m'], yerr=sm['sd_m'],
                fmt='-o', color=color, capsize=5,
                label='Replicate means ± SD'
            )

            # 5) overall shaded band + dashed line + diamonds
            ax.fill_between(
                x_ov,
                ov['mean_o'] - ov['sd_o'],
                ov['mean_o'] + ov['sd_o'],
                color=color, alpha=0.2
            )
            ax.plot(
                x_ov, ov['mean_o'],
                '--D', color=color,
                label='Overall mean ± SD'
            )

            # log scale + force ticks at [eps, ...positive concs...]
            ax.set_xscale('log')
            ax.set_xticks([eps] + nonz)
            ax.set_xticklabels(['0'] + [str(c) for c in nonz])

            ax.set_xlabel("Dex_Conc (nM)")
            ax.set_ylabel("Mean ± SD")
            ax.set_title(label)
            ax.legend(loc='upper left')

        # super‐title
        ttl = "Line plots"
        if title_suffix:
            ttl = f"{title_suffix} — {ttl}"
        fig.suptitle(ttl, y=1.02, fontsize=16, weight='bold')
        plt.tight_layout()

        # save & show
        if save_dir:
            fn = f"line_conc_{time_pt}min.png" if not title_suffix else \
                 f"line_conc_{title_suffix.replace(' ','_')}.png"
            fig.savefig(os.path.join(save_dir, fn), dpi=300, bbox_inches='tight')
        if display:
            plt.show()
        plt.close(fig)