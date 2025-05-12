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
        model: Union[str, RandomForestClassifier] = None
    ):
        self.spots = spots.copy()
        self.clusters = clusters.copy()
        self.cellprops = cellprops.copy()

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
          - MG_SNR: Computed as (signal - cell_intensity_mean-0) / cell_intensity_std-0.
          
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
            pd.DataFrame: One row per cell with counts of nuclear/cytoplasmic spots, clusters, and cell metadata.
        """
        self.spots = self.spots.sort_values(by='unique_cell_id')
        self.clusters = self.clusters.sort_values(by='unique_cell_id')
        self.cellprops = self.cellprops.sort_values(by='unique_cell_id')
        cell_ids = self.cellprops['unique_cell_id']

        # Spot-based counts
        num_spots = self.spots.groupby('unique_cell_id').size().reindex(cell_ids, fill_value=0)
        num_nuc_spots = self.spots[self.spots['is_nuc'] == 1].groupby('unique_cell_id').size().reindex(cell_ids, fill_value=0)
        num_cyto_spots = self.spots[self.spots['is_nuc'] == 0].groupby('unique_cell_id').size().reindex(cell_ids, fill_value=0)

        # Cluster-based counts
        num_ts = self.clusters[self.clusters['is_nuc'] == 1].groupby('unique_cell_id').size().reindex(cell_ids, fill_value=0)
        num_foci = self.clusters[self.clusters['is_nuc'] == 0].groupby('unique_cell_id').size().reindex(cell_ids, fill_value=0)
        num_spots_ts = self.clusters[self.clusters['is_nuc'] == 1].groupby('unique_cell_id')['nb_spots'].sum().reindex(cell_ids, fill_value=0)
        largest_ts = self.clusters[self.clusters['is_nuc'] == 1].groupby('unique_cell_id')['nb_spots'].max().reindex(cell_ids, fill_value=0)
        second_largest_ts = self.clusters[self.clusters['is_nuc'] == 1].groupby('unique_cell_id')['nb_spots']\
                                .apply(self.second_largest).reindex(cell_ids, fill_value=0)
        num_spots_foci = self.clusters[self.clusters['is_nuc'] == 0].groupby('unique_cell_id')['nb_spots'].sum().reindex(cell_ids, fill_value=0)

        # Metadata and morphometrics
        results = pd.DataFrame({
            'unique_cell_id': cell_ids,
            'num_spots': num_spots.values,
            'num_nuc_spots': num_nuc_spots.values,
            'num_cyto_spots': num_cyto_spots.values,
            'num_ts': num_ts.values,
            'num_foci': num_foci.values,
            'num_spots_ts': num_spots_ts.values,
            'largest_ts': largest_ts.values,
            'second_largest_ts': second_largest_ts.values,
            'num_spots_foci': num_spots_foci.values,
            'nuc_area': self.cellprops['nuc_area'].values,
            'cyto_area': self.cellprops['cyto_area'].values,
            'time': self.cellprops['time'].values,
            'dex_conc': self.cellprops['Dex_Conc'].values,
            'replica': self.cellprops['replica'].values,
            'fov': self.cellprops['fov'].values,
            'nas_location': self.cellprops['NAS_location'].values,
            'h5_idx': self.cellprops['h5_idx'].values
        })

        return results        

#############################
# DUSP1_filtering Class
#############################

class DUSP1_filtering:
    def __init__(self, method: str = 'MG', abs_threshold: float = 4.0):
        self.method = method.lower()
        self.abs_threshold = abs_threshold

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
        spots['keep_mg_abs'] = spots['keep_mg'] | (~spots['keep_mg'] & (spots['snr'] > self.abs_threshold))

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
            cellprops=filtered_cellprops
        )
        SSITcellresults = measurer.summarize_filtered_cells()

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
import os
import random
import traceback
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import exposure

def adjust_contrast(image, lower=2, upper=98):
    image = np.array(image)  # ensure it's a NumPy array
    if image.size > 0:
        p_low, p_high = np.percentile(image, (lower, upper))
        return exposure.rescale_intensity(image, in_range=(p_low, p_high), out_range=(0, 1))
    return image

def overlay_mask(ax, image, mask, mask_cmap='rocket', alpha=0.3):
    """Overlay a mask on an image."""
    ax.imshow(image, cmap='gray')
    ax.imshow(mask, cmap=mask_cmap, alpha=alpha)

def draw_spot_circle(ax, x, y, radius=4, color='gold', linewidth=2):
    """Draw a circle around a spot."""
    circle = plt.Circle((x, y), radius, edgecolor=color, facecolor='none', linewidth=linewidth)
    ax.add_patch(circle)

def draw_spot_arrow(ax, x, y, offset=-5, color='magenta'):
    """
    Draws a small arrow from (x + offset, y) to (x, y).
    By default offset is negative => arrow from left to right.
    """
    ax.arrow(x + offset, y, -offset, 0, head_width=5,
             color=color, length_includes_head=True)    

class DUSP1DisplayManager(DUSP1AnalysisManager):
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
        with h5py.File(file_path, 'r') as h5_file:
            num_fov = h5_file['raw_images'].shape[0]
            if fov is None:
                fov = random.choice(range(num_fov))
            # Get the FOV slice; this might return an extra dimension.
            images = np.array(h5_file['raw_images'][fov])
            masks = np.array(h5_file['masks'][fov])
            # Remove the extra FOV dimension if present.
            if images.shape[0] == 1:
                images = np.squeeze(images, axis=0)
            if masks.shape[0] == 1:
                masks = np.squeeze(masks, axis=0)
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

        # (Optional) Rescale intensity if needed.
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
            -  High: > 80% removed
            -  Medium: 40% <= removed < 60%
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
            (self.cellprops['removal_percentage'] > 80),
            (self.cellprops['removal_percentage'] >= 40) & (self.cellprops['removal_percentage'] <= 60),
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
                    print(f"  No cells with removal category '{cat}' in h5_idx {h5}.")
                    continue

                # Select one representative cell at random.
                rep_cell = cat_cells.sample(1).iloc[0]
                print(f"  Selected cell with removal '{cat}' from h5_idx {h5}.")

                # Set instance variables so that the zoom functions use the same FOV.
                self.h5_idx = rep_cell['h5_idx']
                self.fov = rep_cell['fov']
                # Here, we use the 'unique_cell_id' from the cellprops.
                self.cell_label = rep_cell['unique_cell_id']

                # Compose a title in the format "conc_time_FOV{fov}_ID{unique_cell_id}".
                conc = rep_cell.get('conc', "NA")
                time_val = rep_cell.get('time', "NA")
                title_str = f"{conc}_{time_val}_FOV{rep_cell['fov']}_ID{rep_cell['unique_cell_id']}"

                # --- Plot 1: Zoomed-in view of the representative cell ---
                print(f"    Displaying cell plot: {title_str}")
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
                    print(f"    No spots found for cell {self.cell_label} in FOV {self.fov}.")
                    continue
                # Randomly select one spot from the cell and assign it to self.chosen_spot.
                self.chosen_spot = cell_spots.sample(1).iloc[0]
                
                # (Optional) If you also want to set TS and foci, you can add similar logic here.
                # For instance, if the cell has TS or foci in your clusters_df:
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
                print(f"    Displaying spot zoom for cell ID {rep_cell['unique_cell_id']}.")
                self._display_zoom_on_one_spot(spotChannel=0)
        return self.cellprops

    def display_overview_plots(self):
        """
        Display overview plots for the entire dataset.
        This function is a placeholder and should be implemented as needed.
        """
        print("Overview plots are not implemented yet.")
        # Implement your overview plotting logic here.
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np

        # ============================================================================
        # Settings and Data
        # ============================================================================
        metrics = ['nuc_MG_count', 'cyto_MG_count', 'MG_count', 'num_ts', 'num_foci']

        # Sorted unique values for concentrations and time
        concentrations = sorted(filtered_cell_level_results['dex_conc'].unique())
        timepoints = sorted(filtered_cell_level_results['time'].unique())

        # Set common aesthetics
        sns.set_context('talk')
        sns.set_style('whitegrid')

        # Make a copy of the main dataframe
        df = filtered_cell_level_results.copy()

        # ============================================================================
        # Get the control (baseline) data: all rows with time == 0.
        # ============================================================================
        reference_data = df[df['time'] == 0]
        print("Reference (control) data sample:")
        print(reference_data.head())

        # For the histograms below, we also calculate CDF thresholds based on one metric.
        # (In this example, we use 'nuc_MG_count'; update as needed for other metrics.)
        cdf_values = np.sort(reference_data['nuc_MG_count'])
        cdf = np.arange(1, len(cdf_values) + 1) / len(cdf_values)
        cdf_50_threshold = np.interp(0.50, cdf, cdf_values)
        cdf_95_threshold = np.interp(0.95, cdf, cdf_values)

        # Define the concentrations and desired timepoints for the histograms (e.g., concentration 100 only)
        concentrations_to_plot = [100]  # modify as needed
        desired_timepoints = [10, 20, 30, 40, 50, 60, 75, 90, 120, 150, 180]


        # ============================================================================
        # 1. HISTOGRAMS
        # For each desired timepoint and concentration, compare the histogram for the 
        # experimental condition (given dex_conc and time) with the control (time==0) data.
        # ============================================================================
        for time in desired_timepoints:
            for dex_conc in concentrations_to_plot:
                plt.figure(figsize=(10, 6))
                specific_data = df[(df['dex_conc'] == dex_conc) & (df['time'] == time)]
                
                sns.histplot(reference_data['nuc_MG_count'], color='grey', label='Control (0 min)', kde=True)
                sns.histplot(specific_data['nuc_MG_count'], color='blue', label=f'{dex_conc} nM, {time} min', kde=True)
                
                plt.axvline(cdf_50_threshold, color='red', linestyle='--', label='CDF 50% Threshold')
                plt.axvline(cdf_95_threshold, color='red', linestyle='-',  label='CDF 95% Threshold')
                
                plt.annotate(f'Ref Cell Count: {len(reference_data)}\nSpec Cell Count: {len(specific_data)}',
                            xy=(0.77, 0.70), xycoords='axes fraction', verticalalignment='top')
                plt.title(f'Nuclear Distribution Comparison: Control vs {dex_conc} nM, {time} min')
                plt.xlabel('nuc_MG_count')
                plt.ylabel('Density')
                plt.legend()
                plt.show()


        # ============================================================================
        # 2. LINE PLOTS (with control overlay)
        # For each concentration, plot the mean metric value over time
        # and overlay a horizontal dashed line indicating the control (0 min) mean.
        # ============================================================================
        for conc in concentrations:
            fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
            fig.suptitle(f'Line Plots with Shared Control — Concentration {conc} nM', fontsize=16)

            # Experimental data for this concentration
            data_conc = df[df['dex_conc'] == conc]

            # Control data (baseline): dex_conc == 0 and time == 0
            control_data = df[(df['dex_conc'] == 0) & (df['time'] == 0)]

            for i, metric in enumerate(metrics):
                ax = axes[i]

                # Experimental mean metric over time
                grouped_exp = data_conc.groupby('time')[metric].mean().reset_index()

                # Add baseline mean as the 0 min point
                if not control_data.empty:
                    control_mean = control_data[metric].mean()
                    # Create a new row at time=0
                    control_point = pd.DataFrame({'time': [0], metric: [control_mean]})
                    # Concatenate with experimental data
                    combined = pd.concat([control_point, grouped_exp], ignore_index=True)
                else:
                    combined = grouped_exp.copy()

                sns.lineplot(data=combined, x='time', y=metric, marker='o', ax=ax, label=f'{conc} nM + Control')

                ax.set_title(metric)
                ax.set_xlabel('Time (min)')
                ax.set_ylabel(f'Mean {metric}')
                ax.legend()

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()


        # ============================================================================
        # 3. BAR PLOTS (with control overlay)
        # For each concentration, plot a bar chart displaying the mean metric value at each timepoint
        # and overlay a horizontal dashed line indicating the control (0 min) mean.
        # ============================================================================
        for conc in concentrations:
            fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
            fig.suptitle(f'Bar Plots for Concentration {conc} nM', fontsize=16)
            
            # Data for this concentration
            data_conc = df[df['dex_conc'] == conc]
            
            for i, metric in enumerate(metrics):
                ax = axes[i]
                df_grouped = data_conc.groupby('time')[metric].mean().reset_index()
                sns.barplot(data=df_grouped, x='time', y=metric, ax=ax, palette='viridis')
                
                # Compute control mean from reference
                control_mean = reference_data[metric].mean()
                ax.axhline(control_mean, color='black', linestyle='--', label='Control (0 min)')
                
                ax.set_title(metric)
                ax.set_xlabel('Time (min)')
                ax.set_ylabel(f'Mean {metric}')
                ax.legend()
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()


        # ============================================================================
        # 4. CATEGORY-BASED BAR PLOTS FOR 'num_ts' and 'num_foci'
        # For each concentration, display for each timepoint the fraction (percentage) of cells 
        # falling into the categories "0", "1", "2", "3", or ">=4".
        # Now baseline (control) data is included by concatenating the control data.
        # ============================================================================
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
                return '0'

        cat_metrics = ['num_ts', 'num_foci']

        for conc in concentrations:
            data_conc = df[df['dex_conc'] == conc]
            control_data = df[(df['time'] == 0) & (df['dex_conc'] == 0)]
            data_for_plot = pd.concat([control_data, data_conc], ignore_index=True)
            time_points = sorted(data_for_plot['time'].unique())

            fig, axes = plt.subplots(len(time_points), len(cat_metrics), 
                                    figsize=(8 * len(cat_metrics), 4 * len(time_points)))

            # ---- MAKE AXES ALWAYS 2D ----
            if len(time_points) == 1 and len(cat_metrics) == 1:
                axes = np.array([[axes]])
            elif len(time_points) == 1:
                axes = np.expand_dims(axes, axis=0)
            elif len(cat_metrics) == 1:
                axes = np.expand_dims(axes, axis=1)
            # -----------------------------

            fig.suptitle(f'Percentage of Cells by TS Category for {conc} nM (including control)', fontsize=16)

            for row, t in enumerate(time_points):
                for col, metric in enumerate(cat_metrics):
                    ax = axes[row][col]
                    subset = data_for_plot[data_for_plot['time'] == t].copy()
                    subset['category'] = subset[metric].apply(cat_func)

                    counts = subset['category'].value_counts(normalize=True).sort_index() * 100
                    categories = ['0', '1', '2', '3', '>=4']
                    counts = counts.reindex(categories, fill_value=0)

                    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette='viridis')
                    ax.set_title(f'{metric} at {t} min')
                    ax.set_xlabel('Category')
                    ax.set_ylabel('Percentage (%)')

                    for i, v in enumerate(counts.values):
                        ax.text(i, v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=9)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

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
        
        # print("Running display_cell_TS_foci_variations...")
        # self.display_cell_TS_foci_variations(spotChannel=0, cytoChannel=1, nucChannel=2)
        print("All display routines completed.")


#############################
# SpotCropSampler Class
#############################
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from skimage import exposure
from tqdm import tqdm
from contextlib import contextmanager

class SpotCropSampler:
    def __init__(
        self,
        spots_df,
        clusters_df,
        cellprops_df,
        mount_prefix: str = "/Volumes/share"
    ):
        import os

        self.spots     = spots_df
        self.clusters  = clusters_df
        self.cellprops = cellprops_df

        # Build a mapping h5_idx -> NAS_location (as you already have it)
        raw_map = (
            self.cellprops[['h5_idx','NAS_location']]
                .drop_duplicates(subset='h5_idx')
                .set_index('h5_idx')['NAS_location']
                .to_dict()
        )

        self.file_paths = {}
        for idx, rel in raw_map.items():
            # 1) form absolute candidate
            full = rel if os.path.isabs(rel) else os.path.join(mount_prefix, rel)

            # 2) if that path is a directory OR not a file, try popping one level up
            if not os.path.isfile(full):
                parent = os.path.dirname(full)
                grand = os.path.dirname(parent)
                cand2  = os.path.join(grand, os.path.basename(full))
                if os.path.isfile(cand2):
                    full = cand2
                else:
                    raise FileNotFoundError(
                        f"h5_idx {idx}: neither '{full}' nor '{cand2}' exists"
                    )

            # 3) record the valid .h5 path
            self.file_paths[idx] = full

    @contextmanager
    def open_h5(self, h5_idx):
        """Yield the raw_images dataset for that h5_idx."""
        path = self.file_paths[h5_idx]
        with h5py.File(path, 'r') as f:
            yield f['raw_images'], f['masks']

    def get_images_and_masks(self, h5_idx):
        """
        Load and project one FOV from HDF5:
        - randomly pick fov if unspecified
        - returns (images, masks, chosen_h5_idx, chosen_fov)
        """
        images_ds, masks_ds = None, None
        with self.open_h5(h5_idx) as (raw_images, raw_masks):
            num_fov = raw_images.shape[0]
            fov = random.randrange(num_fov)
            imgs = np.array(raw_images[fov])
            msks = np.array(raw_masks[fov])
            # squeeze singleton leading dim
            if imgs.shape[0] == 1:
                imgs = imgs.squeeze(0)
            if msks.shape[0] == 1:
                msks = msks.squeeze(0)
            images_ds, masks_ds = imgs, msks
        return images_ds, masks_ds, h5_idx, fov

    def _select_cells(self, h5_idx, fov, raw_shape):
        """Pick 4 quadrant cells + one central cell, by bounding-box centroids."""
        df = self.cellprops[
            (self.cellprops['h5_idx'] == h5_idx) &
            (self.cellprops['fov']    == fov)
        ].copy()
        if df.empty:
            return pd.DataFrame([])

        H, W = raw_shape[-2], raw_shape[-1]
        xm, ym = W/2, H/2

        # compute centroids from cell_bbox-*
        df['centroid_x'] = (df['cell_bbox-1'] + df['cell_bbox-3'])/2
        df['centroid_y'] = (df['cell_bbox-0'] + df['cell_bbox-2'])/2

        quads = [
            lambda x,y: x <  xm and y <  ym,
            lambda x,y: x >= xm and y <  ym,
            lambda x,y: x <  xm and y >= ym,
            lambda x,y: x >= xm and y >= ym,
        ]
        picks = []
        for quad in quads:
            qdf = df[df.apply(lambda r: quad(r['centroid_x'], r['centroid_y']), axis=1)]
            if not qdf.empty:
                picks.append(qdf.sample(1))

        # center-most
        df['dist2ctr'] = (df['centroid_x']-xm)**2 + (df['centroid_y']-ym)**2
        picks.append(df.loc[[df['dist2ctr'].idxmin()]])

        return pd.concat(picks).drop_duplicates().reset_index(drop=True)

    def _crop_spots_in_cell(
        self, raw_3d, h5_idx, fov, cell_id,
        display, save, save_dir, pad=3, max_spots=20
    ):
        sdf = self.spots[
            (self.spots.h5_idx == h5_idx) &
            (self.spots.fov    == fov) &
            (self.spots.unique_cell_id == cell_id)
        ]
        if sdf.empty:
            return []

        # --- Define categories ---
        likely_true = sdf[sdf['MG_pass']]
        fail_and_abs = sdf[~sdf['MG_pass'] & sdf.get('absolute', False)]
        mg_underestimates_snr = sdf[~sdf['MG_pass'] & (sdf.get('snr_vs_mg', 0) < 0)]
        confirmed_negatives = sdf[~sdf['MG_pass'] & (sdf.get('snr_vs_mg', 0) > 0)]

        # --- Prioritized sampling ---
        parts = []
        used_ids = set()

        def sample_unique(df, n):
            df = df[~df['unique_spot_id'].isin(used_ids)]
            s = df.sample(n=min(n, len(df)))
            used_ids.update(s['unique_spot_id'])
            return s

        # Try to get at least half from interesting negatives (false negatives or MG underestimations)
        n_interesting = max_spots * 3 // 5
        n_fail_and_abs = int(n_interesting * 0.5)
        n_underestimates = n_interesting - n_fail_and_abs

        if not fail_and_abs.empty:
            parts.append(sample_unique(fail_and_abs, n_fail_and_abs))
        if not mg_underestimates_snr.empty:
            parts.append(sample_unique(mg_underestimates_snr, n_underestimates))

        # Then get likely true positives
        remaining = max_spots - sum(len(p) for p in parts)
        if not likely_true.empty and remaining > 0:
            parts.append(sample_unique(likely_true, remaining))

        # Fill remainder with confirmed negatives (e.g., to train against true negatives)
        remaining = max_spots - sum(len(p) for p in parts)
        if not confirmed_negatives.empty and remaining > 0:
            parts.append(sample_unique(confirmed_negatives, remaining))

        # Final set
        if parts:
            sdf = pd.concat(parts).drop_duplicates('unique_spot_id').sample(frac=1).reset_index(drop=True)
        else:
            return []

        crops_info = []
        Z, H, W = raw_3d.shape

        for _, spot in sdf.iterrows():
            x0, y0, z0 = int(spot.x_px), int(spot.y_px), int(spot.z_px)
            x1, x2 = x0-pad, x0+pad+1
            y1, y2 = y0-pad, y0+pad+1
            z1, z2 = max(z0 - 2, 0), min(z0 + 3, Z)

            if x1<0 or y1<0 or x2>W or y2>H:
                continue

            patch_3d = raw_3d[z1:z2, y1:y2, x1:x2]
            if patch_3d.shape[1:] != (2*pad+1, 2*pad+1) or patch_3d.shape[0] < 1:
                continue

            patch = np.max(patch_3d, axis=0)
            p1, p99 = np.percentile(patch, (1,99))
            patch = exposure.rescale_intensity(patch, in_range=(p1,p99), out_range=(0,1))

            # metadata counts
            others = self.spots[
                (self.spots.h5_idx==h5_idx)&
                (self.spots.fov   ==fov)&
                (self.spots.x_px.between(x1,x2-1))&
                (self.spots.y_px.between(y1,y2-1))
            ]
            n_others = len(others)-1
            ts   = self.clusters.query(
                "h5_idx==@h5_idx & fov==@fov & is_nuc==True & "
                "x_px.between(@x1,@x2-1) & y_px.between(@y1,@y2-1)"
            )
            foci = self.clusters.query(
                "h5_idx==@h5_idx & fov==@fov & is_nuc==False & "
                "x_px.between(@x1,@x2-1) & y_px.between(@y1,@y2-1)"
            )
            # get the spot's cluster
            cid = spot.get('cluster_index', None)

            # find that cluster in ts / foci
            ts_entry   = ts[ts['cluster_index'] == cid]
            foci_entry = foci[foci['cluster_index'] == cid]

            ts_size   = int(ts_entry['nb_spots'].iloc[0])   if not ts_entry.empty   else 0
            foci_size = int(foci_entry['nb_spots'].iloc[0]) if not foci_entry.empty else 0

            meta = {
                'h5_idx':            h5_idx,
                'replica':          spot['replica'],
                'fov':               fov,
                'unique_cell_id':    cell_id,
                'unique_spot_id':    spot['unique_spot_id'],
                'cluster_index':     spot['cluster_index'],
                'time':             spot['time'],
                'Dex_conc':        spot['Dex_Conc'],
                'is_nuc':            spot['is_nuc'],
                'signal':           spot['signal'],
                'snr':               spot['snr'],
                'x_px':             spot['x_px'],
                'y_px':             spot['y_px'],
                'z_px':             spot['z_px'],
                'MG_SNR':            spot['MG_SNR'],
                'MG_pass':           spot['MG_pass'],
                'num_spots_in_crop': n_others + 1,
                'num_TS_in_crop':    len(ts),
                'num_foci_in_crop':  len(foci),
                'ts_cluster_size':   ts_size,
                'foci_cluster_size': foci_size,
                'cell intensity mean': spot['cell_intensity_mean-0'],
                'cell intensity std':  spot['cell_intensity_std-0'],
                'nuc intensity mean': spot['nuc_intensity_mean-0'],
                'nuc intensity std':  spot['nuc_intensity_std-0'],
                'cyto intensity mean': spot['cyto_intensity_mean-0'],
                'cyto intensity std':  spot['cyto_intensity_std-0'],
                'snr vs mg': spot['snr_vs_mg'],
                'absolute pass': spot['absolute'],
                'weighted pass': spot['weighted'],
                'mg less than snr': spot['mg_lt_snr'],
            }

        # Precompute full-cell patch from max-projection for display
        if display > 0:
            cdf = self.cellprops[
                (self.cellprops.h5_idx == h5_idx) &
                (self.cellprops.fov    == fov) &
                (self.cellprops.unique_cell_id == cell_id)
            ]
            if not cdf.empty:
                r0, c0, r1, c1 = map(int, cdf.iloc[0][[
                    'cell_bbox-0','cell_bbox-1','cell_bbox-2','cell_bbox-3'
                ]])
                # Project full 3D image for display of whole cell
                full_proj = np.max(raw_3d, axis=0)
                cell_patch = full_proj[r0:r1, c0:c1]
                p1, p99 = np.percentile(cell_patch, (1,99))
                cell_patch = exposure.rescale_intensity(
                    cell_patch, in_range=(p1, p99), out_range=(0, 1)
                )
            else:
                display = 0

        # Loop over selected spots
        for _, spot in sdf.iterrows():

            if display > 0:
                fig, (axC, axP) = plt.subplots(1, 2, figsize=(10, 5))

                # --- LEFT: whole-cell patch, gold = MG_pass, blue = chosen spot ---
                axC.imshow(cell_patch, cmap='gray')
                mg_spots = self.spots[
                    (self.spots.h5_idx == h5_idx) &
                    (self.spots.fov    == fov) &
                    (self.spots.unique_cell_id == cell_id) &
                    (self.spots.MG_pass)
                ]
                for _, m in mg_spots.iterrows():
                    rx, ry = m.x_px - c0, m.y_px - r0
                    if 0 <= rx < cell_patch.shape[1] and 0 <= ry < cell_patch.shape[0]:
                        circle = plt.Circle((rx, ry), 2, edgecolor='gold', facecolor='none', linewidth=1)
                        axC.add_patch(circle)

                rx0, ry0 = x0 - c0, y0 - r0
                circle = plt.Circle((rx0, ry0), 2, edgecolor='blue', facecolor='none', linewidth=2)
                axC.add_patch(circle)
                axC.set_title(f"Cell {cell_id} (max Z-projection)")
                axC.axis('off')

                # --- RIGHT: 7×7 crop from 5-slice Z projection ---
                axP.imshow(patch, cmap='gray')
                for _, s2 in others.iterrows():
                    rx, ry = s2.x_px - x1, s2.y_px - y1
                    if (rx, ry) != (pad, pad):
                        circle = plt.Circle((rx, ry), 1, edgecolor='gold', facecolor='none', linewidth=1)
                        axP.add_patch(circle)

                center_col = 'blue'
                if spot.cluster_index in ts['cluster_index'].values:
                    center_col = 'magenta'
                elif spot.cluster_index in foci['cluster_index'].values:
                    center_col = 'cyan'

                circle = plt.Circle((rx0, ry0), 2, edgecolor=center_col, facecolor='none', linewidth=2)
                axP.add_patch(circle)

                axP.set_title(
                    f"Spot {spot.unique_spot_id} (5-slice Z-max)\n"
                    f"MG_SNR={spot.MG_SNR:.2f}, SNR={spot.snr:.2f}\n"
                    f"{meta['num_spots_in_crop']} spots in patch"
                )
                axP.axis('off')

                plt.tight_layout()
                plt.show()
                display -= 1

            if save and save_dir:
                fn = f"crop_h5{h5_idx}_f{fov}_c{cell_id}_s{spot.unique_spot_id}.npy"
                np.save(os.path.join(save_dir, fn), patch)

            crops_info.append({'crop': patch, 'meta': meta})

        return crops_info

    def run(
        self,
        save_dir,
        display: int = 0,
        save_individual: bool = False,
        save_summary: bool = True,
        file_prefix: str = "",
        pad: int = 3,
        cells_per_quad: int = 1,
        spots_per_cell: int = 20,
        spotChannel: int = 0
    ):
        """
        Main entry:
        - load & project spotChannel
        - select cells
        - crop spots using 5-slice Z max projection
        - optional per‐spot saving
        - optional summary saving
        Returns:
        crops, meta  (lists of patches and dicts)
        """
        os.makedirs(save_dir, exist_ok=True)
        all_crops, all_meta = [], []

        for h5_idx in tqdm(sorted(self.spots.h5_idx.unique()), desc="h5_idx"):
            imgs, msks, _, fov = self.get_images_and_masks(h5_idx=h5_idx)
            raw_3d = imgs[spotChannel]  # (Z, Y, X)
            proj_2d = np.max(raw_3d, axis=0)
            p1, p99 = np.percentile(proj_2d, (1, 99))
            proj_2d = exposure.rescale_intensity(proj_2d, in_range=(p1, p99), out_range=(0, 1))

            # Use projected image shape to select cells
            cells = self._select_cells(h5_idx, fov, proj_2d.shape)
            for cid in cells['unique_cell_id']:
                ci = self._crop_spots_in_cell(
                    raw_3d, h5_idx, fov, cid,
                    display=display,
                    save=save_individual,
                    save_dir=save_dir,
                    pad=pad,
                    max_spots=spots_per_cell
                )
                for e in ci:
                    all_crops.append(e['crop'])
                    all_meta .append(e['meta'])

        # summary save
        if save_summary:
            crops_np = np.stack(all_crops)
            name_crops = f"{file_prefix}_all_crops.npy" if file_prefix else "all_crops.npy"
            np.save(os.path.join(save_dir, name_crops), crops_np)

            meta_df = pd.DataFrame(all_meta)
            name_meta = f"{file_prefix}_all_crop_metadata.csv" if file_prefix else "all_crop_metadata.csv"
            meta_df.to_csv(os.path.join(save_dir, name_meta), index=False)

        return all_crops, all_meta