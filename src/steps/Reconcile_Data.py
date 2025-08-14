import numpy as np
import pandas as pd
import os

from src import load_data

def main(
        receipt, 
        step_name, 
        new_params:dict = None, 
        p_range = None, 
        t_range = None,
        run_in_parallel:bool = False
        ):
    data = load_data(receipt)

    # required updates to the receipt
    if step_name not in receipt['step_order']:
        receipt['step_order'].append(step_name)
    if step_name not in receipt['steps'].keys():
        receipt['steps'][step_name] = {}
    receipt['steps'][step_name]['task_name'] = 'reconcile_data'
    if new_params:
        for k, v in new_params.items():
            receipt['steps'][step_name][k] = v

    # get data
    spot_key = receipt['steps'][step_name]['spot_key']
    spot_df = data[spot_key]

    cell_key = receipt['steps'][step_name]['cell_key']
    cell_df = data[cell_key]

    # use data to make new dataframe
    # Ensure nuc_label matches cell_label where both exist
    if 'nuc_label' in spot_df.columns and 'cell_label' in spot_df.columns:
        mask = spot_df['nuc_label'] > 0
        assert (spot_df.loc[mask, 'nuc_label'] == spot_df.loc[mask, 'cell_label']).all()

    # Count total spots per cell
    spot_counts = spot_df.groupby(['timepoint', 'fov', 'cell_label']).size().reset_index(name='nb_rna')

    # Count spots inside nucleus per cell
    if 'is_nuc' in spot_df.columns:
        spot_df['is_nuc'] = spot_df['is_nuc'].fillna(False).astype(bool)
        in_nuc_counts = spot_df[spot_df['is_nuc']].groupby(['timepoint', 'fov', 'cell_label']).size().reset_index(name='nb_rna_in_nuc')
        out_nuc_counts = spot_df[~spot_df['is_nuc']].groupby(['timepoint', 'fov', 'cell_label']).size().reset_index(name='nb_rna_out_nuc')
    else:
        in_nuc_counts = spot_counts[['timepoint', 'fov', 'cell_label']].copy()
        in_nuc_counts['nb_rna_in_nuc'] = 0
        out_nuc_counts = spot_counts[['timepoint', 'fov', 'cell_label']].copy()
        out_nuc_counts['nb_rna_out_nuc'] = 0

    # Merge counts into cell_df
    label = 'cell_label' if 'cell_label' in cell_df.columns else 'nuc_label'
    master_df = cell_df.merge(spot_counts, how='left', left_on=['timepoint', 'fov', label], right_on=['timepoint', 'fov', 'cell_label'])
    master_df = master_df.merge(in_nuc_counts, how='left', on=['timepoint', 'fov', 'cell_label'])
    master_df = master_df.merge(out_nuc_counts, how='left', on=['timepoint', 'fov', 'cell_label'])

    # Fill NaNs with zeros
    master_df[['nb_rna', 'nb_rna_in_nuc', 'nb_rna_out_nuc']] = master_df[['nb_rna', 'nb_rna_in_nuc', 'nb_rna_out_nuc']].fillna(0).astype(int)

    # save dataframe
    save_dir = receipt['dirs']['results_dir']
    master_df.to_csv(os.path.join(save_dir, 'cellresults_reconciled_data.csv'), index=False)

    return receipt


if __name__ == '__main__':
    from src import Receipt, load_data
    path = r'C:\Users\Jack\Documents\GitHub\AngelFISH\Publications\IntronDiffusion\FirstAttempt.json'
    receipt = Receipt(path=path)
    main(receipt, 'reconcile_data', {'spot_key': 'introns_spotresults_filtered', 'cell_key': 'cellproperties'})

