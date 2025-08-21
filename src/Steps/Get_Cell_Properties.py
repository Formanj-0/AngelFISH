import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import skimage as sk
from copy import copy



from AngelFISH.src import abstract_task, load_data
import re


class get_cell_properties(abstract_task):
    @classmethod
    def task_name(cls):
        return 'get_cell_properties'

    def extract_args(self, p, t):
        given_args = self.receipt['steps'][self.step_name]
        nuc_mask_name = self.receipt['steps'][self.step_name].get('nuc_mask_name', 'nuc_masks')
        cyto_mask_name = self.receipt['steps'][self.step_name].get('cyto_mask_name', 'cyto_masks')


        data_to_send = {}
        data_to_send['image'] = self.data['images'][p, t].compute()
        data_to_send['metadata'] = self.data['metadata']
        data_to_send['sharpnesses'] = self.data.get('sharpnesses', None)

        nuc_masks = self.data.get(nuc_mask_name, None)
        if nuc_masks is not None:
            data_to_send['nuc_mask'] = nuc_masks[p,t].compute()

        cyto_masks = self.data.get(cyto_mask_name, None)
        if cyto_masks is not None:
            data_to_send['cyto_mask'] = cyto_masks[p,t].compute()

        args = {**data_to_send, **given_args}

        args['fov'] = p
        args['timepoint'] = t

        return args

    def preallocate_memmory(self):
        self.temp_dir = os.path.join(self.receipt['dirs']['analysis_dir'], 'cell_properties')
        os.makedirs(self.temp_dir, exist_ok=True)

    @staticmethod
    def image_processing_function(
            image, 
            fov, 
            timepoint, 
            cell_mask=None, 
            nuc_mask=None, 
            middle_zs:int = None,
            props_to_measure= ['label', 'bbox', 'area', 'centroid', 'intensity_max', 'intensity_mean', 'intensity_min', 'intensity_std'], 
            sharpnesses:dict=None, 
            sharpness_metric:str=None,
            metadata:callable=None,
            **kwargs):
        
        if middle_zs is not None and isinstance(middle_zs, int) and image.ndim >= 3:
            if sharpness_metric is not None:
                z_sharpnesses = [sharpnesses[f'{fov}'][f'{timepoint}'][k][sharpness_metric] for k in sharpnesses[f'{fov}'][f'{timepoint}'].keys()]
            z_dim = image.shape[1]
            if sharpness_metric is not None and z_sharpnesses:
                # Find the z index with the maximum sharpness
                max_z = np.argmax(z_sharpnesses)
                half = middle_zs // 2
                start = max(0, min(z_dim - middle_zs, max_z - half))
                end = start + middle_zs
                image = np.max(image[:, start:end, ...], axis=1) # This will always include middle_zs slices
            else:
                if middle_zs > z_dim:
                    middle_zs = z_dim
                start = (z_dim - middle_zs) // 2
                end = start + middle_zs
                image = np.max(image[:, start:end, ...], axis=1)
        else:
            image = np.max(image, axis=1)

        image = np.moveaxis(image, 0, -1) # move channel axis to the back
    
        if nuc_mask is not None and len(nuc_mask.shape) == 3:
            nuc_mask = np.max(nuc_mask, axis=0)

        if cell_mask is not None and len(nuc_mask.shape) == 3:
            nuc_mask = np.max(nuc_mask, axis=0)

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

        combined_df = pd.concat([nuc_df, cell_df, cyto_df], axis=1)
        combined_df['fov'] = [fov]*len(combined_df)
        combined_df['timepoint'] = [timepoint]*len(combined_df)
        combined_df['touching_border'] = touching_border(combined_df, image)

        expermental_metadata = metadata(p=fov, t=timepoint, z=0 ,c=0).get('experimental_metadata', None)
        if expermental_metadata is not None:
            for key, value in expermental_metadata.items():
                combined_df[key] = [value] * len(combined_df)


        return {'cell_properties': combined_df}

    def write_results(self, results, p, t):
        if results['cell_properties'] is not None:
            cell_path = os.path.join(self.temp_dir, f'p{p}_t{t}_cellprops.csv')
            results['cell_properties'].to_csv(cell_path, index=False)

    def compress_and_release_memory(self):
        results_dir = self.receipt['dirs']['results_dir']

        # List all files in the temp_dir
        all_files = os.listdir(self.temp_dir)

        pattern = re.compile(r'.*_cellprops\.csv$')

        files = [os.path.join(self.temp_dir, f) for f in all_files if pattern.match(f)]

        final_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None

        if final_df is not None:
            final_df.to_csv(os.path.join(results_dir, f'cellproperties.csv'), index=False)

        # Delete all files in the temp_dir
        for f in all_files:
            file_path = os.path.join(self.temp_dir, f)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

        # Delete the temp_dir itself
        try:
            os.rmdir(self.temp_dir)
        except Exception as e:
            print(f"Error deleting directory {self.temp_dir}: {e}")

    @property
    def required_keys(self):
        return []






































