import h5py
import matplotlib.pyplot as plt
import dask.array as da
import numpy as np
import dask.dataframe as dp
import pandas as pd
from typing import Union
from abc import ABC, abstractmethod

"""
self: for each instance of this class has its own self.
    contains:
        methods
        variables
"""

class Analysis_Manager:
    def __init__(self, location: Union[str, list[str]] = None):
        # given:
        # h5 locations
        #   give me a location
        #   give me a list of locations
        #   give me none -> got to here and display these \\munsky-nas.engr.colostate.edu\share\Users\Jack\All_Analysis
        if location is None:
            self.select_from_list()
        elif isinstance(location, str):
            self.location = [location]
        elif isinstance(location, list): # TODO make sure its a list of str
            self.location = location
        else:
            raise ValueError('Location is not properly defined')
        
        self._load_in_h5()
        
    def select_from_list(self) -> list[str]: # TODO: this requires user input
        pass

    def select_analysis(self, analysis_name: str = None, date_range: list[str] = None):
        self._find_analysis_names()
        self._filter_on_date(date_range)
        self._filter_on_name(analysis_name)
        self._find_analysis()
        self._handle_duplicates()

    def list_analysis_names(self):
        self._find_analysis_names()
        for name in self.analysis_names:
            print(name)
        return self.analysis_names

    def select_datasets(self, dataset_name) -> list:
        if hasattr(self, 'analysis'):
            self.datasets = [h[dataset_name] for h in self.analysis]
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
        temp_h5_files = self.h5_files
        for h_idx, h in enumerate(temp_h5_files):
            for dataset_name in self.analysis_names:
                for date in self.dates:
                    if f'Analysis_{dataset_name}_{date}' in list(h.keys()):
                        self.analysis.append(h[f'Analysis_{dataset_name}_{date}'])
                    else:
                        self.h5_files.pop(h_idx)

                        

    def _handle_duplicates(self): # requires user input
        pass

    def _find_analysis_names(self):
        self.analysis_names = []
        for h in self.h5_files:
            self.analysis_names.append(list(h.keys()))
        self.analysis_names = set([dataset for sublist in self.analysis_names for dataset in sublist])
        self.analysis_names = [d for d in self.analysis_names if 'Analysis' in d]

    def _load_in_h5(self):
        self.h5_files = []
        for l in self.location:
            self.h5_files.append(h5py.File(l, 'r'))

        self.raw_images = [da.from_array(h['raw_images']) for h in self.h5_files]
        self.masks = [da.from_array(h['masks']) for h in self.h5_files]

#%%
class Analysis(ABC):
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

    # @abstractmethod
    # def select_data(self, identifying_feature):
    #     pass

class SpotDetection_Confirmation(Analysis):
    def __init__(self, am, seed = None):
        super().__init__(am, seed)
    
    def get_data(self):
        self.spots = self.am.select_datasets('df_spotresults')
        for i, s in enumerate(self.spots):
            self.spots[i] = pd.read_hdf(s)
            self.spots[i]['h5_idx'] = [i]*len(self.spots[i])
        self.spots = pd.concat(self.spots, axis=0)
        self.images = self.am.raw_images
        self.masks = self.am.masks

    def save_data(self, location):
        self.spots.to_csv(location, index=False)

    def display(self):
        spot = self.spots.sample(n=1, random_state=self.seed).iloc[0]
        h5_idx = spot['h5_idx']
        image = self.images[h5_idx]
        mask = self.masks[h5_idx]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # axs[0] - Plot the show image with arrows pointing at all the spots, with the selected spot being a different color
        axs[0].imshow(image, cmap='gray')
        for _, s in self.spots.iterrows():
            color = 'red' if s.equals(spot) else 'blue'
            axs[0].arrow(s['x'], s['y'], 0, 0, color=color, head_width=5)
        axs[0].set_title('All spots with selected spot highlighted')

        # axs[1] - Select the cell with the spot in it and show a bound box around that cell
        cell_mask = mask == mask[int(spot['y']), int(spot['x'])]
        axs[1].imshow(image, cmap='gray')
        axs[1].imshow(cell_mask, cmap='jet', alpha=0.5)
        axs[1].set_title('Cell containing the selected spot')

        # axs[2] - Zoom in further on the spot and show a nxn box around the spot
        n = 20
        x, y = int(spot['x']), int(spot['y'])
        zoomed_image = image[max(0, y-n):y+n, max(0, x-n):x+n]
        axs[2].imshow(zoomed_image, cmap='gray')
        axs[2].set_title('Zoomed in on selected spot')

        plt.show()


class GR_Confirmation(Analysis):
    def __init__(self, am, seed = None):
        super().__init__(am, seed)

    def get_data(self):
        h = self.am.h5_files
        d = self.am.select_datasets('cell_properties')
        self.cellprops = []
        for i, s in enumerate(h):
            self.cellprops.append(pd.read_hdf(s.filename, d[i].name))
            self.cellprops[i]['h5_idx'] = [i]*len(self.cellprops[i])
        self.cellprops = pd.concat(self.cellprops, axis=0)
        self.illumination_profiles = da.from_array(self.am.select_datasets('illumination_profiles'))[0, :, : ,:]
        self.images = self.am.raw_images
        self.masks = self.am.masks


    def save_data(self, location):
        self.cellprops.to_csv(location, index=False)

    def display(self, GR_Channel:int = 0, Nuc_Channel:int = 1):
        # select a random fov, then display it
        h5_idx = np.random.choice(self.cellprops['h5_idx'])
        fov = np.random.choice(self.cellprops[self.cellprops['h5_idx'] == h5_idx]['fov'])
        temp_img = self.images[h5_idx][fov, 0, GR_Channel, :, :, :]
        temp_mask = self.masks[h5_idx][fov, 0, GR_Channel, :, :, :]

        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(np.max(temp_img, axis=0))

        # display the illumination profile
        axs[1].imshow(self.illumination_profiles[GR_Channel])

        # use the illumination profile to correct it, then display it
        epsilon = 1e-6
        correction_profiles = 1.0 / (self.illumination_profiles[GR_Channel] + epsilon)
        temp_img *= correction_profiles[None, :, :]
        axs[2].imshow(np.max(temp_img, axis=0))
        plt.show()

        # select a random cell in the fov and display it using the bonded boxs, also include measurments
        # include a transparent mask over the random cell
        fig, axs = plt.subplots(1, 2)

        cell_label = np.random.choice(np.unique(self.cellprops[(self.cellprops['fov'] == fov) &  
                                                                (self.cellprops['h5_idx'] == h5_idx)]['nuc_label']))
        row = self.cellprops[(self.cellprops['fov'] == fov) & 
                             (self.cellprops['nuc_label'] == cell_label) & 
                             (self.cellprops['h5_idx'] == h5_idx)]

        axs[0].imshow(np.max(temp_img, axis=0))
        axs[0].imshow(np.max(temp_mask, axis=0), alpha=0.4, cmap='jet')
        cell_mask = temp_mask == cell_label
        cell_center = np.array(np.nonzero(np.max(cell_mask, axis=0))).mean(axis=1)
        axs[0].arrow(cell_center[1], cell_center[0], 0, 0, color='red', head_width=10)
        # axs[1].set_title('Randomly selected cell with arrow')

        try:
            row_min = int(row['cell_bbox-0'])
            col_min = int(row['cell_bbox-1'])
            row_max = int(row['cell_bbox-2'])
            col_max = int(row['cell_bbox-3'])

            axs[1].imshow(np.max(temp_img, axis=0)[row_min:row_max, col_min:col_max])
            axs[1].imshow(np.max(temp_mask, axis=0)[row_min:row_max, col_min:col_max], alpha=0.25, cmap='jet')
        except:
            print(f'fov {fov}, h5 {h5_idx}, nuc_label {cell_label} failed')

        plt.show()

        # Nuc Mask
        temp_mask = self.masks[h5_idx][fov, 0, Nuc_Channel, :, :, :]
        temp_img = self.images[h5_idx][fov, 0, Nuc_Channel, :, :, :]

        fig, axs = plt.subplots(1, 2)

        axs[0].imshow(np.max(temp_img, axis=0))
        axs[0].imshow(np.max(temp_mask, axis=0), alpha=0.4, cmap='jet')
        cell_mask = temp_mask == cell_label
        cell_center = np.array(np.nonzero(np.max(cell_mask, axis=0))).mean(axis=1)
        axs[0].arrow(cell_center[1], cell_center[0], 0, 0, color='red', head_width=10)
        # axs[1].set_title('Randomly selected cell with arrow')

        try:
            row_min = int(row['cell_bbox-0'])
            col_min = int(row['cell_bbox-1'])
            row_max = int(row['cell_bbox-2'])
            col_max = int(row['cell_bbox-3'])

            axs[1].imshow(np.max(temp_img, axis=0)[row_min:row_max, col_min:col_max])
            axs[1].imshow(np.max(temp_mask, axis=0)[row_min:row_max, col_min:col_max], alpha=0.25, cmap='jet')
        except:
            print(f'fov {fov}, h5 {h5_idx}, nuc_label {cell_label} failed')

        plt.show()


        print(
            self.validate_measurements(np.max(self.masks[h5_idx][fov, 0, GR_Channel, :, :, :], axis=0), 
                                       np.max(self.masks[h5_idx][fov, 0, Nuc_Channel, :, :, :], axis=0), 
                                       np.max(self.images[h5_idx][fov, 0, GR_Channel, :, :, :], axis=0), 
                                       cell_label, row)
              )

    def validate_measurements(self, cell_mask, nuc_mask, image, label, measurements):
        # calculate cell area
        cell_area = np.sum(cell_mask == label).compute()
        nuc_area = np.sum(nuc_mask == label).compute()

        # calculate average intensity
        cell_avgInt = np.mean(image[cell_mask == label]).compute()
        nuc_avgInt = np.mean(image[nuc_mask == label]).compute()

        # calculate std intensity
        cell_stdInt = np.std(image[cell_mask == label]).compute()
        nuc_stdInt = np.std(image[nuc_mask == label]).compute()

        # calculate max intensity
        cell_maxInt = np.max(image[cell_mask == label]).compute()
        nuc_maxInt = np.max(image[nuc_mask == label]).compute()

        # calculate min intensity
        cell_minInt = np.min(image[cell_mask == label]).compute()
        nuc_minInt = np.min(image[nuc_mask == label]).compute()

        # compare
        r = []
        r.append(
            ['cell_area', cell_area, float(measurements['cell_area']), np.isclose(cell_area, float(measurements['cell_area']))]
        )

        r.append(
            ['nuc_area', nuc_area, float(measurements['nuc_area']), np.isclose(nuc_area, float(measurements['nuc_area']))]
        )

        r.append(
            ['cyto_area', cyto_area := cell_area - nuc_area, float(measurements['cyto_area']), np.isclose(cyto_area, float(measurements['cyto_area']))]
        )

        r.append(
            ['cell_avgInt', cell_avgInt, float(measurements['cell_intensity_mean-0']), np.isclose(cell_avgInt, float(measurements['cell_intensity_mean-0']))]
        )

        r.append(
            ['nuc_avgInt', nuc_avgInt, float(measurements['nuc_intensity_mean-0']), np.isclose(nuc_avgInt, float(measurements['nuc_intensity_mean-0']))]
        )

        r.append(
            ['cell_stdInt', cell_stdInt, float(measurements['cell_intensity_std-0']), np.isclose(cell_stdInt, float(measurements['cell_intensity_std-0']))]
        )

        r.append(
            ['nuc_stdInt', nuc_stdInt, float(measurements['nuc_intensity_std-0']), np.isclose(nuc_stdInt, float(measurements['nuc_intensity_std-0']))]
        )

        r.append(
            ['cell_maxInt', cell_maxInt, float(measurements['cell_intensity_max-0']), np.isclose(cell_maxInt, float(measurements['cell_intensity_max-0']))]
        )

        r.append(
            ['nuc_maxInt', nuc_maxInt, float(measurements['nuc_intensity_max-0']), np.isclose(nuc_maxInt, float(measurements['nuc_intensity_max-0']))]
        )

        r.append(
            ['cell_minInt', cell_minInt, float(measurements['cell_intensity_min-0']), np.isclose(cell_minInt, float(measurements['cell_intensity_min-0']))]
        )

        r.append(
            ['nuc_minInt', nuc_minInt, float(measurements['nuc_intensity_min-0']), np.isclose(nuc_minInt, float(measurements['nuc_intensity_min-0']))]
        )

        results = pd.DataFrame(r, columns=['measurement', 'numpy calculated', 'step calculated', 'are close?'])
        return results






if __name__ == '__main__':
    ana = Analysis_Manager(r'\\munsky-nas.engr.colostate.edu\share\smFISH_images\Eric_smFISH_images\20220225\DUSP1_Dex_0min_20220224\DUSP1_Dex_0min_20220224.h5')
    print(ana.location)
    print(ana.h5_files)
    ana.list_analysis_names()
    ana.select_analysis()
    ana.list_datasets()
    print(ana.select_datasets('df_spotresults'))


