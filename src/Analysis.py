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


class AnalysisManager:
    """
    This class is made to select data for further analysis.
    It provides methods to load, filter, and retrieve datasets from HDF5 files.
    """
    def __init__(self, location:Union[str, list[str]]=None, log_location:str=None):
        # given:
        # h5 locations
        #   give me a location
        #   give me a list of locations
        #   give me none -> got to here and display these \\munsky-nas.engr.colostate.edu\share\Users\Jack\All_Analysis
        if location is None: # TODO make these if statement better
            self.select_from_list(log_location)
        elif isinstance(location, str):
            self.location = [location]
        elif isinstance(location, list): # TODO make sure its a list of str
            self.location = location
        else:
            raise ValueError('Location is not properly defined')
        
        self._load_in_h5()
        
    def select_from_list(self, log_location) -> list[str]: 
        # log_location = r'Y:\Users\Jack\All_Analysis' # TODO: make this work for all users
        # get the log files 
        log_files = os.listdir(log_location)

        # read the log files and spit on ' -> '
        self.location = []
        for l in log_files:
            with open(os.path.join(log_location, l), 'r') as file:
                content = file.read()
            first, second = content.split(' -> ')
            name = first.split(r'/')[-1]
            drive = os.path.splitdrive(log_location)[0] + os.sep
            second = os.path.join(*second.split('/')).strip()
            location = os.path.join(drive, second, name)
            # print(location)
            self.location.append(location)
        
    def select_analysis(self, analysis_name: str = None, date_range: list[str] = None):
        self._find_analysis_names()
        self._filter_on_date(date_range)
        self._filter_on_name(analysis_name)
        self._find_analysis()
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
                    print('missing datasets')
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

    
    def get_images_and_masks(self):
        self.raw_images = [da.from_array(h['raw_images']) for h in self.h5_files]
        self.masks = [da.from_array(h['masks']) for h in self.h5_files]
        return self.raw_images, self.masks
    
    def close(self):
        for h in self.h5_files:
            h.close()

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


# Analysis children
class SpotDetection_Confirmation(Analysis):
    def __init__(self, am, seed = None):
        super().__init__(am, seed)
        self.fov = None
        self.h5_idx = None
        self.cell_label = None
    
    def get_data(self):
        h = self.am.h5_files
        d = self.am.select_datasets('spotresults')
        d1 = self.am.select_datasets('cellresults')
        d2 = self.am.select_datasets('cell_properties')
        d3 = self.am.select_datasets('clusterresults')

        self.spots = []
        self.clusters = []
        self.cellprops = []
        self.cellspots = []
        for i, s in enumerate(h):
            self.spots.append(pd.read_hdf(s.filename, d[i].name))
            self.spots[i]['h5_idx'] = [i]*len(self.spots[i])
            try:
                self.clusters.append(pd.read_hdf(s.filename, d3[i].name))
                self.clusters[-1]['h5_idx'] = [i]*len(self.clusters[-1])
            except AttributeError:
                pass
            self.cellprops.append(pd.read_hdf(s.filename, d2[i].name))
            self.cellprops[i]['h5_idx'] = [i]*len(self.cellprops[i])
            self.cellspots.append(pd.read_hdf(s.filename, d1[i].name))
            self.cellspots[i]['h5_idx'] = [i]*len(self.cellspots[i])

        self.spots = pd.concat(self.spots, axis=0)
        self.clusters = pd.concat(self.clusters, axis=0)
        self.cellprops = pd.concat(self.cellprops, axis=0)
        self.cellspots = pd.concat(self.cellspots, axis=0)
        self.images, self.masks = self.am.get_images_and_masks()

    def save_data(self, location):
        self.spots.to_csv(location, index=False)
        self.clusters.to_csv(location, index=False)
        self.cellprops.to_csv(location, index=False)
        self.cellspots.to_csv(location, index=False)


    def display(self, newFOV:bool=False, newCell:bool=False, 
                spotChannel:int=0, cytoChannel:int=1, nucChannel:int=2):
        if self.fov is None or newFOV:
            # select a random self.fov, then display it
            self.h5_idx = np.random.choice(self.spots['h5_idx'])
            self.fov = np.random.choice(self.spots[self.spots['h5_idx'] == self.h5_idx]['fov'])
            tmp_spot = self.images[self.h5_idx][self.fov, 0, spotChannel, :, :, :]
            tmp_nuc = self.images[self.h5_idx][self.fov, 0, nucChannel, :, :, :]
            tmp_cyto = self.images[self.h5_idx][self.fov, 0, cytoChannel, :, :, :]

            tmp_nucmask = self.masks[self.h5_idx][self.fov, 0, nucChannel, :, :, :]
            tmp_cellmask = self.masks[self.h5_idx][self.fov, 0, cytoChannel, :, :, :]

        if self.cell_label is None or newCell:
            self.cell_label = np.random.choice(np.unique(self.cellprops[(self.cellprops['fov'] == self.fov) &  
                                                            (self.cellprops['h5_idx'] == self.h5_idx) & 
                                                            (self.cellprops['cell_label'] != 0)]['cell_label']))
        
        fovSpots = self.spots[(self.spots['fov'] == self.fov) & 
                    (self.spots['h5_idx'] == self.h5_idx)]

        cellSpots = fovSpots[fovSpots['cell_label'] == self.cell_label]

        spotRow = cellSpots.iloc[np.random.choice(np.arange(len(cellSpots)))]

        cell = self.cellprops[(self.cellprops['fov'] == self.fov) & 
                            (self.cellprops['h5_idx'] == self.h5_idx) &
                            (self.cellprops['cell_label'] == self.cell_label)]


        # Plot spots
        fig, axs = plt.subplots(1, 3)
        axs[0].axis('off')
        axs[0].set_title('Total FOV')
        axs[1].axis('off')
        axs[1].set_title('Zoom in on cell')
        axs[2].axis('off')
        axs[2].set_title('Zoom in on spot')


        # display FOV and masks. 
        # nuc mask less transparent
        # cell mask more transparent
        axs[0].imshow(np.max(tmp_spot, axis=0), vmin=np.min(tmp_spot), vmax=np.max(tmp_spot))
        axs[0].imshow(np.max(tmp_nucmask, axis=0), alpha=0.2, cmap='jet')
        axs[0].imshow(np.max(tmp_cellmask, axis=0), alpha=0.1, cmap='jet')

        # outline the selected cell
        cell_mask = tmp_cellmask == self.cell_label
        contours = find_contours(np.max(cell_mask.compute(), axis=0), 0.5)
        for contour in contours:
            axs[0].plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')

        # put red arrows on all spots in fov 
        # put blue arrows on selected spot
        for _, spot in fovSpots.iterrows():
            axs[0].arrow(spot['x_px'] + 2, spot['y_px'] + 2, 0, 0, color='red', head_width=5)

        axs[0].arrow(spotRow['x_px'] + 2, spotRow['y_px'] + 2, 0, 0, color='blue', head_width=5)

        # zoom in on the cells
        try:
            row_min = int(cell['cell_bbox-0'])
            col_min = int(cell['cell_bbox-1'])
            row_max = int(cell['cell_bbox-2'])
            col_max = int(cell['cell_bbox-3'])

            axs[1].imshow(np.max(tmp_spot, axis=0)[row_min:row_max, col_min:col_max])
            axs[1].imshow(np.max(tmp_nucmask, axis=0)[row_min:row_max, col_min:col_max], alpha=0.2, cmap='jet')
            axs[1].imshow(np.max(tmp_cellmask, axis=0)[row_min:row_max, col_min:col_max], alpha=0.1, cmap='jet')

            # put arrows on the spots within this fov
            for _, spot in cellSpots.iterrows():
                if row_min <= spot['y_px'] < row_max and col_min <= spot['x_px'] < col_max:
                    axs[1].arrow(spot['x_px'] - col_min + 2, spot['y_px'] - row_min + 2, 0, 0, color='red', head_width=5)

            axs[1].arrow(spotRow['x_px'] - col_min + 2, spotRow['y_px'] - row_min + 2, 0, 0, color='blue', head_width=5)
        except:
            print(f'self.fov {self.fov}, h5 {self.h5_idx}, cell_label {self.cell_label} failed')

        # zoom in further on the spot
        try:
            print(f'number of selected spots: {len(spotRow)}')
            spot_row_min = max(int(spotRow['y_px']) - 15, 0)
            spot_row_max = min(int(spotRow['y_px']) + 15, tmp_spot.shape[1])
            spot_col_min = max(int(spotRow['x_px']) - 15, 0)
            spot_col_max = min(int(spotRow['x_px']) + 15, tmp_spot.shape[2])

            axs[2].arrow(spotRow['x_px'] - spot_col_min + 2, spotRow['y_px'] - spot_row_min + 2, 0, 0, color='blue', head_width=2)
            axs[2].imshow(np.max(tmp_spot, axis=0)[spot_row_min:spot_row_max, spot_col_min:spot_col_max])
            axs[2].imshow(np.max(tmp_nucmask, axis=0)[spot_row_min:spot_row_max, spot_col_min:spot_col_max], alpha=0.4, cmap='jet')
            axs[2].imshow(np.max(tmp_cellmask, axis=0)[spot_row_min:spot_row_max, spot_col_min:spot_col_max], alpha=0.2, cmap='jet')

        except:
            print(f'Zoom in on spot failed for self.fov {self.fov}, h5 {self.h5_idx}, cell_label {self.cell_label}')

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



class GR_Confirmation(Analysis):
    """
    GR_Confirmation is a class designed to confirm the accuracy of image processing results 
    for ICC of GR (Glucocorticoid Receptor) performed by Eric Rons. This class handles two 
    channels: one for the nucleus and one for the GR. Cell masks are generated by dilating 
    the nuclear mask, which is initially generated by Cellpose. A illumination pattern correction
    was applied to even out the images.

    Attributes:
    -----------
    am : object
        An instance of the AnalysisManager class that handles the dataset and image operations.
    seed : int, optional
        A seed value for random number generation to ensure reproducibility (default is None).
    cellprops : list
        A list to store cell properties extracted from the datasets.
    illumination_profiles : array
        An array to store illumination profiles for image correction.
    images : array
        An array to store images from the datasets.
    masks : array
        An array to store masks from the datasets.
    Methods:
    --------
    __init__(self, am, seed=None):
        Initializes the GR_Confirmation class with the given AnalysisManager instance and seed value.
    get_data(self):
        Retrieves and processes the necessary data from the datasets, including cell properties, 
        illumination profiles, images, and masks.
    save_data(self, location):
        Saves the cell properties data to a CSV file at the specified location.
    display(self, GR_Channel=0, Nuc_Channel=1):
        Displays a random field of view (FOV) with the GR and nucleus channels, including 
        illumination correction and cell mask overlays. Also displays a randomly selected cell 
        with its bounding box and measurements.
    validate_measurements(self, cell_mask, nuc_mask, image, label, measurements):
        Validates the calculated measurements (area, intensity, etc.) of the cell and nucleus 
        against the provided measurements and returns a DataFrame with the comparison results.
    """
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
        self.images, self.masks = self.am.get_images_and_masks()



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
        # correction_profiles = self.illumination_profiles[GR_Channel]
        temp_img *= correction_profiles[np.newaxis, :, :]
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
                                       np.max(self.images[h5_idx][fov, 0, GR_Channel, :, :, :].compute()*correction_profiles[np.newaxis, :, :], axis=0), 
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
    ana = AnalysisManager(r'\\munsky-nas.engr.colostate.edu\share\smFISH_images\Eric_smFISH_images\20220225\DUSP1_Dex_0min_20220224\DUSP1_Dex_0min_20220224.h5')
    print(ana.location)
    print(ana.h5_files)
    ana.list_analysis_names()
    ana.select_analysis()
    ana.list_datasets()
    print(ana.select_datasets('df_spotresults'))


