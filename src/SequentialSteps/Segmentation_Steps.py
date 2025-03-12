import numpy as np
import pathlib
import os
import tifffile
import matplotlib.pyplot as plt
from cellpose import models
from skimage.io import imread
import skimage as sk
import bigfish
import bigfish.stack as stack
# import bigfish.segmentation as segmentation
import bigfish.multistack as multistack
import bigfish.plot as plot
import dask.array as da
from abc import ABC, abstractmethod
import tifffile
import pandas as pd
from scipy import ndimage as ndi
from copy import copy

from src import SequentialStepsClass

from src.Parameters import Parameters
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt



#%% Abstract Class
class CellSegmentation(SequentialStepsClass):
    def main(self, masks, image, fov, timepoint, nucChannel, cytoChannel, 
             display_plots, do_3D_Segmentation, **kwargs):
        if len(image.shape) == 3:
            image = np.max(image, axis=0) 
        
        if timepoint == 0:
            nuc_mask = self.segment_nuclei(**kwargs)

            cell_mask = self.segment_cells(**kwargs)

            nuc_mask, cell_mask = self.align_nuc_cell_masks(nuc_mask, cell_mask, **kwargs)

            self.plot_segmentation(display_plots, image, nuc_mask, cell_mask, nucChannel, cytoChannel, do_3D_Segmentation)

            return {'cell_mask':cell_mask, 'nuc_mask':nuc_mask}

    def segment_nuclei(self, **kwargs):
        pass

    def segment_cells(self, **kwargs):
        pass

    def align_nuc_cell_masks(self, nuc_mask, cell_mask):
        if nuc_mask is not None and cell_mask is not None:
            nuc_mask, cell_mask = multistack.match_nuc_cell(nuc_mask, cell_mask, single_nuc=True, cell_alone=False)
        return nuc_mask, cell_mask

    def plot_segmentation(self, display_plots, image, nuc_mask, cell_mask, nuc_channel, cyto_channel, do_3D_Segmentation):
        if display_plots:
            num_sub_plots = 0
            if nuc_mask is not None:
                num_sub_plots += 2
            if cell_mask is not None:
                num_sub_plots += 2
            fig, axs = plt.subplots(1, num_sub_plots, figsize=(12, 5))
            i = 0
            if nuc_mask is not None:
                if do_3D_Segmentation:
                    axs[i].imshow(np.max(image,axis=0)[nuc_channel, :, :])
                    axs[i].set_title('Nuclei')
                    i += 1
                    axs[i].imshow(np.max(nuc_mask,axis=0))
                    axs[i].set_title('Nuclei Segmentation, NC: ' + str(np.max(nuc_mask)))
                    i += 1
                else:
                    axs[i].imshow(image[nuc_channel,:,:])
                    axs[i].set_title('Nuclei')
                    i += 1
                    axs[i].imshow(nuc_mask)
                    axs[i].set_title('Nuclei Segmentation, NC: ' + str(np.max(nuc_mask)))
                    i += 1
            if cell_mask is not None:
                if do_3D_Segmentation:
                    axs[i].imshow(np.max(image,axis=0)[cyto_channel, :, :])
                    axs[i].set_title('cell_mask')
                    i += 1
                    axs[i].imshow(np.max(cell_mask, axis=0))
                    axs[i].set_title('cell_mask Segmentation, NC: ' + str(np.max(nuc_mask)))
                    i += 1
                else:
                    axs[i].imshow(image[cyto_channel,:,:])
                    axs[i].set_title('cell_mask')
                    i += 1
                    axs[i].imshow(cell_mask)
                    axs[i].set_title('cell_mask Segmentation, NC: ' + str(np.max(nuc_mask)))
                    i += 1
            plt.tight_layout()
            plt.show()



#%% Steps
class DilationedCytoMask(SequentialStepsClass):
    def __init__(self):
        super().__init__()

    def main(self, timepoint, fov, nucChannel, psuedoCyto, masks,  nuc_mask, dilation_size: int = 20, display_plots: bool = False, **kwargs):
        cell_mask = masks[fov, timepoint, psuedoCyto, 0, :, :].squeeze().compute()

        mask = nuc_mask[0, :, :].squeeze() > 0
        nuc_mask =  nuc_mask[0, :, :].squeeze()
        nuc_mask = nuc_mask.compute()
        for i in range(dilation_size):
            mask = sk.morphology.binary_dilation(mask)
        
        # watershed 
        markers = np.zeros_like(nuc_mask, dtype=int)
        distance = ndi.distance_transform_edt(mask)
        for label in np.unique(nuc_mask):
            if label == 0:
                continue
            d = copy(distance)
            d[nuc_mask!=label] = 0
            center = np.unravel_index(np.argmax(d), distance.shape)
            markers[center] = label
        cell_mask = sk.segmentation.watershed(-distance, markers, mask=mask)

        # match nuc and cell mask
        nuc_mask, cell_mask = multistack.match_nuc_cell(nuc_mask.astype(np.uint8), cell_mask.astype(np.uint8), single_nuc=False, cell_alone=False)

        if display_plots:
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(nuc_mask)
            axs[1].imshow(cell_mask)
            # axs[2].imshow(cyto_mask)
            plt.show()

        return {'cytoChannel': psuedoCyto, 'cell_mask': cell_mask}


class SimpleCellposeSegmentaion(CellSegmentation):
    """
    A class for performing cell segmentation using the Cellpose model.
    Methods
    -------
    main(image, cytoChannel, nucChannel, masks, timepoint, fov, cellpose_model_type, cellpose_diameter, 
         cellpose_channel_axis, cellpose_invert, cellpose_normalize, do_3D_Segmentation, cellpose_min_size, 
         cellpose_flow_threshold, cellpose_cellprob_threshold, cellpose_pretrained_model, display_plots, **kwargs)
        Main method to perform segmentation on the given image.
    Parameters
    ----------
    image : ndarray
        The input image to be segmented.
    cytoChannel : int
        The channel index for cytoplasm.
    nucChannel : int
        The channel index for nuclei.
    masks : list
        List to store the segmentation masks.
    timepoint : int
        The timepoint index for the image.
    fov : int
        The field of view index for the image.
    cellpose_model_type : str or list of str, optional
        The type of Cellpose model to use. Default is ['cyto3', 'nuclei'].
    cellpose_diameter : float or list of float, optional
        The diameter of the cells to be segmented. Default is 180.
    cellpose_channel_axis : int, optional
        The axis of the channels in the image. Default is 0.
    cellpose_invert : bool or list of bool, optional
        Whether to invert the image for segmentation. Default is False.
    cellpose_normalize : bool, optional
        Whether to normalize the image for segmentation. Default is True.
    do_3D_Segmentation : bool, optional
        Whether to perform 3D segmentation. Default is False.
    cellpose_min_size : float or list of float, optional
        The minimum size of the cells to be segmented. Default is 500.
    cellpose_flow_threshold : float or list of float, optional
        The flow threshold for the Cellpose model. Default is 0.
    cellpose_cellprob_threshold : float or list of float, optional
        The cell probability threshold for the Cellpose model. Default is 0.
    cellpose_pretrained_model : str or list of str, optional
        The path to the pretrained Cellpose model. Default is False.
    display_plots : bool, optional
        Whether to display plots of the segmentation results. Default is False.
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(self):
        super().__init__()

    def main(self, 
             image, 
             cytoChannel, 
             nucChannel, 
             masks,
             timepoint: int,
             fov: int,
             cellpose_model_type: str | list[str] = ['cyto3', 'nuclei'], 
             cellpose_diameter: float | list[float] = 180, 
             cellpose_channel_axis: int = 0,
             cellpose_invert: bool | list = False, 
             cellpose_normalize: bool = True, 
             do_3D_Segmentation: bool = False, # This is not implemented
             cellpose_min_size: float | list[float] = 500, 
             cellpose_flow_threshold: float | list[float] = 0, 
             cellpose_cellprob_threshold: float | list[float] = 0,
             cellpose_pretrained_model: str | list[str] = False,
             display_plots: bool = False,
               **kwargs):
        if image.shape[1] >= 1:
            image = np.max(image, axis=1) 
        
        if timepoint == 0:
            nuc_mask = self.segment_nuclei(image, nucChannel, cellpose_min_size, cellpose_flow_threshold,
                                           cellpose_cellprob_threshold, cellpose_model_type, cellpose_diameter,
                                           cellpose_channel_axis, cellpose_invert, cellpose_normalize, do_3D_Segmentation,
                                           cellpose_pretrained_model)

            cell_mask = self.segment_cells(image, cytoChannel, nucChannel, cellpose_min_size, cellpose_flow_threshold,
                                           cellpose_cellprob_threshold, cellpose_model_type, cellpose_diameter,
                                           cellpose_channel_axis, cellpose_invert, cellpose_normalize, do_3D_Segmentation, 
                                           cellpose_pretrained_model)

            nuc_mask, cell_mask = self.align_nuc_cell_masks(nuc_mask, cell_mask)

            self.plot_segmentation(display_plots, image, nuc_mask, cell_mask, nucChannel, cytoChannel, do_3D_Segmentation)

            return {'nuc_mask': nuc_mask, 'cell_mask': cell_mask}
        
    def unpack_lists(self, cellpose_min_size, 
             cellpose_flow_threshold, 
             cellpose_cellprob_threshold,
             cellpose_model_type,
             cellpose_diameter,
             pretrained_model):
        if isinstance(cellpose_min_size, list):
            nuc_min_size = cellpose_min_size[1]
            cyto_min_size = cellpose_min_size[0]
        else:
            nuc_min_size = cellpose_min_size
            cyto_min_size = cellpose_min_size
        
        if isinstance(cellpose_flow_threshold, list):
            nuc_flow_threshold = cellpose_flow_threshold[1]
            cyto_flow_threshold = cellpose_flow_threshold[0]
        else:
            nuc_flow_threshold = cellpose_flow_threshold
            cyto_flow_threshold = cellpose_flow_threshold
        
        if isinstance(cellpose_cellprob_threshold, list):
            nuc_cellprob_threshold = cellpose_cellprob_threshold[1]
            cyto_cellprob_threshold = cellpose_cellprob_threshold[0]
        else:
            nuc_cellprob_threshold = cellpose_cellprob_threshold
            cyto_cellprob_threshold = cellpose_cellprob_threshold

        if isinstance(cellpose_model_type, list):
            nuc_model_type = cellpose_model_type[1]
            cyto_model_type = cellpose_model_type[0]
        else:
            nuc_model_type = cellpose_model_type
            cyto_model_type = cellpose_model_type

        if isinstance(cellpose_diameter, list):
            nuc_diameter = cellpose_diameter[1]
            cyto_diameter = cellpose_diameter[0]
        else:
            nuc_diameter = cellpose_diameter
            cyto_diameter = cellpose_diameter

        if isinstance(pretrained_model, list):
            nuc_pretrained_model = pretrained_model[1]
            cyto_pretrained_model = pretrained_model[0]
        else:
            nuc_pretrained_model = pretrained_model
            cyto_pretrained_model = pretrained_model

        return (nuc_min_size, cyto_min_size, nuc_flow_threshold, cyto_flow_threshold, nuc_cellprob_threshold, 
                cyto_cellprob_threshold, nuc_model_type, cyto_model_type, nuc_diameter, cyto_diameter, nuc_pretrained_model,
                cyto_pretrained_model) 

    def segment_nuclei(self, image, nucChannel, cellpose_min_size, cellpose_flow_threshold, 
                       cellpose_cellprob_threshold, cellpose_model_type, cellpose_diameter, 
                       cellpose_channel_axis, cellpose_invert, cellpose_normalize, cellpose_do_3D,
                       cellpose_pretrained_model):
        if nucChannel is not None:
            (nuc_min_size, cyto_min_size, nuc_flow_threshold, cyto_flow_threshold, 
            nuc_cellprob_threshold, cyto_cellprob_threshold, nuc_model_type, 
            cyto_model_type, nuc_diameter, cyto_diameter, 
            nuc_pretrained_model, cyto_pretrained_model) = self.unpack_lists(cellpose_min_size, 
                                                                        cellpose_flow_threshold, 
                                                                        cellpose_cellprob_threshold,
                                                                        cellpose_model_type,
                                                                        cellpose_diameter,
                                                                        cellpose_pretrained_model)
            model_location = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            nuc_pretrained_model = os.path.join(model_location, nuc_pretrained_model) if nuc_pretrained_model else None

            cp = models.CellposeModel(model_type=nuc_model_type, gpu=True, pretrained_model=nuc_pretrained_model)
            sz = models.SizeModel(cp)

            model = models.Cellpose(gpu=True)
            model.cp = cp
            model.sz = sz
            # nucmodel = models.Cellpose(model_type=nuc_model_type, gpu=True)
            # if cp is not None:
            #     nucmodel.cp = cp
            channels = [0, 0]
            nuc_image = image[nucChannel, :, :]
            nuc_mask, flows, styles, diams = model.eval(nuc_image,
                                                channels=channels, 
                                                diameter=nuc_diameter, 
                                                invert=cellpose_invert, 
                                                normalize=cellpose_normalize, 
                                                channel_axis=cellpose_channel_axis, 
                                                do_3D=cellpose_do_3D,
                                                min_size=nuc_min_size, 
                                                flow_threshold=nuc_flow_threshold, 
                                                cellprob_threshold=nuc_cellprob_threshold)
                                                # net_avg=True


            return nuc_mask
        
    def segment_cells(self, image, cytoChannel, nucChannel, cellpose_min_size, cellpose_flow_threshold,
                      cellpose_cellprob_threshold, cellpose_model_type, cellpose_diameter,
                      cellpose_channel_axis, cellpose_invert, cellpose_normalize, cellpose_do_3D,
                      cellpose_pretrained_model):
        if cytoChannel is not None:
            (nuc_min_size, cyto_min_size, nuc_flow_threshold, cyto_flow_threshold, 
            nuc_cellprob_threshold, cyto_cellprob_threshold, nuc_model_type,
            cyto_model_type, nuc_diameter, cyto_diameter, 
            nuc_pretrained_model, cyto_pretrained_model) = self.unpack_lists(cellpose_min_size, 
                                                                        cellpose_flow_threshold, 
                                                                        cellpose_cellprob_threshold,
                                                                        cellpose_model_type,
                                                                        cellpose_diameter,
                                                                        cellpose_pretrained_model)
            
            model_location = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            cyto_pretrained_model = os.path.join(model_location, cyto_pretrained_model) if cyto_pretrained_model else None


            cp = models.CellposeModel(model_type=cyto_model_type, gpu=True, pretrained_model=cyto_pretrained_model)
            sz = models.SizeModel(cp)

            model = models.Cellpose(gpu=True)
            model.cp = cp
            model.sz = sz

            channels = [0, 0]
            cyto_image = image[cytoChannel, :, :]
            cell_mask, flows, styles, diams = model.eval(cyto_image,
                                                    channels=channels, 
                                                    diameter=cyto_diameter, 
                                                    invert=cellpose_invert, 
                                                    normalize=cellpose_normalize, 
                                                    channel_axis=cellpose_channel_axis, 
                                                    do_3D=cellpose_do_3D,
                                                    min_size=cyto_min_size, 
                                                    flow_threshold=cyto_flow_threshold, 
                                                    cellprob_threshold=cyto_cellprob_threshold,)
            return cell_mask


class BIGFISH_Tensorflow_Segmentation(SequentialStepsClass):
    def __init__(self):
        super().__init__()

    def main(self, id, list_images, nucChannel, cytoChannel, segmentation_smoothness: int = 7,
             watershed_threshold:int = 500, watershed_alpha:float = 0.9, bigfish_targetsize:int = 256, verbose:bool = False, **kwargs):
        raise Exception('This code has not been implemented due to needing to change tensor flow version')
    
        # img = list_images[id]
        # nuc = img[:, :, :, nucChannel[0]]
        # cell = img[:, :, :, cytoChannel[0]]
        #
        # model_nuc = segmentation.unet_3_classes_nuc()
        #
        # nuc_label = segmentation.apply_unet_3_classes(
        #     model_nuc, nuc, target_size=bigfish_targetsize, test_time_augmentation=True)
        #
        # # plot.plot_segmentation(nuc, nuc_label, rescale=True)
        #
        # # apply watershed
        # cell_label = segmentation.cell_watershed(cell, nuc_label, threshold=watershed_threshold, alpha=watershed_alpha)
        #
        # # plot.plot_segmentation_boundary(cell, cell_label, nuc_label, contrast=True, boundary_size=4)
        #
        # nuc_label = segmentation.clean_segmentation(nuc_label, delimit_instance=True)
        # cell_label = segmentation.clean_segmentation(cell_label, smoothness=segmentation_smoothness, delimit_instance=True)
        # nuc_label, cell_label = multistack.match_nuc_cell(nuc_label, cell_label, single_nuc=False, cell_alone=True)
        #
        # plot.plot_images([nuc_label, cell_label], titles=["Labelled nuclei", "Labelled cells"])
        #
        # mask_cytosol = cell_label.copy()
        # mask_cytosol[nuc_label > 0 and cell_label > 0] = 0
        #
        # num_of_cells = np.max(nuc_label)
        #
        # output = CellSegmentationOutput(list_cell_masks=cell_label, list_nuc_masks=nuc_label,
        #                                 list_cyto_masks=mask_cytosol, segmentation_successful=1,
        #                                 number_detected_cells=num_of_cells, id=id)
        # return output

class BoxCells(SequentialStepsClass):
    def __init__(self):
        super().__init__()

    def main(self, nuc_mask, cell_mask, save_mask_location, df_spotresults: pd.DataFrame):

        for region in sk.measure.regionprops(cell_mask):
            minr, minc, maxr, maxc = region.bbox
            for i, row in df_spotresults.iterrows():
                x, y = row['x_px'], row['y_px']
                try:
                    cell_mask[0, y-1:y+2, x-1:x+2] = -1
                except:
                    pass
            cropped_cell = cell_mask[minr:maxr, minc:maxc]
            file_counter = 0
            file_path = os.path.join(save_mask_location, f'cell_crop_{file_counter}.csv')
            while os.path.exists(file_path):
                file_path = os.path.join(save_mask_location, f'cell_crop_{file_counter}.csv')
                file_counter += 1
            tifffile.imwrite(file_path, cropped_cell)

        for region in sk.measure.regionprops(nuc_mask):
            minr, minc, maxr, maxc = region.bbox
            for i, row in df_spotresults.iterrows():
                x, y = row['x_px'], row['y_px']
                try:
                    cell_mask[0, y-1:y+2, x-1:x+2] = -1
                except:
                    pass
            cropped_nuc = nuc_mask[minr:maxr, minc:maxc]
            file_counter = 0
            file_path = os.path.join(save_mask_location, f'nuc_crop_{file_counter}.csv')
            while os.path.exists(file_path):
                file_path = os.path.join(save_mask_location, f'nuc_crop_{file_counter}.csv')
                file_counter += 1
            tifffile.imwrite(file_path, cropped_nuc)
    
        os.makedirs(save_mask_location, exist_ok=True)


class GeneralCellposeSegmentation(CellSegmentation):
    """
    A class for performing cell segmentation using the Cellpose model.
    Methods
    -------
    main(image, cytoChannel, nucChannel, masks, timepoint, fov, cellpose_model_type, cellpose_diameter, 
         cellpose_channel_axis, cellpose_invert, cellpose_normalize, do_3D_Segmentation, cellpose_min_size, 
         cellpose_flow_threshold, cellpose_cellprob_threshold, cellpose_pretrained_model, display_plots, **kwargs)
        Main method to perform segmentation on the given image.
    Parameters
    ----------
    image : ndarray
        The input image to be segmented.
    cytoChannel : int
        The channel index for cytoplasm.
    nucChannel : int
        The channel index for nuclei.
    masks : list
        List to store the segmentation masks.
    timepoint : int
        The timepoint index for the image.
    fov : int
        The field of view index for the image.
    cellpose_model_type : str or list of str, optional
        The type of Cellpose model to use. Default is ['cyto3', 'nuclei'].
    cellpose_diameter : float or list of float, optional
        The diameter of the cells to be segmented. Default is 180.
    cellpose_channel_axis : int, optional
        The axis of the channels in the image. Default is 0.
    cellpose_invert : bool or list of bool, optional
        Whether to invert the image for segmentation. Default is False.
    cellpose_normalize : bool, optional
        Whether to normalize the image for segmentation. Default is True.
    do_3D_Segmentation : bool, optional
        Whether to perform 3D segmentation. Default is False.
    cellpose_min_size : float or list of float, optional
        The minimum size of the cells to be segmented. Default is 500.
    cellpose_flow_threshold : float or list of float, optional
        The flow threshold for the Cellpose model. Default is 0.
    cellpose_cellprob_threshold : float or list of float, optional
        The cell probability threshold for the Cellpose model. Default is 0.
    cellpose_pretrained_model : str or list of str, optional
        The path to the pretrained Cellpose model. Default is False.
    display_plots : bool, optional
        Whether to display plots of the segmentation results. Default is False.
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(self):
        super().__init__()

    def main(self, 
             image, 
             cytoChannel, 
             nucChannel, 
             FISHChannel,
             mask_structure: dict,
             cellpose_model_type: str | list[str] = ['cyto3', 'nuclei'], 
             cellpose_diameter: float | list[float] = 180, 
             cellpose_channel_axis: int = 0,
             cellpose_invert: bool | list = False, 
             cellpose_normalize: bool = True, 
             do_3D_Segmentation: bool = False, # This is not implemented
             cellpose_min_size: float | list[float] = 500, 
             cellpose_flow_threshold: float | list[float] = 0, 
             cellpose_cellprob_threshold: float | list[float] = 0,
             cellpose_pretrained_model: str | list[str] = False,
             display_plots: bool = False,
               **kwargs):
        if image.shape[1] >= 1:
            image = np.max(image, axis=1)
        
        results = {}
        mask_structure = {name: ms for (name, ms) in mask_structure.items() if ms[2] is not None}
        for i, (name, ms) in enumerate(mask_structure.items()):
            structure = ms[0]
            channel = ms[1]
            parent = ms[2]

            if channel == 'nucChannel':
                channel = nucChannel
            elif channel == 'cytoChannel':
                channel = cytoChannel
            elif channel == 'FISHChannel':
                channel = FISHChannel
            else:
                channel = channel

            mask = self.segment(image, channel, cellpose_min_size, cellpose_flow_threshold,
                                cellpose_cellprob_threshold, cellpose_model_type, cellpose_diameter,
                                cellpose_channel_axis, cellpose_invert, cellpose_normalize, do_3D_Segmentation,
                                cellpose_pretrained_model, i)

            results[name] = mask

        nuc_key = next((key for key in results.keys() if 'nuc' in key.lower()), None)
        cell_key = next((key for key in results.keys() if 'cell' in key.lower()), None)
        if nuc_key and cell_key:
            results[nuc_key], results[cell_key] = self.align_nuc_cell_masks(results[nuc_key], results[cell_key])
        self.plot_segmentation(display_plots, image, results)

        return results
        
    def unpack_lists(self, cellpose_min_size, 
             cellpose_flow_threshold, 
             cellpose_cellprob_threshold,
             cellpose_model_type,
             cellpose_diameter,
             pretrained_model, i):
        if isinstance(cellpose_min_size, list):
            min_size = cellpose_min_size[i]
        else:
            min_size = cellpose_min_size
        
        if isinstance(cellpose_flow_threshold, list):
            flow_threshold = cellpose_flow_threshold[i]
        else:
            flow_threshold = cellpose_flow_threshold
        
        if isinstance(cellpose_cellprob_threshold, list):
            cellprob_threshold = cellpose_cellprob_threshold[i]
        else:
            cellprob_threshold = cellpose_cellprob_threshold

        if isinstance(cellpose_model_type, list):
            model_type = cellpose_model_type[i]
        else:
            model_type = cellpose_model_type

        if isinstance(cellpose_diameter, list):
            diameter = cellpose_diameter[i]
        else:
            diameter = cellpose_diameter

        if isinstance(pretrained_model, list):
            pretrained_model = pretrained_model[i]
        else:
            pretrained_model = pretrained_model

        return (min_size, flow_threshold, cellprob_threshold, 
                model_type, diameter, pretrained_model) 

    def segment(self, image, channel, cellpose_min_size, cellpose_flow_threshold, 
                       cellpose_cellprob_threshold, cellpose_model_type, cellpose_diameter, 
                       cellpose_channel_axis, cellpose_invert, cellpose_normalize, cellpose_do_3D,
                       cellpose_pretrained_model, i):
        (min_size, flow_threshold, cellprob_threshold, 
        model_type, diameter, pretrained_model)  = self.unpack_lists(cellpose_min_size, 
                                                                cellpose_flow_threshold, 
                                                                cellpose_cellprob_threshold,
                                                                cellpose_model_type,
                                                                cellpose_diameter,
                                                                cellpose_pretrained_model, i)
            
        model_location = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        pretrained_model = os.path.join(model_location, pretrained_model) if pretrained_model else None

        cp = models.CellposeModel(model_type=model_type, gpu=True, pretrained_model=pretrained_model)
        sz = models.SizeModel(cp)

        model = models.Cellpose(gpu=True)
        model.cp = cp
        model.sz = sz

        channels = [0, 0]
        image = image[channel, :, :].compute()
        mask, flows, styles, diams = model.eval(image,
                                            channels=channels, 
                                            diameter=diameter, 
                                            invert=cellpose_invert, 
                                            normalize=cellpose_normalize, 
                                            channel_axis=cellpose_channel_axis, 
                                            do_3D=cellpose_do_3D,
                                            min_size=min_size, 
                                            flow_threshold=flow_threshold, 
                                            cellprob_threshold=cellprob_threshold)
                                            # net_avg=True
        return mask

    def align_masks(self, mask_dict):
        # Find the mask with the largest number of objects
        largest_mask_name = max(mask_dict, key=lambda k: np.max(mask_dict[k]))
        largest_mask = mask_dict[largest_mask_name]

        # Initialize the final aligned masks with the largest mask
        aligned_masks = {name: np.zeros_like(mask) for name, mask in mask_dict.items()}
        aligned_masks[largest_mask_name] = largest_mask

        # Iterate over each mask and align with the largest mask
        for name, mask in mask_dict.items():
            if name == largest_mask_name:
                continue

            # Create markers for watershed
            markers = np.zeros_like(mask, dtype=int)
            distance = distance_transform_edt(mask > 0)
            for label in np.unique(largest_mask):
                if label == 0:
                    continue
                d = distance.copy()
                d[largest_mask != label] = 0
                center = np.unravel_index(np.argmax(d), distance.shape)
                markers[center] = label

            # Apply watershed to align masks
            aligned_mask = watershed(-distance, markers, mask=mask > 0)
            aligned_masks[name] = aligned_mask

        return aligned_masks

    def plot_segmentation(self, display_plots, image, mask_dict):
        if display_plots:
            num_sub_plots = len(mask_dict) + 1
            fig, axs = plt.subplots(1, num_sub_plots, figsize=(15, 5))
            axs[0].imshow(np.max(image, axis=0))
            axs[0].set_title('Original Image')
            axs[0].axis('off')
            i = 1
            for name, mask in mask_dict.items():
                axs[i].imshow(mask, cmap='nipy_spectral')
                axs[i].set_title(name)
                axs[i].axis('off')
                i += 1
            plt.tight_layout()
            plt.show()




if __name__ == '__main__':
    pass
    # print(os.path.dirname(os.path.dirname(__file__)))
