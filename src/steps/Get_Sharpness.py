import numpy as np
from typing import Union
from bigfish import stack, detection, multistack, plot
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import napari
from magicgui import magicgui
import pathlib
import re
from skimage import feature
from scipy.ndimage import convolve
import cv2
import skimage as sk
from scipy.stats import norm, entropy
from scipy import ndimage as ndi


from src import abstract_task, load_data
import json


class calculate_sharpness(abstract_task):
    @classmethod
    def task_name(cls):
        return 'detect_spots'

    def extract_args(self, p, t):
        given_args = self.receipt['steps'][self.step_name]
        c = given_args['channel']
        data_to_send = {}
        data_to_send['zyx_image'] = self.data['images'][p, t, c].compute()

        args = {**data_to_send, **given_args}

        args['fov'] = p
        args['timepoint'] = t

        return args

    def preallocate_memmory(self):
        self.sharpnesses = {}

    @staticmethod
    def image_processing_function(
        zyx_image: np.array,
        p:int,
        t:int,
        sharpness_metric: Union[list,str]=None,
        **kwargs
        ):
        if sharpness_metric is None:
            sharpness_metric = ['derivative', 'fft', 'edge', 'tenengrad', 'laplacian', 'bayes']
        if isinstance(sharpness_metric, str):
            sharpness_metric = [sharpness_metric]

        sharpness_results = {}
        for z in range(zyx_image.shape[0]):
            sharpness_results[z] = {}
            for metric in sharpness_metric:
                sharpness_results[z][metric] = calculate_sharpness_metric(zyx_image[z], metric)

        def calculate_sharpness_metric(img, sharpness_metric: str = 'fft'):
            if sharpness_metric == 'derivative':
                gy, gx = np.gradient(img)
                gnorm = np.sqrt(gx ** 2 + gy ** 2)
                sharpness = np.average(gnorm)
                return sharpness

            elif sharpness_metric == 'fft':
                fftimage = np.fft.fft2(img)
                fftshift = np.fft.fftshift(fftimage)
                fftshift = np.absolute(fftshift)
                M = np.amax(fftshift)
                Th = (fftshift > M // 1000).sum()
                sharpness = Th // img.shape[0] * img.shape[1]
                return sharpness*1000

            elif sharpness_metric == 'edge':
                edges1 = feature.canny(img, sigma=3)
                kernel = np.ones((3, 3))
                kernel[1, 1] = 0
                sharpness = convolve(edges1, kernel, mode="constant")
                sharpness = sharpness[edges1 != 0].sum()
                return sharpness
            
            elif sharpness_metric == 'tenengrad':
                gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                g = np.sqrt(gx**2 + gy**2)
                sharpness = np.mean(g)
                return sharpness
            
            elif sharpness_metric == 'laplacian':
                laplacian = cv2.Laplacian(img, cv2.CV_64F)
                variance = np.var(laplacian)
                return variance
            
            elif sharpness_metric == 'bayes':
                '''
                This is my sharpness metric. I assume that the out of 
                focus foreground follows some model, in this case a gaussian with a 
                mean and std found from the z slice. Then I assume that when the image
                is in focus it deviated from this basic model. I then calculate the 
                difference between the out of focus model and the pdf of pixel intensities.

                The reason this works is because when the image is out of focus the foregound
                becomes fairly 'smooth' and there no roughness, and when the image is in focus
                it gains roughness, so more peaks and more valleys. This cause the distribution of 
                pixel intensities to take on multiple peaks or even become flat. When the image is out of focus 
                you should only expect one peak, when it is in focus you should expect multiple.
                '''
                # find foreground
                thresh = sk.filters.threshold_otsu(img)
                binary = img > thresh
                binary = ndi.binary_fill_holes(binary)
                data = img[binary].flatten()

                # find pdf of actual pixel intensities 
                hist, bin_edges = np.histogram(data, bins=100, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                # fit model of out of focus foreground
                mu, std = norm.fit(data)
                gauss_pdf = norm.pdf(bin_centers, mu, std)

                # Shannon entropy between the real pdf and the model pdf
                kl = entropy(hist, gauss_pdf)
                return kl

            else:
                raise Exception('Sharpness metric not recognized')

        return sharpness_results

    def write_results(self, results, p, t):
        if self.sharpnesses[p] is None:
            self.sharpnesses[p] = {}
        self.sharpnesses[p][t] = results

    def compress_and_release_memory(self):
        results_dir = self.receipt['dirs']['results_dir']
        output_path = os.path.join(results_dir, f"sharpnesses.json")
        with open(output_path, "w") as f:
            json.dump(self.sharpnesses, f, indent=2)

    @property
    def required_keys(self):
        return ['FISHChannel']
























