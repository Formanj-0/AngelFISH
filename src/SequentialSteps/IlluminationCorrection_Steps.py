import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import exposure
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import dask.array as da
from abc import abstractmethod
import os
import sys

# append the path two directories before this file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src import IndependentStepClass
from src.Parameters import Parameters


class IlluminationCorrection(IndependentStepClass):
    def __init__(self):
        """
        Initialize the IlluminationCorrection class.
        """
        super().__init__()

    def main(self, images, sigma_dict, display_plots=False, imported_profiles=None, **kwargs):
        """
        Full pipeline to create profiles, correct images, and visualize.

        Parameters:
        - images: Dask array of shape [P, T, C, Z, Y, X].
        - sigma_dict: Dictionary of sigma values for smoothing per channel.
        - display_plots: Boolean to control visualization.
        - imported_profiles: ndarray of shape [C, Y, X], precomputed illumination profiles.

        Returns:
        - corrected_images: Dask array of shape [P, T, C, Z, Y, X].
        - illumination_profiles: ndarray of shape [C, Y, X].
        - corrected_IL_profile: ndarray of shape [C, Y, X].
        """
        if not isinstance(images, da.Array):
            raise TypeError("Expected 'images' to be a dask.array.")

        print("Starting illumination correction pipeline...")
        print(f"Initial shape of images: {images.shape}")
        print(f"Initial image type: {images.dtype}")
        print(f"Initial chunking of images: {images.chunks}")

        if imported_profiles is not None:
            illumination_profiles = np.array(imported_profiles)
            print("Using imported illumination profiles.")
        else:
            print("Creating new illumination profiles...")
            illumination_profiles = self.create_illumination_profiles(images, sigma_dict)
            print(f"Illumination profiles created with shape: {illumination_profiles.shape}")

        print("Fitting Gaussian profiles to the illumination profiles...")
        fitted_profiles = np.stack([
            self.fit_gaussian_2d(illumination_profiles[c])
            for c in range(illumination_profiles.shape[0])
        ], axis=0)
        print(f"Fitted Gaussian profiles created with shape: {fitted_profiles.shape}")

        fitted_profiles /= fitted_profiles.max(axis=(1, 2), keepdims=True)

        print("Applying illumination correction to images...")
        corrected_images = self.apply_correction(images, fitted_profiles)
        print(f"Shape of corrected images: {corrected_images.shape}")
        print(f"Type of corrected images: {corrected_images.dtype}")

        print("Creating corrected illumination profiles...")
        corrected_IL_profile = self.create_illumination_profiles(
            corrected_images, sigma_dict, original_profile=illumination_profiles
        )
        print(f"Corrected profiles created with shape: {corrected_IL_profile.shape}")

        if display_plots:
            print("Visualizing illumination profiles...")
            self.visualize_profiles(fitted_profiles, corrected_images, images, corrected_IL_profile)

        print("Illumination correction pipeline complete.")
        return {'images': corrected_images, 'illumination_profiles': fitted_profiles, 'corrected_IL_profile': corrected_IL_profile}

    def validate_sigma_dict(self, num_channels, sigma_dict):
        print(f"Validating sigma_dict with {num_channels} channels...")
        if len(sigma_dict) != num_channels:
            raise ValueError(f"Expected sigma_dict to have {num_channels} entries, but got {len(sigma_dict)}.")
        print("sigma_dict validated.")

    def gaussian_2d(self, x, y, x0, y0, sigma_x, sigma_y, amplitude, offset):
        return offset + amplitude * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2)))

    def fit_gaussian_2d(self, illumination_profile):
        print("Fitting Gaussian to illumination profile...")
        y = np.arange(illumination_profile.shape[0])
        x = np.arange(illumination_profile.shape[1])
        x, y = np.meshgrid(x, y)
        xdata = np.vstack((x.ravel(), y.ravel()))
        ydata = illumination_profile.ravel()

        initial_guess = (
            illumination_profile.shape[1] / 2,
            illumination_profile.shape[0] / 2,
            illumination_profile.shape[1] / 4,
            illumination_profile.shape[0] / 4,
            np.max(illumination_profile),
            np.min(illumination_profile)
        )

        bounds = (
            (0, 0, 1, 1, 0, 0),
            (illumination_profile.shape[1], illumination_profile.shape[0],
             illumination_profile.shape[1], illumination_profile.shape[0],
             np.inf, np.inf)
        )

        try:
            popt, _ = curve_fit(
                lambda xy, x0, y0, sigma_x, sigma_y, amplitude, offset:
                    self.gaussian_2d(xy[0], xy[1], x0, y0, sigma_x, sigma_y, amplitude, offset),
                xdata, ydata, p0=initial_guess, bounds=bounds
            )
            fitted_profile = self.gaussian_2d(x, y, *popt).reshape(illumination_profile.shape)
            print(f"Gaussian fitting complete. Parameters: {popt}")
        except RuntimeError:
            print("Gaussian fitting failed. Using initial guess as fallback.")
            fitted_profile = self.gaussian_2d(x, y, *initial_guess).reshape(illumination_profile.shape)

        return fitted_profile

    def create_illumination_profiles(self, images, sigma_dict, original_profile=None):
        print("Computing percentile projection along Z-axis...")
        projected = da.percentile(images, 10, axis=3, keepdims=True)  # 10th percentile along Z
        print("Computing median projection across P...")
        median_profile = da.median(projected, axis=0).compute()  # Median across P [T, C, 1, Y, X]

        num_channels = median_profile.shape[1]
        self.validate_sigma_dict(num_channels, sigma_dict)

        print("Smoothing profiles for each channel...")
        smoothed_profiles = np.stack([
            gaussian_filter(median_profile[0, c, 0], sigma=sigma_dict[c])
            for c in range(num_channels)
        ], axis=0)

        if original_profile is not None:
            print("Normalizing corrected profiles to match original profile scale...")
            smoothed_profiles *= original_profile.max(axis=(1, 2), keepdims=True) / smoothed_profiles.max(axis=(1, 2), keepdims=True)

        print("Smoothing complete. Returning illumination profiles.")
        return smoothed_profiles

    def apply_correction(self, images, fitted_profile):
        print("Preparing correction profile...")
        epsilon = 1e-6
        correction_profile = 1 / (fitted_profile + epsilon)

        def correct_block(block, correction_profile):
            correction_expanded = correction_profile[np.newaxis, np.newaxis, :, np.newaxis, :, :]
            return block * correction_expanded

        print("Applying correction to image blocks...")
        corrected_images = da.map_blocks(
            correct_block,
            images,
            correction_profile=correction_profile,
            dtype=images.dtype
        )
        print("Correction applied to all images.")
        return corrected_images

    def visualize_profiles(self, illumination_profile, corrected_images, images, corrected_IL_profile):
        print("Visualizing illumination profiles and intensity distributions...")
        original_proj = da.percentile(images, 10, axis=(0, 1, 3)).compute()  # [C, Y, X]
        corrected_proj = da.percentile(corrected_images, 10, axis=(0, 1, 3)).compute()

        normalized_profile = illumination_profile / illumination_profile.max(axis=(1, 2), keepdims=True)
        normalized_corrected = corrected_IL_profile / corrected_IL_profile.max(axis=(1, 2), keepdims=True)

        num_channels = illumination_profile.shape[0]
        for c in range(num_channels):
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))

            sns.heatmap(normalized_profile[c], cmap="viridis", ax=axes[0, 0], cbar=True)
            axes[0, 0].set_title(f"Original Illumination Profile (Channel {c})")
            axes[0, 0].axis("off")
            axes[0, 0].contour(normalized_profile[c], levels=10, colors='white', linewidths=0.5)

            sns.heatmap(normalized_corrected[c], cmap="viridis", ax=axes[0, 1], cbar=True)
            axes[0, 1].set_title(f"Corrected Illumination Profile (Channel {c})")
            axes[0, 1].axis("off")
            axes[0, 1].contour(normalized_corrected[c], levels=10, colors='white', linewidths=0.5)

            axes[1, 0].hist(original_proj[c].ravel(), bins=50, alpha=0.7, label="Original", density=True)
            axes[1, 0].set_title(f"Intensity Distribution (Channel {c}) - Original")
            axes[1, 0].legend()

            axes[1, 1].hist(corrected_proj[c].ravel(), bins=50, alpha=0.7, label="Corrected", density=True, color="orange")
            axes[1, 1].set_title(f"Intensity Distribution (Channel {c}) - Corrected")
            axes[1, 1].legend()

            plt.tight_layout()
            plt.show()

        print("Visualization complete.")
