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

from src import  IndependentStepClass  # TODO: remove this
from src.Parameters import Parameters


class IlluminationCorrection(IndependentStepClass):
    def __init__(self):
        """
        Initialize the IlluminationCorrection class.
        """
        super().__init__()

    def main(self, images, sigma_dict, display_plots=False, illumination_profiles=None, **kwargs):
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
        """
        # Ensure images is a dask.array
        if not isinstance(images, da.Array):
            raise TypeError("Expected 'images' to be a dask.array.")
        
        print("Starting illumination correction pipeline...")
        print(f"Initial shape of images: {images.shape}")
        print(f"Initial image type: {images.dtype}")
        print(f"Initial chunking of images: {images.chunks}")

        # Use or create illumination profiles
        if illumination_profiles is not None:
            if not isinstance(illumination_profiles, np.ndarray):
                raise TypeError("Imported profiles must be a NumPy array.")
            # illumination_profiles = imported_profiles
            print("Using imported illumination profiles.")
        else:
            print("Creating new illumination profiles...")
            illumination_profiles = self.create_illumination_profiles(images, sigma_dict)
            print(f"Illumination profiles created with shape: {illumination_profiles.shape}")

        # Fit Gaussian to each channel of the illumination profiles
        print("Fitting Gaussian profiles to the illumination profiles...")
        fitted_profiles = np.stack([
            self.fit_gaussian_2d(illumination_profiles[c])
            for c in range(illumination_profiles.shape[0])
        ], axis=0)
        print(f"Fitted Gaussian profiles created with shape: {fitted_profiles.shape}")

        # Normalize profiles
        fitted_profiles /= fitted_profiles.max(axis=(1, 2), keepdims=True)

        # Apply illumination correction
        print("Applying illumination correction to images...")
        print(f"Shape of images before correction: {images.shape}")
        # print(f"Chunking of images before correction: {images.chunks}")
        corrected_images = self.apply_correction(images, fitted_profiles)
        print(f"Shape of corrected images: {corrected_images.shape}")
        print(f"Type of corrected images: {corrected_images.dtype}")
        # print(f"Chunking of corrected images: {corrected_images.chunks}")
        if not isinstance(corrected_images, da.Array):
            raise TypeError("Expected 'corrected_images' to be a dask.array.")
        print("Illumination correction applied.")

        # Create corrected profiles
        print("Creating corrected illumination profiles...")
        corrected_IL_profile = self.create_illumination_profiles(corrected_images, sigma_dict)
        print(f"Corrected profiles created with shape: {corrected_IL_profile.shape}")

        # Visualize if requested
        if display_plots:
            print("Visualizing illumination profiles...")
            self.visualize_profiles(fitted_profiles, corrected_images, images, corrected_IL_profile)

        print("Illumination correction pipeline complete.")
        return {'images': corrected_images, 'illumination_profiles': fitted_profiles, 'corrected_IL_profile': corrected_IL_profile}

    def validate_sigma_dict(self, num_channels, sigma_dict):
        """
        Ensure that sigma_dict has the same length as the number of channels.
        """
        print(f"Validating sigma_dict with {num_channels} channels...")
        if len(sigma_dict) != num_channels:
            raise ValueError(f"Expected sigma_dict to have {num_channels} entries, but got {len(sigma_dict)}.")
        print("sigma_dict validated.")

    def gaussian_2d(self, x, y, x0, y0, sigma_x, sigma_y, amplitude, offset):
        """2D Gaussian function."""
        return offset + amplitude * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2)))

    def fit_gaussian_2d(self, illumination_profile):
        """
        Fit a 2D Gaussian to the illumination profile.

        Parameters:
        - illumination_profile: ndarray of shape [Y, X], the smoothed illumination profile.

        Returns:
        - fitted_profile: ndarray of shape [Y, X], the Gaussian fit to the illumination profile.
        """
        print("Fitting Gaussian to illumination profile...")

        # Define coordinate grid
        y = np.arange(illumination_profile.shape[0])
        x = np.arange(illumination_profile.shape[1])
        x, y = np.meshgrid(x, y)

        # Flatten grid and data
        xdata = np.vstack((x.ravel(), y.ravel()))
        ydata = illumination_profile.ravel()

        # Initial guess for Gaussian parameters
        initial_guess = (
            illumination_profile.shape[1] / 2,  # x0 (center X)
            illumination_profile.shape[0] / 2,  # y0 (center Y)
            illumination_profile.shape[1] / 4,  # sigma_x
            illumination_profile.shape[0] / 4,  # sigma_y
            np.max(illumination_profile),       # amplitude
            np.min(illumination_profile)        # offset
        )

        # Parameter bounds to ensure stability
        bounds = (
            (0, 0, 1, 1, 0, 0),  # Lower bounds: non-negative parameters
            (illumination_profile.shape[1], illumination_profile.shape[0],
            illumination_profile.shape[1], illumination_profile.shape[0],
            np.inf, np.inf)     # Upper bounds
        )

        try:
            # Fit Gaussian using curve_fit
            popt, _ = curve_fit(
                lambda xy, x0, y0, sigma_x, sigma_y, amplitude, offset:
                self.gaussian_2d(xy[0], xy[1], x0, y0, sigma_x, sigma_y, amplitude, offset),
                xdata, ydata, p0=initial_guess, bounds=bounds
            )

            # Generate the fitted profile
            fitted_profile = self.gaussian_2d(x, y, *popt).reshape(illumination_profile.shape)
            print(f"Gaussian fitting complete. Parameters: {popt}")
        except RuntimeError:
            print("Gaussian fitting failed. Using initial guess as fallback.")
            fitted_profile = self.gaussian_2d(x, y, *initial_guess).reshape(illumination_profile.shape)

        return fitted_profile

    def create_illumination_profiles(self, images, sigma_dict, original_profile=None):
        """
        Create illumination profiles for each channel.

        Parameters:
        - images: Dask array of shape [P, T, C, Z, Y, X].
        - sigma_dict: Dictionary of sigma values for smoothing per channel.
        - original_profile: ndarray of shape [C, Y, X], original profile for normalization.

        Returns:
        - illumination_profiles: ndarray of shape [C, Y, X].
        """
        print("Computing median projection along Z-axis...")
        # Compute median along Z-axis for each position (P) and channel (C)
        median_projected = da.median(images, axis=3, keepdims=True)  # Median along Z [P, T, C, 1, Y, X]

        print("Computing median projection across P...")
        # Compute median across all positions (P)
        median_profile = da.median(median_projected, axis=0).compute()  # Median across P [T, C, 1, Y, X]

        num_channels = median_profile.shape[1]
        self.validate_sigma_dict(num_channels, sigma_dict)

        print("Smoothing profiles for each channel...")
        # Apply Gaussian smoothing to the illumination profile for each channel
        smoothed_profiles = np.stack([
            gaussian_filter(median_profile[0, c, 0], sigma=sigma_dict[c])
            for c in range(num_channels)
        ], axis=0)

        # Normalize the smoothed profiles to match the scale of the original profile
        if original_profile is not None:
            print("Normalizing corrected profiles to match original profile scale...")
            smoothed_profiles *= original_profile.max(axis=(1, 2), keepdims=True) / smoothed_profiles.max(axis=(1, 2), keepdims=True)

        print("Smoothing complete. Returning illumination profiles.")
        return smoothed_profiles

    def apply_correction(self, images, fitted_profile):
        """
        Apply illumination correction using the inverse of the Gaussian fit.

        Parameters:
        - images: Dask array of shape [P, T, C, Z, Y, X].
        - fitted_profile: ndarray of shape [C,Y, X], the Gaussian fit to the illumination profile.

        Returns:
        - corrected_images: Dask array of shape [P, T, C, Z, Y, X].
        """
        print("Preparing correction profile...")

        epsilon = 1e-6
        # Normalize correction profile to preserve original intensity scale
        correction_profile = 1 / (fitted_profile + epsilon)

        def correct_block(block, correction_profile):
            """
            Correct a single block of the dask array using the correction profile.

            Parameters:
            - block: ndarray of shape [P, T, C, Z, Y, X].
            - correction_profile: ndarray of shape [C, Y, X].

            Returns:
            - corrected_block: ndarray of the same shape as block.
            """
            # Expand correction profile to match block dimensions [1, 1, C, 1, Y, X]
            correction_expanded = correction_profile[np.newaxis, np.newaxis, :, np.newaxis, :, :]

            # Apply correction using broadcasting
            corrected_block = block * correction_expanded
            return corrected_block

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
        """
        Visualize illumination profiles and intensity distributions.

        Parameters:
        - illumination_profile: ndarray of shape [C, Y, X], the fitted illumination profile.
        - corrected_images: Dask array of shape [P, T, C, Z, Y, X], corrected images.
        - images: Dask array of shape [P, T, C, Z, Y, X], original images.
        - corrected_IL_profile: ndarray of shape [C, Y, X], the corrected illumination profile.
        """
        print("Visualizing illumination profiles and intensity distributions...")

        # Compute mean projection along Z-axis for original and corrected images
        print("Computing mean projection along Z-axis for original and corrected images...")
        original_mean = images.mean(axis=(0, 1, 3)).compute()  # Mean along P, T, and Z [C, Y, X]
        corrected_mean = corrected_images.mean(axis=(0, 1, 3)).compute()  # Mean along P, T, and Z [C, Y, X]

        # Normalize illumination profile for visualization
        print("Normalizing profiles for visualization...")
        normalized_profile = illumination_profile / illumination_profile.max(axis=(1, 2), keepdims=True)
        normalized_corrected = corrected_IL_profile / corrected_IL_profile.max(axis=(1, 2), keepdims=True)  # Normalize by original profile for consistency

        num_channels = illumination_profile.shape[0]
        for c in range(num_channels):
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))

            # Top-left: Original smoothed profile with contours
            sns.heatmap(normalized_profile[c], cmap="viridis", ax=axes[0, 0], cbar=True)
            axes[0, 0].set_title(f"Original Illumination Profile (Channel {c})")
            axes[0, 0].axis("off")
            axes[0, 0].contour(normalized_profile[c], levels=10, colors='white', linewidths=0.5)

            # Top-right: Corrected smoothed profile with contours
            sns.heatmap(normalized_corrected[c], cmap="viridis", ax=axes[0, 1], cbar=True)
            axes[0, 1].set_title(f"Corrected Illumination Profile (Channel {c})")
            axes[0, 1].axis("off")
            axes[0, 1].contour(normalized_corrected[c], levels=10, colors='white', linewidths=0.5)

            # Bottom-left: Intensity distribution - Original
            original_flat = original_mean[c].ravel()
            axes[1, 0].hist(original_flat, bins=50, alpha=0.7, label="Original", density=True)
            axes[1, 0].set_title(f"Intensity Distribution (Channel {c}) - Original")
            axes[1, 0].legend()

            # Bottom-right: Intensity distribution - Corrected
            corrected_flat = corrected_mean[c].ravel()
            axes[1, 1].hist(corrected_flat, bins=50, alpha=0.7, label="Corrected", density=True, color="orange")
            axes[1, 1].set_title(f"Intensity Distribution (Channel {c}) - Corrected")
            axes[1, 1].legend()

            plt.tight_layout()
            plt.show()

        print("Visualization complete.")


