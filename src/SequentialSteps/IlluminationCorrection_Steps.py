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
        """
        # Ensure images is a dask.array
        if not isinstance(images, da.Array):
            raise TypeError("Expected 'images' to be a dask.array.")
        
        print("Starting illumination correction pipeline...")
        print(f"Initial shape of images: {images.shape}")
        print(f"Initial image type: {images.dtype}")
        print(f"Initial chunking of images: {images.chunks}")

        # Use or create illumination profiles
        if imported_profiles is not None:
            if not isinstance(imported_profiles, np.ndarray):
                raise TypeError("Imported profiles must be a NumPy array.")
            illumination_profiles = imported_profiles
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
        return {'images': corrected_images, 'illumination_profiles': fitted_profiles}


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
        correction_profile = fitted_profile.mean(axis=(1, 2), keepdims=True) / (fitted_profile + epsilon)

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
        normalized_corrected = corrected_IL_profile / illumination_profile.max(axis=(1, 2), keepdims=True)  # Normalize by original profile for consistency

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


#%%
# #%%
# from skimage.filters import threshold_otsu
# import dask.array as da
# import numpy as np
# from scipy.ndimage import gaussian_filter
# import matplotlib.pyplot as plt
# from dask import delayed

# class IlluminationCorrection_BGFG(IndependentStepClass):
#     def __init__(self):
#         """
#         Initialize the IlluminationCorrection class.
#         """
#         super().__init__()

#     def otsu_threshold(self, max_projection):
#         """
#         Apply Otsu's method to separate foreground and background.

#         Parameters:
#         - max_projection: ndarray of shape [Y, X], 2D max projection of an image.

#         Returns:
#         - foreground_mask: Binary mask of the foreground.
#         - background_mask: Binary mask of the background.
#         """
#         print("Computing Otsu threshold...")
#         threshold_value = threshold_otsu(max_projection)
#         print(f"Otsu threshold value: {threshold_value}")

#         # Create binary masks
#         foreground_mask = max_projection > threshold_value
#         background_mask = ~foreground_mask

#         return foreground_mask, background_mask

#     # def create_illumination_profiles(self, images, sigma_dict):
#     #     """
#     #     Create illumination profiles for each channel.

#     #     Parameters:
#     #     - images: Dask array of shape [P, T, C, Z, Y, X].

#     #     Returns:
#     #     - illumination_profiles: ndarray of shape [C, Y, X].
#     #     """
#     #     print("Computing max projection along Z-axis for all channels...")
#     #     max_projected = images.max(axis=3)  # Lazy operation [P, T, C, Y, X]

#     #     print("Computing median across P dimension...")
#     #     median_projection = da.median(max_projected, axis=0)  # Lazy operation [T, C, Y, X]

#     #     num_channels = median_projection.shape[1]
#     #     self.validate_sigma_dict(num_channels, sigma_dict)

#     #     print("Smoothing profiles and applying Otsu threshold for each channel...")
#     #     illumination_profiles = []
#     #     for c in range(num_channels):
#     #         print(f"Processing channel {c}...")

#     #         # Delayed computation for Otsu and smoothing
#     #         channel_projection = delayed(median_projection)[0, c]  # Extract channel projection
#     #         foreground_mask = delayed(self.otsu_threshold)(channel_projection)[0]
#     #         smoothed_profile = delayed(gaussian_filter)(channel_projection * foreground_mask, sigma=sigma_dict[c])

#     #         illumination_profiles.append(smoothed_profile)

#     #     # Stack profiles (forcing computation)
#     #     illumination_profiles = da.stack([da.from_delayed(profile, shape=(None, None), dtype=np.float32)
#     #                                     for profile in illumination_profiles], axis=0)
#     #     print("Illumination profiles created.")
#     #     return illumination_profiles.compute()  # Trigger computation

#     def create_illumination_profiles(self, images, sigma_dict):
#         """
#         Create illumination profiles for each channel.

#         Parameters:
#         - images: Dask array of shape [P, T, C, Z, Y, X].

#         Returns:
#         - illumination_profiles: ndarray of shape [C, Y, X].
#         """
#         print("Computing max projection along Z-axis for all channels...")
#         # Compute max projection along Z-axis once for efficiency
#         max_projected = images.max(axis=3)  # Shape [P, T, C, Y, X]

#         print("Computing median across P dimension...")
#         median_projection = da.median(max_projected, axis=0)  # Shape [T, C, Y, X]
#         median_projection_computed = median_projection.compute()  # Convert to NumPy array

#         num_channels = median_projection_computed.shape[1]
#         self.validate_sigma_dict(num_channels, sigma_dict)

#         print("Smoothing profiles and applying Otsu threshold for each channel...")
#         illumination_profiles = []
#         for c in range(num_channels):
#             print(f"Processing channel {c}...")
#             # Extract channel-specific max projection (median value for [T, Y, X])
#             channel_projection = median_projection_computed[0, c]

#             # Apply Otsu thresholding
#             foreground_mask, _ = self.otsu_threshold(channel_projection)

#             # Smooth only the foreground for illumination profile
#             smoothed_profile = gaussian_filter(channel_projection * foreground_mask, sigma=sigma_dict[c])
#             illumination_profiles.append(smoothed_profile)

#         illumination_profiles = np.stack(illumination_profiles, axis=0)  # Stack profiles into [C, Y, X]
#         print("Illumination profiles created.")
#         return illumination_profiles

#     def main(self, images, sigma_dict, display_plots=False, imported_profiles=None, **kwargs):
#         """
#         Full pipeline to create profiles, correct images, and visualize.

#         Parameters:
#         - images: Dask array of shape [P, T, C, Z, Y, X].
#         - sigma_dict: Dictionary of sigma values for smoothing per channel.
#         - display_plots: Boolean to control visualization.
#         - imported_profiles: ndarray of shape [C, Y, X], precomputed illumination profiles.

#         Returns:
#         - corrected_images: Dask array of shape [P, T, C, Z, Y, X].
#         - illumination_profiles: ndarray of shape [C, Y, X].
#         """
#         print("Starting illumination correction pipeline...")

#         # Validate sigma_dict
#         self.validate_sigma_dict(images.shape[2], sigma_dict)

#         if imported_profiles is not None:
#             if not isinstance(imported_profiles, np.ndarray):
#                 raise TypeError("Imported profiles must be a NumPy array.")
#             illumination_profiles = imported_profiles
#             print("Using imported illumination profiles.")
#         else:
#             print("Creating new illumination profiles...")
#             illumination_profiles = self.create_illumination_profiles(images, sigma_dict)
#             print("New illumination profiles created.")

#         print("Applying illumination correction to images...")
#         corrected_images = self.apply_correction(images, illumination_profiles)
#         print("Illumination correction applied.")

#         if display_plots:
#             print("Visualizing illumination profiles...")
#             self.visualize_profiles(illumination_profiles, corrected_images, sigma_dict)

#         print("Illumination correction pipeline complete.")
#         New_Parameters({'images': corrected_images, 'illumination_profiles': illumination_profiles})
#         return corrected_images, illumination_profiles

#     def validate_sigma_dict(self, num_channels, sigma_dict):
#         """
#         Ensure that sigma_dict has the same length as the number of channels.
#         """
#         print(f"Validating sigma_dict with {num_channels} channels...")
#         if len(sigma_dict) != num_channels:
#             raise ValueError(f"Expected sigma_dict to have {num_channels} entries, but got {len(sigma_dict)}.")
#         print("sigma_dict validated.")

#     def apply_correction(self, images, illumination_profiles):
#         """
#         Apply illumination correction to the input images.

#         Parameters:
#         - images: Dask array of shape [P, T, C, Z, Y, X].
#         - illumination_profiles: ndarray of shape [C, Y, X], smoothed illumination profiles.

#         Returns:
#         - corrected_images: Dask array of shape [P, T, C, Z, Y, X].
#         """
#         print("Preparing correction profiles...")
#         epsilon = 1e-6
#         correction_profiles = 1.0 / (illumination_profiles + epsilon)

#         def correct_block(block, profiles):
#             corrected_block = np.zeros_like(block)
#             for c in range(block.shape[2]):  # Loop over channels
#                 correction_profile = profiles[c]
#                 for z in range(block.shape[3]):  # Loop over Z slices
#                     slice_ = block[:, :, c, z, :, :]
#                     corrected_slice = slice_ * correction_profile[np.newaxis, np.newaxis, :, :]
#                     corrected_block[:, :, c, z, :, :] = corrected_slice
#             return corrected_block

#         print("Applying correction to image blocks...")
#         corrected_images = da.map_blocks(
#             correct_block,
#             images,
#             correction_profiles,
#             dtype=images.dtype,
#             chunks=images.chunks
#         )
#         print("Correction applied to all images.")
#         return corrected_images

    
#     def visualize_profiles(self, illumination_profiles, corrected_images, sigma_dict):
#         """
#         Visualize illumination profiles before and after correction, including foreground masks.

#         Parameters:
#         - illumination_profiles: ndarray of shape [C, Y, X].
#         - corrected_images: Dask array of shape [P, T, C, Z, Y, X].
#         - sigma_dict: Dictionary of sigma values for smoothing.
#         """
#         print("Creating smoothed profiles for corrected images...")
#         corrected_max_projected = corrected_images.max(axis=3, keepdims=True)  # Max project along Z
#         corrected_profiles = self.create_illumination_profiles(corrected_max_projected, sigma_dict)

#         for c in range(illumination_profiles.shape[0]):
#             print(f"Visualizing channel {c}...")
#             fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

#             # Original smoothed illumination profile
#             sns.heatmap(illumination_profiles[c], cmap='hot', cbar=True, ax=axes[0])
#             axes[0].set_title(f'Original Illumination Profile - Channel {c}')
#             axes[0].axis('off')

#             # Foreground mask (Otsu threshold)
#             otsu_threshold = threshold_otsu(illumination_profiles[c])
#             foreground_mask = illumination_profiles[c] > otsu_threshold
#             sns.heatmap(foreground_mask, cmap='gray', cbar=False, ax=axes[1])
#             axes[1].set_title(f'Foreground Mask (Otsu) - Channel {c}')
#             axes[1].axis('off')

#             # Corrected smoothed profile
#             sns.heatmap(corrected_profiles[c], cmap='hot', cbar=True, ax=axes[2])
#             axes[2].set_title(f'Corrected Illumination Profile - Channel {c}')
#             axes[2].axis('off')

#             plt.tight_layout()
#             plt.show()

#         print("Visualization complete.")

