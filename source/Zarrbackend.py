import xarray as xr
from pathlib import Path
import numpy as np

class ZarrImageDataset:
    def __init__(self, zarr_path):
        self._zarr_path = Path(zarr_path)
        self._ds = None
        self._loaded = False

    @property
    def zarr_path(self):
        return self._zarr_path

    @zarr_path.setter
    def zarr_path(self, new_path):
        self._zarr_path = Path(new_path)
        self._loaded = False  # force reload on next access

    @property
    def dataset(self):
        """Lazily load the dataset if not already loaded."""
        if not self._loaded:
            self._ds = xr.open_zarr(self._zarr_path, chunks="auto")
            self._loaded = True
        return self._ds

    @property
    def variables(self):
        return list(self.dataset.data_vars)

    @property
    def metadata(self):
        return self.dataset.attrs

    @metadata.setter
    def metadata(self, new_metadata: dict):
        self._ds.attrs.update(new_metadata)
        # Save metadata back to disk
        self._ds.to_zarr(self._zarr_path, mode="w")

    def get_image(self, time_index=0, variable="big_images"):
        return self.dataset[variable].isel(time=time_index)

    def get_slice(self, time_range=slice(0, 10), variable="big_images"):
        return self.dataset[variable].isel(time=time_range)

    def compute_image(self, time_index=0, variable="big_images"):
        return self.get_image(time_index, variable).compute()

    def save_processed(self, dataarray, name="processed", out_path="zarr_output/processed.zarr"):
        dataarray.to_dataset(name=name).to_zarr(out_path, mode="w")





if __name__ == '__main__':
    zimg = ZarrImageDataset("zarr_output/big_image_stack.zarr")

    # Access metadata
    print(zimg.metadata)

    # Set metadata
    zimg.metadata = {"project": "neuro123", "date": "2025-04-27"}

    # List variables
    print(zimg.variables)

    # Get one image lazily
    lazy_img = zimg.get_image(5)

    # Compute it into memory
    img_np = zimg.compute_image(5)

    # Get a slice of images
    slice_stack = zimg.get_slice(slice(10, 20))

    # Compute mean and save
    mean = slice_stack.mean(dim="time")
    zimg.save_processed(mean, name="mean_over_time")
