from bigfish import stack, detection, multistack, plot
from magicgui import magicgui
import pandas as pd
import os
import numpy as np
import re
import napari
from typing import Union

from AngelFISH.src import load_data


def identify_spots(receipt, step_name:str, new_params:dict=None, gui:bool=False):
    # required updates to the receipt
    if step_name not in receipt['step_order']:
        receipt['step_order'].append(step_name)
    if step_name not in receipt['steps'].keys():
        receipt['steps'][step_name] = {}
    receipt['steps'][step_name]['task_name'] = 'identify_spots' # make sure to change this for different steps 
    if new_params:
        for k, v in new_params.items():
            receipt['steps'][step_name][k] = v

    # Load Data and Args
    data = load_data(receipt)
    args = receipt['steps'][step_name]
    spot_name = args['spot_name']

    metadata = data['metadata']
    voxel_size_yx = metadata(p=0, t=0)['PixelSizeUm'] * 1000
    voxel_size_z = np.abs(metadata(p=0, t=0, z=1)['ZPosition_um_Intended'] - metadata(p=0, t=0, z=0)['ZPosition_um_Intended']) * 1000

    # Preallocate memmory
    temp_dir = os.path.join(receipt['dirs']['analysis_dir'], f'{spot_name}_CanidentSpots')
    os.makedirs(temp_dir, exist_ok=True)
    thresholds = {}

    def run(receipt, data, p_range=None, t_range=None):
        args = receipt['steps'][step_name]
        # Extract args that will change  
        spot_yx = args['spot_yx']
        spot_z = args['spot_z']
        images = data['images']
        channel = args['channel']
        spot_name = args['spot_name']

        is_3d = images.shape[3] > 1
        voxel_size_nm = (int(voxel_size_z), int(voxel_size_yx), int(voxel_size_yx)) if is_3d else (int(voxel_size_yx), int(voxel_size_yx))
        spot_size_nm = (int(spot_z), int(spot_yx), int(spot_yx)) if is_3d else (int(spot_yx), int(spot_yx))
        threshold = args.get('threshold', None)
        use_log_hook = args.get('use_log_hook', False)
        minimum_distance = args.get('minDistance', None)
        fig_dir = receipt['dirs']['fig_dir']
        display_plots = args.get('display_plots', False)
        min_snr = args.get('min_snr', 0)
        max_snr = args.get('max_snr', np.inf)
        min_signal = args.get('min_signal', 0)
        max_signal = args.get('max_signal', np.inf)
        background_filter_min_z_score = args.get('min_z_score', -np.inf)
        mask_name = args.get('mask_name', None)

        # run image processing
        def run_frame(rna_image, p, t, mask=None):
            if use_log_hook:
                if minimum_distance is None:
                    spot_radius_px = detection.get_object_radius_pixel(
                            voxel_size_nm=voxel_size_nm, 
                            object_radius_nm=spot_size_nm, 
                            ndim=len(rna_image.shape))
                else:
                    spot_radius_px = minimum_distance
            else:
                spot_radius_px = None

            # detect spots
            canidate_spots, individual_thershold = detection.detect_spots(
                                            images=rna_image, 
                                            return_threshold=True, 
                                            threshold=threshold,
                                            voxel_size=voxel_size_nm if not use_log_hook else None,
                                            spot_radius=spot_size_nm if not use_log_hook else None,
                                            log_kernel_size=spot_radius_px if use_log_hook else None,
                                            minimum_distance=spot_radius_px if use_log_hook else None,)
    
            # calc snr and signal 
            snr_spots, signal = compute_snr_spots(
                image=rna_image.astype(np.float64), 
                spots=canidate_spots.astype(np.float64), 
                voxel_size=voxel_size_nm, 
                spot_radius=spot_size_nm if not use_log_hook else spot_radius_px)
            
            # concate results into dataframe
            snr_spots = np.array(snr_spots).reshape(-1, 1)
            signal = np.array(signal).reshape(-1, 1)
            spots_df = pd.DataFrame(np.hstack([canidate_spots, snr_spots, signal]), columns=['z (px)', 'y (px)', 'x (px)', 'snr', 'max signal'] if is_3d else ['y (px)', 'x (px)', 'snr', 'max signal'])
            spots_df['timepoint'] = [t]*len(spots_df)
            spots_df['fov'] = [p]*len(spots_df)
            spots_df['channel'] = [channel]*len(spots_df)
            expermental_metadata = metadata(p=p, t=t, z=0 ,c=channel).get('experimental_metadata', None)
            if expermental_metadata is not None:
                for key, value in expermental_metadata.items():
                    spots_df[key] = [value] * len(spots_df)

            # save data to temp 
            path = os.path.join(temp_dir, f'p{p}_t{t}_{spot_name}_canidateSpotPreFilter.csv')
            spots_df.to_csv(path, index=False)

            # filter canidate spots
            spots_df = spots_df[
                (spots_df['snr'] >= min_snr) &
                (spots_df['snr'] <= max_snr) &
                (spots_df['max signal'] >= min_signal) &
                (spots_df['max signal'] <= max_signal)
            ].reset_index(drop=True)

            # filter based on cell props
            if mask is not None:
                labels = np.unique(mask)

                # Create a mask of the same shape, marking spot areas as True
                spot_coords = spots_df[['z (px)', 'y (px)', 'x (px)']] if is_3d else spots_df[['y (px)', 'x (px)']]
                spot_mask = np.zeros_like(mask, dtype=bool)
                
                # Define a radius around each spot (in pixels) to exclude
                radius_yx = int(spot_yx // voxel_size_yx)
                radius_z = int(spot_z // voxel_size_z) if is_3d else 0

                # Mark region around each spot
                for coord in spot_coords.values:
                    if is_3d:
                        z, y, x = map(int, coord)
                        z_start, z_end = max(0, z - radius_z), min(mask.shape[0], z + radius_z + 1)
                        y_start, y_end = max(0, y - radius_yx), min(mask.shape[1], y + radius_yx + 1)
                        x_start, x_end = max(0, x - radius_yx), min(mask.shape[2], x + radius_yx + 1)
                        spot_mask[z_start:z_end, y_start:y_end, x_start:x_end] = True
                    else:
                        y, x = map(int, coord)
                        y_start, y_end = max(0, y - radius_yx), min(mask.shape[0], y + radius_yx + 1)
                        x_start, x_end = max(0, x - radius_yx), min(mask.shape[1], x + radius_yx + 1)
                        spot_mask[y_start:y_end, x_start:x_end] = True

                mean_background = {}
                std_background = {}
                for l in labels:
                    cell_mask = (mask == l)
                    background_mask = cell_mask & (~spot_mask)
                    if np.any(background_mask):
                        mean_background[l] = np.mean(rna_image[background_mask])
                        std_background[l] = np.std(rna_image[background_mask])
                    else:
                        # Fallback in case no background pixels are left
                        mean_background[l] = 0
                        std_background[l] = 1

                # Label each spot with corresponding mask
                coords = spots_df[['z (px)', 'y (px)', 'x (px)']] if is_3d else spots_df[['y (px)', 'x (px)']]
                if is_3d:
                    spot_labels = [mask[int(c[0]), int(c[1]), int(c[2])] for c in coords.values]
                else:
                    spot_labels = [mask[int(c[0]), int(c[1])] for c in coords.values]
                spots_df['spot_labels'] = spot_labels

                # Filter spots based on z-score relative to spot-excluded background
                spots_df = spots_df[
                    spots_df.apply(
                        lambda row: row['max signal'] >= mean_background[row['spot_labels']] + background_filter_min_z_score * std_background[row['spot_labels']],
                        axis=1
                    )
                ].reset_index(drop=True)

            # add additional information
            spots_df['timepoint'] = [t]*len(spots_df)
            spots_df['fov'] = [p]*len(spots_df)
            spots_df['channel'] = [channel]*len(spots_df)
            expermental_metadata = metadata(p=p, t=t, z=0 ,c=channel).get('experimental_metadata', None)
            if expermental_metadata is not None:
                for key, value in expermental_metadata.items():
                    spots_df[key] = [value] * len(spots_df)

            # save data to temp 
            path = os.path.join(temp_dir, f'p{p}_t{t}_{spot_name}_canidateSpots.csv')
            spots_df.to_csv(path, index=False)

            # diplay plots
            if display_plots:
                plot.plot_elbow(
                    images=rna_image, 
                    voxel_size=voxel_size_nm if not use_log_hook else None, 
                    spot_radius=spot_size_nm if not use_log_hook else None,
                    log_kernel_size=spot_radius_px if use_log_hook else None,
                    minimum_distance=spot_radius_px if use_log_hook else None, 
                    path_output=os.path.join(fig_dir, f'fov{p}_tp{t}_ch{channel}_{spot_name}-elbow.png'))

        # run the data
        for p in range(data['pp']) if p_range is None else p_range:
            for t in range(data['tt']) if t_range is None else t_range:
                if mask_name is not None:
                    mask = data[mask_name][p, t].compute()
                else:
                    mask = None
                rna_image = images[p, t, channel]
                rna_image = np.squeeze(rna_image)
                rna_image = rna_image.compute()
                run_frame(rna_image, p, t, mask)

    def compress_data(save_data:bool=True):
        # Compress Data 
        results_dir = receipt['dirs']['results_dir']
        # List all files in the temp_dir
        all_files = os.listdir(temp_dir)
        def match_files_and_save(pattern):
            # Use regex to find matching files
            re_pattern = re.compile(rf'^p\d+_t\d+_{re.escape(spot_name)}_{pattern}\.csv$')
            files = [os.path.join(temp_dir, f) for f in all_files if re_pattern.match(f)]
            df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None
            # write data
            if save_data:
                if df is not None:
                    df.to_csv(os.path.join(results_dir ,f'{spot_name}_{pattern}.csv'), index=False)
            return df
        canidateSpots = match_files_and_save('canidateSpots')
        canidateSpotPreFilter = match_files_and_save('canidateSpotPreFilter')
        
        return canidateSpots, canidateSpotPreFilter

    def release_memory():
        all_files = os.listdir(temp_dir)
        # Delete all files in the temp_dir
        for f in all_files:
            file_path = os.path.join(temp_dir, f)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"[Error] deleting file {file_path}: {e}")

        # Delete the temp_dir itself
        try:
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"[Error] deleting directory {temp_dir}: {e}")

    if not gui:
        run(receipt, data)
        compress_data()
        release_memory()
    else:
        viewer = napari.Viewer()
        viewer.add_image(data['images'], name="images", axis_labels=('p', 't', 'c', 'z', 'y', 'x'),
                        scale=[1, 1, 1, voxel_size_z/voxel_size_yx, 1, 1])

        @magicgui(
                call_button='Run',
                spot_z={'min': 0, 'max': 1e6, 'step': 1.0}  
        )
        def interface( 
            Channel: int = args.get('channel', 0),  
            nucChannel: int = args.get('nucChannel', 0),
            spot_yx: int = args.get('spot_yx', 130), 
            spot_z: int = args.get('spot_z', 500),  
            threshold: Union[int, str] = args.get('threshold', None), 
            use_log_hook: bool = args.get('use_log_hook', False),  
            display_plots: bool = args.get('display_plots', False), 
            minDistance: float = args.get('minDistance', None),
            min_snr: float = args.get('min_snr', 0),
            max_snr: float = args.get('max_snr', 1e6),
            min_signal: float = args.get('min_signal', 0),
            max_signal: float = args.get('max_signal', 1e6),
            min_z_score: float = args.get('min_z_score', -1e6),
            mask_name: str = args.get('mask_name', '')
            ):
            try:
                receipt['steps'][step_name]['spot_name'] = spot_name
                receipt['steps'][step_name]['channel'] = Channel
                receipt['steps'][step_name]['nucChannel'] = nucChannel
                receipt['steps'][step_name]['spot_yx'] = spot_yx
                receipt['steps'][step_name]['spot_z'] = spot_z
                receipt['steps'][step_name]['threshold'] = threshold
                receipt['steps'][step_name]['use_log_hook'] = use_log_hook
                receipt['steps'][step_name]['display_plots'] = display_plots
                receipt['steps'][step_name]['minDistance'] = minDistance
                receipt['steps'][step_name]['min_snr'] = min_snr
                receipt['steps'][step_name]['max_snr'] = max_snr
                receipt['steps'][step_name]['min_signal'] = min_signal
                receipt['steps'][step_name]['max_signal'] = max_signal
                receipt['steps'][step_name]['min_z_score'] = min_z_score
                receipt['steps'][step_name]['mask_name'] = mask_name if len(mask_name) > 1 else None

                # Get current p and t from the viewer's dims
                current_p = int(viewer.dims.current_step[0])
                current_t = int(viewer.dims.current_step[1])
                run(receipt, data, [current_p], [current_t])
            except Exception as e:
                print(f"[Error] Exception during spot detection: {e}")
            canidateSpots, canidateSpotPreFilter = compress_data(False)

            for layer_name in ["spots", 'filtered_spots']:
                if layer_name in viewer.layers:
                    viewer.layers.remove(layer_name)

            # Show spots
            if canidateSpots is not None:
                coords = canidateSpots[["fov", "timepoint", "channel", "z (px)", "y (px)", "x (px)"]].values
                viewer.add_points(
                    coords,
                    name="spots",
                    size=5,
                    face_color="red",
                    scale=[1, 1, 1, voxel_size_z/voxel_size_yx, 1, 1]
                )
                viewer.layers['spots'].refresh()
            else:
                print('no spots found')

            if canidateSpotPreFilter is not None:
                # Find entries in canidateSpotPreFilter that are not in canidateSpots
                # Drop based on spot coordinates and metadata columns
                key_cols = ["fov", "timepoint", "channel", "z (px)", "y (px)", "x (px)"]
                diff = pd.merge(
                    canidateSpotPreFilter,
                    canidateSpots[key_cols].drop_duplicates(),
                    on=key_cols,
                    how="left",
                    indicator=True
                )
                diff = diff[diff["_merge"] == "left_only"].drop(columns=["_merge"])
                if not diff.empty:
                    coords = diff[["fov", "timepoint", "channel", "z (px)", "y (px)", "x (px)"]].values
                    viewer.add_points(
                        coords,
                        name="filtered_spots",
                        size=5,
                        face_color="blue",
                        scale=[1, 1, 1, voxel_size_z/voxel_size_yx, 1, 1]
                    )
                    viewer.layers['filtered_spots'].refresh()
                else:
                    print('no filtered spots found')

        viewer.window.add_dock_widget(interface, area='right')

        def on_destroyed(obj=None):
            print('cleaning up')
            run(receipt, data)
            compress_data()
            release_memory()

        viewer.window._qt_window.destroyed.connect(on_destroyed)

    return receipt



























def compute_snr_spots(image, spots, voxel_size, spot_radius, display_plots: bool = False):
    """Compute signal-to-noise ratio (SNR) based on spot coordinates.

    .. math::

        \\mbox{SNR} = \\frac{\\mbox{max(spot signal)} -
        \\mbox{mean(background)}}{\\mbox{std(background)}}

    Background is a region twice larger surrounding the spot region. Only the
    y and x dimensions are taking into account to compute the SNR.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    spots : np.ndarray
        Coordinate of the spots, with shape (nb_spots, 3) or (nb_spots, 2).
        One coordinate per dimension (zyx or yx coordinates).
    voxel_size : int, float, Tuple(int, float), List(int, float) or None
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions. Not used if 'log_kernel_size' and 'minimum_distance' are
        provided.
    spot_radius : int, float, Tuple(int, float), List(int, float) or None
        Radius of the spot, in nanometer. One value per spatial dimension (zyx
        or yx dimensions). If it's a scalar, the same radius is applied to
        every dimensions. Not used if 'log_kernel_size' and 'minimum_distance'
        are provided.

    Returns
    -------
    snr : float
        Median signal-to-noise ratio computed for every spots.

    """
    # check parameters
    stack.check_array(
        image,
        ndim=[2, 3],
        dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_range_value(image, min_=0)
    stack.check_array(
        spots,
        ndim=2,
        dtype=[np.float32, np.float64, np.int32, np.int64])
    stack.check_parameter(
        voxel_size=(int, float, tuple, list),
        spot_radius=(int, float, tuple, list))

    # check consistency between parameters
    ndim = image.ndim
    if ndim != spots.shape[1]:
        raise ValueError("Provided image has {0} dimensions but spots are "
                         "detected in {1} dimensions."
                         .format(ndim, spots.shape[1]))
    if isinstance(voxel_size, (tuple, list)):
        if len(voxel_size) != ndim:
            raise ValueError(
                "'voxel_size' must be a scalar or a sequence with {0} "
                "elements.".format(ndim))
    else:
        voxel_size = (voxel_size,) * ndim
    if isinstance(spot_radius, (tuple, list)):
        if len(spot_radius) != ndim:
            raise ValueError(
                "'spot_radius' must be a scalar or a sequence with {0} "
                "elements.".format(ndim))
    else:
        spot_radius = (spot_radius,) * ndim

    # cast spots coordinates if needed
    if spots.dtype == np.float64:
        spots = np.round(spots).astype(np.int64)

    # cast image if needed
    image_to_process = image.copy().astype(np.float64)

    # clip coordinate if needed
    if ndim == 3:
        spots[:, 0] = np.clip(spots[:, 0], 0, image_to_process.shape[0] - 1)
        spots[:, 1] = np.clip(spots[:, 1], 0, image_to_process.shape[1] - 1)
        spots[:, 2] = np.clip(spots[:, 2], 0, image_to_process.shape[2] - 1)
    else:
        spots[:, 0] = np.clip(spots[:, 0], 0, image_to_process.shape[0] - 1)
        spots[:, 1] = np.clip(spots[:, 1], 0, image_to_process.shape[1] - 1)

    # compute radius used to crop spot image
    radius_pixel = detection.get_object_radius_pixel(
        voxel_size_nm=voxel_size,
        object_radius_nm=spot_radius,
        ndim=ndim)
    radius_signal_ = [np.sqrt(ndim) * r for r in radius_pixel]
    radius_signal_ = tuple(radius_signal_)

    # compute the neighbourhood radius
    radius_background_ = tuple(i * 2 for i in radius_signal_)

    # ceil radii
    radius_signal = np.ceil(radius_signal_).astype(np.uint16)
    radius_background = np.ceil(radius_background_).astype(np.uint16)

    snr_spots = []
    max_signals = []

    # Loop over each spot
    for spot in spots:
        # Extract spot coordinates
        spot_y = spot[ndim - 2]
        spot_x = spot[ndim - 1]
        radius_signal_yx = radius_signal[-1]
        radius_background_yx = radius_background[-1]
        edge_background_yx = radius_background_yx - radius_signal_yx

        if ndim == 3:
            spot_z = spot[0]
            radius_background_z = radius_background[0]
            # Compute max signal for the current spot
            max_signal = image_to_process[spot_z, spot_y, spot_x]
            # Extract background region
            spot_background_, _ = detection.get_spot_volume(
                image_to_process, spot_z, spot_y, spot_x,
                radius_background_z, radius_background_yx)
            spot_background = spot_background_.copy()

            # Remove signal region from the background crop
            spot_background[:, edge_background_yx:-edge_background_yx, edge_background_yx:-edge_background_yx] = -1
            spot_background = spot_background[spot_background >= 0]

        else:
            # For 2D images
            max_signal = image_to_process[spot_y, spot_x]
            spot_background_, _ = detection.get_spot_surface(
                image_to_process, spot_y, spot_x, radius_background_yx)
            spot_background = spot_background_.copy()

            # Remove signal region from the background crop
            spot_background[edge_background_yx:-edge_background_yx, edge_background_yx:-edge_background_yx] = -1
            spot_background = spot_background[spot_background >= 0]

        # Compute mean and standard deviation of the background
        mean_background = np.mean(spot_background)
        std_background = np.std(spot_background)

        # Compute SNR
        snr = (max_signal - mean_background) / (std_background + 1e-8)  # Add small value to avoid division by zero
        snr_spots.append(snr)
        max_signals.append(max_signal)

    # Return both SNRs and max_signals for further analysis
    return snr_spots, max_signals