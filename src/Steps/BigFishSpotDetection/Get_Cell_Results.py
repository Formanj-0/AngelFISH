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

        # run image processing
        def run_frame(rna_image, p, t):
            if ((nuc_mask is not None and np.max(nuc_mask) != 0) or (cell_mask is not None and np.max(cell_mask) != 0)):
                #### Extract cell level results
                if nucChannel is not None:
                    nuc = image[nucChannel, :, :, :].squeeze()
                rna = image[FISHChannel, :, :, :].squeeze()
                # convert masks to max projection
                if nuc_mask is not None and len(nuc_mask.shape) != 2:
                    nuc_mask = np.max(nuc_mask, axis=0)
                if cell_mask is not None and len(cell_mask.shape) != 2:
                    cell_mask = np.max(cell_mask, axis=0)

                ndim = 2
                if len(rna.shape) == 3:
                    ndim = 3
                    rna = np.max(rna, axis=0)

                # convert types
                nuc_mask = nuc_mask.squeeze().astype("uint16") if nuc_mask is not None else None
                cell_mask = cell_mask.squeeze().astype("uint16") if cell_mask is not None else None

                # remove transcription sites
                if nuc_mask is not None:
                    spots_no_ts, foci, ts = multistack.remove_transcription_site(spots, clusters, nuc_mask, ndim=3)
                    if verbose:
                        print("detected spots (without transcription sites)")
                        print("\r shape: {0}".format(spots_no_ts.shape))
                        print("\r dtype: {0}".format(spots_no_ts.dtype))
                else:
                    spots_no_ts, foci, ts = None, None, None

                # get spots inside and outside nuclei
                if nuc_mask is not None:
                    spots_in, spots_out = multistack.identify_objects_in_region(nuc_mask, spots, ndim=3)
                    if verbose:
                        print("detected spots (inside nuclei)")
                        print("\r shape: {0}".format(spots_in.shape))
                        print("\r dtype: {0}".format(spots_in.dtype), "\n")
                        print("detected spots (outside nuclei)")
                        print("\r shape: {0}".format(spots_out.shape))
                        print("\r dtype: {0}".format(spots_out.dtype))
                else:
                    spots_in, spots_out = None, None

                # extract fov results
                cell_mask = cell_mask.astype("uint16") if cell_mask is not None else None # nuc_mask.astype("uint16")
                nuc_mask = nuc_mask.astype("uint16") if nuc_mask is not None else None
                if cell_mask is None:
                    cell_mask = nuc_mask
                rna = rna.astype("uint16")
                other_images = {}
                other_images["dapi"] = np.max(nuc, axis=0).astype("uint16") if nuc is not None else None

                fov_results = multistack.extract_cell( # this function is incredibly poorly written be careful looking at it
                    cell_label=cell_mask,
                    ndim=ndim,
                    nuc_label=nuc_mask,
                    rna_coord=spots_no_ts,
                    others_coord={"foci": foci, "transcription_site": ts},
                    image=rna,
                    others_image=other_images,)
                if verbose:
                    print("number of cells identified: {0}".format(len(fov_results)))

                cell_label = cell_mask
                nuc_label = nuc_mask
                # cycle through cells and save the results
                for i, cell_results in enumerate(fov_results):
                    # get cell results
                    cell_mask = cell_results["cell_mask"]
                    cell_coord = cell_results["cell_coord"]
                    nuc_mask = cell_results["nuc_mask"]
                    nuc_coord = cell_results["nuc_coord"]
                    rna_coord = cell_results["rna_coord"]
                    foci_coord = cell_results["foci"]
                    ts_coord = cell_results["transcription_site"]
                    image_contrasted = cell_results["image"]
                    
                    if verbose:
                        print("cell {0}".format(i))
                        print("\r number of rna {0}".format(len(rna_coord)))
                        print("\r number of foci {0}".format(len(foci_coord)))
                        print("\r number of transcription sites {0}".format(len(ts_coord)))

                    # plot individual cells
                    if display_plots:
                        plot.plot_cell(
                            ndim=3, cell_coord=cell_coord, nuc_coord=nuc_coord,
                            rna_coord=rna_coord, foci_coord=foci_coord, other_coord=ts_coord,
                            image=image_contrasted, cell_mask=cell_mask, nuc_mask=nuc_mask, rescale=True, contrast=True,
                            title="Cell {0}".format(i),
                            path_output=os.path.join(kwargs['fig_dir'], f'fov{fov}_tp{timepoint}_ch{FISHChannel}_cell_{i}.png'))

                df = multistack.summarize_extraction_results(fov_results, ndim=3)
                df['fov'] = [fov]*len(df)
                df['timepoint'] = [timepoint]*len(df)
                df['FISH_Channel'] = [FISHChannel]*len(df)

                ##### Update spots to include cells
                c_list = [cell_label[s[1], s[2]] for s in spots] # TODO make this work for and 3D
                n_list = [nuc_label[s[1], s[2]] for s in spots]
                is_nuc = [(n>0 and c>0) for n,c in zip(n_list,c_list)]
                errors = [(n>0 and c>0 and n!=c) for n,c in zip(n_list,c_list)]
                if any(errors):
                    raise ValueError('Miss matching cell labels')

                # For clusters
                c_list_clusters = [cell_label[s[1], s[2]] for s in clusters] # TODO make this work for and 3D
                n_list_clusters = [nuc_label[s[1], s[2]] for s in clusters]
                is_nuc_clusters = [(n>0 and c>0) for n,c in zip(n_list_clusters,c_list_clusters)]
                errors_clusters = [(n>0 and c>0 and n!=c) for n,c in zip(n_list_clusters,c_list_clusters)]
                if any(errors_clusters):
                    raise ValueError('Miss matching cell labels in clusters')
                
                clusters = np.hstack([clusters, np.array(is_nuc_clusters).reshape(-1, 1), np.array(c_list_clusters).reshape(-1, 1), np.array(n_list_clusters).reshape(-1, 1)])
                spots = np.hstack([spots, np.array(is_nuc).reshape(-1, 1), np.array(c_list).reshape(-1, 1), np.array(n_list).reshape(-1, 1)])
                

        # run the data
        for p in range(data['pp']) if p_range is None else p_range:
            for t in range(data['tt']) if t_range is None else t_range:
                rna_image = images[p, t, channel]
                rna_image = np.squeeze(rna_image)
                rna_image = rna_image.compute()
                run_frame(rna_image, p, t)

    def compress_data(save_data:bool=True):
        # Compress Data 
        results_dir = receipt['dirs']['results_dir']
        # List all files in the temp_dir
        all_files = os.listdir(temp_dir)
        # Use regex to find matching files
        pattern = re.compile(r'.*_canidateSpots\.csv$')
        files = [os.path.join(temp_dir, f) for f in all_files if pattern.match(f)]
        final_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None
        # write data
        if save_data:
            if final_df is not None:
                final_df.to_csv(os.path.join(results_dir ,f'{spot_name}_canidateSpots.csv'), index=False)
        return final_df

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
                call_button='Run'
        )
        def interface( 
            Channel: int = args.get('channel', 0),  
            nucChannel: int = args.get('nucChannel', 0),
            spot_yx: float = args.get('spot_yx', 1.0), 
            spot_z: float = args.get('spot_z', 1.0),  
            threshold: Union[int, str] = args.get('threshold', None), 
            use_log_hook: bool = args.get('use_log_hook', False),  
            display_plots: bool = args.get('display_plots', False), 
            minDistance: float = args.get('minDistance', None),
            min_snr: float = args.get('min_snr', 0),
            max_snr: float = args.get('max_snr', np.inf),
            min_signal: float = args.get('min_signal', 0),
            max_signal: float = args.get('max_signal', np.inf),
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
                # Get current p and t from the viewer's dims
                current_p = int(viewer.dims.current_step[0])
                current_t = int(viewer.dims.current_step[1])
                run(receipt, data, [current_p], [current_t])
            except Exception as e:
                print(f"[Error] Exception during spot detection: {e}")
            final_df = compress_data(False)

            for layer_name in ["spots"]:
                if layer_name in viewer.layers:
                    viewer.layers.remove(layer_name)

            # Show spots
            if final_df is not None:
                coords = final_df[["fov", "timepoint", "channel", "z (px)", "y (px)", "x (px)"]].values
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

        viewer.window.add_dock_widget(interface, area='right')

        def on_destroyed(obj=None):
            print('cleaning up')
            compress_data()
            release_memory()

        viewer.window._qt_window.destroyed.connect(on_destroyed)

    return receipt









