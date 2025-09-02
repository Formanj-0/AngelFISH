from bigfish import stack, detection, multistack, plot
from magicgui import magicgui
import pandas as pd
import os
import numpy as np
import re
import napari
from typing import Union

from AngelFISH.src import load_data


def decompose_dense_regions(receipt, step_name:str, new_params:dict=None, gui:bool=False):
    # required updates to the receipt
    if step_name not in receipt['step_order']:
        receipt['step_order'].append(step_name)
    if step_name not in receipt['steps'].keys():
        receipt['steps'][step_name] = {}
    receipt['steps'][step_name]['task_name'] = 'decompose_dense_regions'  # make sure to change this for different steps 
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
    temp_dir = os.path.join(receipt['dirs']['analysis_dir'], spot_name)
    os.makedirs(temp_dir, exist_ok=True)

    def run(receipt, data, p_range=None, t_range=None):
        args = receipt['steps'][step_name]
        # Extract args that will change  
        spot_yx = args['spot_yx']
        spot_z = args['spot_z']
        images = data['images']
        channel = args['channel']
        spot_name = args['spot_name']
        alpha = args['alpha']
        beta = args['beta']
        gamma = args['gamma']
        cluster_radius = args['cluster_radius']
        min_num_spot_for_cluster = args['min_num_spot_for_cluster']
        is_3d = images.shape[3] > 1
        voxel_size_nm = (int(voxel_size_z), int(voxel_size_yx), int(voxel_size_yx)) if is_3d else (int(voxel_size_yx), int(voxel_size_yx))
        spot_size_nm = (int(spot_z), int(spot_yx), int(spot_yx)) if is_3d else (int(spot_yx), int(spot_yx))
        fig_dir = receipt['dirs']['fig_dir']
        display_plots = args.get('display_plots', False)
        use_log_hook = args.get('use_log_hook', False)
        minimum_distance = args.get('minDistance', None)


        # run image processing
        def run_frame(rna_image, p, t):
            canidate_spots = data[f'{spot_name}_canidateSpots']
            canidate_spots = canidate_spots[(canidate_spots['fov'] == p) & (canidate_spots['timepoint'] == t)]
            canidate_spots = canidate_spots[['z (px)','y (px)','x (px)'] if is_3d else ['y (px)','x (px)']].values

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

            spots_post_decomposition, dense_regions, reference_spot = detection.decompose_dense(
                                    image=rna_image, 
                                    spots=canidate_spots, 
                                    voxel_size=voxel_size_nm, 
                                    spot_radius=spot_size_nm if not use_log_hook else spot_radius_px,
                                    alpha=alpha,
                                    beta=beta,
                                    gamma=gamma)
            
            spots_post_clustering, clusters = detection.detect_clusters(
                                                spots=spots_post_decomposition, 
                                                voxel_size=voxel_size_nm, 
                                                radius=int(cluster_radius), 
                                                nb_min_spots=min_num_spot_for_cluster)

            # convert to dataframes
            spots_post_decomposition = pd.DataFrame(spots_post_decomposition, columns=['z (px)', 'y (px)', 'x (px)'] if is_3d else ['y (px)', 'x (px)'])
            dense_regions = pd.DataFrame(dense_regions, columns=['z (px)', 'y (px)', 'x (px)', 'nb_rna', 'area', 'avg_intensity', 'index'] if is_3d else ['y (px)', 'x (px)', 'nb_rna', 'area', 'avg_intensity', 'index'])
            spots_post_clustering = pd.DataFrame(spots_post_clustering, columns=['z (px)', 'y (px)', 'x (px)', 'cluster index'] if is_3d else ['y (px)', 'x (px)', 'cluster index'])
            clusters = pd.DataFrame(clusters, columns=['z (px)', 'y (px)', 'x (px)', 'nb_rna', 'cluster index'] if is_3d else ['y (px)', 'x (px)', 'nb_rna', 'cluster index'])

            # add metadata
            def add_metadata_to_df(df):
                df['timepoint'] = [t]*len(df)
                df['fov'] = [p]*len(df)
                df['channel'] = [channel]*len(df)
                expermental_metadata = metadata(p=p, t=t, z=0 ,c=channel).get('experimental_metadata', None)
                if expermental_metadata is not None:
                    for key, value in expermental_metadata.items():
                        df[key] = [value] * len(df)
                return df

            dense_regions = add_metadata_to_df(dense_regions)
            spots_post_decomposition = add_metadata_to_df(spots_post_decomposition)
            clusters = add_metadata_to_df(clusters)
            spots_post_clustering = add_metadata_to_df(spots_post_clustering)

            # save data to temp 
            path = os.path.join(temp_dir, f'p{p}_t{t}_{spot_name}_denseRegions.csv')
            dense_regions.to_csv(path, index=False)

            path = os.path.join(temp_dir, f'p{p}_t{t}_{spot_name}_spotsPostDecomposition.csv')
            spots_post_decomposition.to_csv(path, index=False)

            path = os.path.join(temp_dir, f'p{p}_t{t}_{spot_name}_clusters.csv')
            clusters.to_csv(path, index=False)

            path = os.path.join(temp_dir, f'p{p}_t{t}_{spot_name}_spotsPostClustering.csv')
            spots_post_clustering.to_csv(path, index=False)

            if display_plots:
                plot.plot_reference_spot(reference_spot, 
                            rescale=True, 
                            path_output=os.path.join(fig_dir, f'fov{p}_tp{t}_ch{channel}-ref_spot.png'))  

        # run the data
        for p in range(data['pp']) if p_range is None else p_range:
            for t in range(data['tt']) if t_range is None else t_range:
                rna_image = images[p, t, channel]
                rna_image = np.squeeze(rna_image)
                rna_image = rna_image.compute()
                run_frame(rna_image, p, t)

    def compress_data(save_data:bool=True):
        results_dir = receipt['dirs']['results_dir']
        # List all files in the temp_dir
        all_files = os.listdir(temp_dir)
        def match_files_and_save(pattern):
            # Use regex to find matching files
            re_pattern = re.compile(rf'.*_{pattern}\.csv$')
            files = [os.path.join(temp_dir, f) for f in all_files if re_pattern.match(f)]
            df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None
            # write data
            if save_data:
                if df is not None:
                    df.to_csv(os.path.join(results_dir ,f'{spot_name}_{pattern}.csv'), index=False)
            return df

        spotsPostDecomposition = match_files_and_save('spotsPostDecomposition')
        denseRegions = match_files_and_save('denseRegions')
        spotsPostClustering = match_files_and_save('spotsPostClustering')
        clusters = match_files_and_save('clusters')

        return spotsPostDecomposition, denseRegions, spotsPostClustering, clusters

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
                alpha: float = args.get('alpha', 0.7),
                beta: float = args.get('beta', 1),
                gamma: float = args.get('gamma', 5),
                cluster_radius: int = args.get('cluster_radius', 500),
                min_num_spot_for_cluster: int = args.get('min_num_spot_for_cluster', 4)):
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
                receipt['steps'][step_name]['alpha'] = alpha
                receipt['steps'][step_name]['beta'] = beta
                receipt['steps'][step_name]['gamma'] = gamma
                receipt['steps'][step_name]['cluster_radius'] = cluster_radius
                receipt['steps'][step_name]['min_num_spot_for_cluster'] = min_num_spot_for_cluster
                current_p = int(viewer.dims.current_step[0])
                current_t = int(viewer.dims.current_step[1])
                run(receipt, data, [current_p], [current_t])
            except Exception as e:
                print(f"[Error] Exception during decomposition: {e}")
            spotsPostDecomposition, denseRegions, spotsPostClustering, clusters = compress_data(False)

            for layer_name in ["spots_post_decomposition", 'dense_regions', 'spots_post_clustering', 'clusters']:
                if layer_name in viewer.layers:
                    viewer.layers.remove(layer_name)

            # Show spots
            if spotsPostDecomposition is not None:
                coords = spotsPostDecomposition[["fov", "timepoint", "channel", "z (px)", "y (px)", "x (px)"]].values
                viewer.add_points(
                    coords,
                    name="spots_post_decomposition",
                    size=5,
                    face_color="red",
                    scale=[1, 1, 1, voxel_size_z/voxel_size_yx, 1, 1]
                )
                viewer.layers['spots_post_decomposition'].refresh()
            else:
                print('no spots_post_decomposition found')

            if denseRegions is not None:
                coords = denseRegions[["fov", "timepoint", "channel", "z (px)", "y (px)", "x (px)"]].values
                viewer.add_points(
                    coords,
                    name="dense_regions",
                    size=5,
                    face_color="green",
                    scale=[1, 1, 1, voxel_size_z/voxel_size_yx, 1, 1]
                )
                viewer.layers['dense_regions'].refresh()
            else:
                print('no dense_regions found')

            if spotsPostClustering is not None:
                coords = spotsPostClustering[["fov", "timepoint", "channel", "z (px)", "y (px)", "x (px)"]].values
                viewer.add_points(
                    coords,
                    name="spots_post_clustering",
                    size=5,
                    face_color="blue",
                    scale=[1, 1, 1, voxel_size_z/voxel_size_yx, 1, 1]
                )
                viewer.layers['spots_post_clustering'].refresh()
            else:
                print('no spots_post_clustering found')

            if clusters is not None:
                coords = clusters[["fov", "timepoint", "channel", "z (px)", "y (px)", "x (px)"]].values
                viewer.add_points(
                    coords,
                    name="clusters",
                    size=5,
                    face_color="yellow",
                    scale=[1, 1, 1, voxel_size_z/voxel_size_yx, 1, 1]
                )
                viewer.layers['clusters'].refresh()
            else:
                print('no clusters found')

        viewer.window.add_dock_widget(interface, area='right')

        def on_destroyed(obj=None):
            print('cleaning up')
            compress_data()
            release_memory()

        viewer.window._qt_window.destroyed.connect(on_destroyed)

    return receipt


















