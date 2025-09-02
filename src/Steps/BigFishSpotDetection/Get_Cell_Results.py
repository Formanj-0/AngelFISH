from bigfish import stack, detection, multistack, plot
from magicgui import magicgui
import pandas as pd
import os
import numpy as np
import re
import napari
from typing import Union

from AngelFISH.src import load_data


def get_cell_counts(receipt, step_name:str, new_params:dict=None, gui:bool=False):
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
    temp_dir = os.path.join(receipt['dirs']['analysis_dir'], f'{spot_name}_cellCounts')
    os.makedirs(temp_dir, exist_ok=True)

    def run(receipt, data, p_range=None, t_range=None):
        args = receipt['steps'][step_name]
        # Extract args that will change  
        images = data['images']
        channel = args['channel']
        spot_name = args['spot_name']
        is_3d = images.shape[3] > 1
        nuc_channel = args.get('nuc_channel', None)
        nuc_masks = data.get('nuc_masks', None)
        cyto_masks = data.get('cyto_masks', None)

        fig_dir = receipt['dirs']['fig_dir']
        display_plots = args.get('display_plots', False)

        # run image processing
        def run_frame(rna_image, nuc_image, p, t, nuc_mask, cell_mask):
            if ((nuc_mask is not None and np.max(nuc_mask) != 0) or (cell_mask is not None and np.max(cell_mask) != 0)):
                spots = data[f'{spot_name}_spotsPostClustering']
                spots = spots[(spots['fov'] == p) & (spots['timepoint'] == t)]
                spots = spots[['z (px)','y (px)','x (px)', 'cluster index'] if is_3d else ['y (px)','x (px)', 'cluster index']].values

                clusters = data[f'{spot_name}_clusters']
                clusters = clusters[(clusters['fov'] == p) & (clusters['timepoint'] == t)]
                clusters = clusters[['z (px)','y (px)','x (px)', 'nb_rna', 'cluster index'] if is_3d else ['y (px)','x (px)', 'nb_rna', 'cluster index']].values

                # convert masks to max projection
                if nuc_mask is not None and len(nuc_mask.shape) != 2:
                    nuc_mask = np.max(nuc_mask, axis=0)
                if cell_mask is not None and len(cell_mask.shape) != 2:
                    cell_mask = np.max(cell_mask, axis=0)

                ndim = 2
                if is_3d:
                    ndim = 3
                    rna_image = np.max(rna_image, axis=0)

                # convert types
                nuc_mask = nuc_mask.squeeze().astype("uint16") if nuc_mask is not None else None
                cell_mask = cell_mask.squeeze().astype("uint16") if cell_mask is not None else None

                # remove transcription sites
                if nuc_mask is not None:
                    spots_no_ts, foci, ts = multistack.remove_transcription_site(spots, clusters, nuc_mask, ndim=3 if is_3d else 2)
                else:
                    spots_no_ts, foci, ts = None, None, None

                # get spots inside and outside nuclei
                if nuc_mask is not None:
                    spots_in, spots_out = multistack.identify_objects_in_region(nuc_mask, spots, ndim=3)
                else:
                    spots_in, spots_out = None, None

                # extract fov results
                cell_mask = cell_mask.astype("uint16") if cell_mask is not None else None # nuc_mask.astype("uint16")
                nuc_mask = nuc_mask.astype("uint16") if nuc_mask is not None else None
                if cell_mask is None:
                    cell_mask = nuc_mask
                rna_image = rna_image.astype("uint16")
                other_images = {}
                other_images["dapi"] = np.max(nuc_image, axis=0).astype("uint16") if is_3d else nuc_image

                fov_results = multistack.extract_cell( # this function is incredibly poorly written be careful looking at it
                    cell_label=cell_mask,
                    ndim=ndim,
                    nuc_label=nuc_mask,
                    rna_coord=spots_no_ts,
                    others_coord={"foci": foci, "transcription_site": ts},
                    image=rna_image,
                    others_image=other_images,)

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

                    # plot individual cells
                    if display_plots:
                        plot.plot_cell(
                            ndim=3, cell_coord=cell_coord, nuc_coord=nuc_coord,
                            rna_coord=rna_coord, foci_coord=foci_coord, other_coord=ts_coord,
                            image=image_contrasted, cell_mask=cell_mask, nuc_mask=nuc_mask, rescale=True, contrast=True,
                            title="Cell {0}".format(i),
                            path_output=os.path.join(fig_dir, f'fov{p}_tp{t}_ch{channel}_cell_{i}.png'))

                cell_counts = multistack.summarize_extraction_results(fov_results, ndim=3 if is_3d else 2)

                def add_metadata_to_df(df):
                    df['timepoint'] = [t]*len(df)
                    df['fov'] = [p]*len(df)
                    df['channel'] = [channel]*len(df)
                    expermental_metadata = metadata(p=p, t=t, z=0 ,c=channel).get('experimental_metadata', None)
                    if expermental_metadata is not None:
                        for key, value in expermental_metadata.items():
                            df[key] = [value] * len(df)
                    return df

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

                spots = pd.DataFrame(spots, columns=['z (px)','y (px)','x (px)', 'is nuc', 'cell label', 'nuc label'] if is_3d else ['y (px)','x (px)', 'is nuc', 'cell label', 'nuc label'])
                clusters = pd.DataFrame(clusters, columns=['z (px)','y (px)','x (px)', 'num rna', 'cluster index', 'is nuc', 'cell label', 'nuc label'] if is_3d else ['y (px)','x (px)', 'num rna', 'cluster index', 'is nuc', 'cell label', 'nuc label'])
                ts = pd.DataFrame(ts, columns=['z (px)','y (px)','x (px)', 'num rna', 'cluster index'] if is_3d else ['y (px)','x (px)', 'num rna', 'cluster index'])
                foci = pd.DataFrame(foci, columns=['z (px)','y (px)','x (px)', 'num rna', 'cluster index'] if is_3d else ['y (px)','x (px)', 'num rna', 'cluster index'])
                spots_in = pd.DataFrame(spots_in, columns=['z (px)','y (px)','x (px)'] if is_3d else ['y (px)','x (px)'])
                spots_out = pd.DataFrame(spots_out, columns=['z (px)','y (px)','x (px)'] if is_3d else ['y (px)','x (px)'])
                spots_no_ts = pd.DataFrame(spots_no_ts, columns=['z (px)','y (px)','x (px)', 'foci index'] if is_3d else ['y (px)','x (px)', 'foci index'])

                cell_counts = add_metadata_to_df(cell_counts)
                spots = add_metadata_to_df(spots)
                clusters = add_metadata_to_df(clusters)
                ts = add_metadata_to_df(ts)
                foci = add_metadata_to_df(foci)
                spots_in = add_metadata_to_df(spots_in)
                spots_out = add_metadata_to_df(spots_out)
                spots_no_ts = add_metadata_to_df(spots_no_ts)

                path = os.path.join(temp_dir, f'p{p}_t{t}_{spot_name}_cellCounts.csv')
                cell_counts.to_csv(path, index=False)
                path = os.path.join(temp_dir, f'p{p}_t{t}_{spot_name}_spots.csv')
                spots.to_csv(path, index=False)
                path = os.path.join(temp_dir, f'p{p}_t{t}_{spot_name}_clusters.csv')
                clusters.to_csv(path, index=False)
                path = os.path.join(temp_dir, f'p{p}_t{t}_{spot_name}_ts.csv')
                ts.to_csv(path, index=False)
                path = os.path.join(temp_dir, f'p{p}_t{t}_{spot_name}_foci.csv')
                foci.to_csv(path, index=False)
                path = os.path.join(temp_dir, f'p{p}_t{t}_{spot_name}_spots_in.csv')
                spots_in.to_csv(path, index=False)
                path = os.path.join(temp_dir, f'p{p}_t{t}_{spot_name}_spots_out.csv')
                spots_out.to_csv(path, index=False)
                path = os.path.join(temp_dir, f'p{p}_t{t}_{spot_name}_spots_no_ts.csv')
                spots_no_ts.to_csv(path, index=False)

        # run the data
        for p in range(data['pp']) if p_range is None else p_range:
            for t in range(data['tt']) if t_range is None else t_range:
                rna_image = images[p, t, channel]
                rna_image = np.squeeze(rna_image)
                rna_image = rna_image.compute()
                if nuc_channel:
                    nuc_image = images[p, t, nuc_channel].compute()
                else:
                    nuc_image = None
                if nuc_masks is not None:
                    nuc_mask = nuc_masks[p, t].compute()
                else:
                    nuc_mask = None
                if cyto_masks is not None:
                    cyto_mask = cyto_masks[p,t].compute()
                else:
                    cyto_mask = None
                run_frame(rna_image, nuc_image, p, t, nuc_mask, cyto_mask)

    def compress_data(save_data:bool=True):
        # Compress Data 
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

        cellCounts = match_files_and_save('cellCounts')
        spots = match_files_and_save('spots')
        clusters = match_files_and_save('clusters')
        ts = match_files_and_save('ts')
        foci = match_files_and_save('foci')
        spots_in = match_files_and_save('spots_in')
        spots_out = match_files_and_save('spots_out')
        spots_no_ts = match_files_and_save('spots_no_ts')
        return cellCounts, spots, clusters, ts, foci, spots_in, spots_out, spots_no_ts


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
        def add_dummy_channels(mask):
            """
            Expand a 5D mask (p, t, z, y, x) into 6D (p, t, c, z, y, x) with zeros in extra channels.
            """
            mask = np.array(mask)
            temp = np.zeros_like(data['images'])
            temp[:, :, receipt['steps'][step_name]['channel'], :, :, :] = mask
            return temp
        if data.get('nuc_masks', False):
            nuc_masks = add_dummy_channels(data['nuc_masks'])
            viewer.add_image(nuc_masks, name="nuc masks", axis_labels=('p', 't', 'c', 'z', 'y', 'x'),
                scale=[1, 1, 1, voxel_size_z/voxel_size_yx, 1, 1])
        if data.get('cyto_masks', False):
            cyto_masks = add_dummy_channels(data['cyto_masks'])
            viewer.add_image(cyto_masks, name="cyto masks", axis_labels=('p', 't', 'c', 'z', 'y', 'x'),
                    scale=[1, 1, 1, voxel_size_z/voxel_size_yx, 1, 1])
        @magicgui(
                call_button='Run'
        )
        def interface( 
            Channel: int = args.get('channel', 0),  
            nucChannel: int = args.get('nucChannel', 0),
            ):
            try:
                receipt['steps'][step_name]['spot_name'] = spot_name
                receipt['steps'][step_name]['channel'] = Channel
                receipt['steps'][step_name]['nucChannel'] = nucChannel
                # Get current p and t from the viewer's dims
                current_p = int(viewer.dims.current_step[0])
                current_t = int(viewer.dims.current_step[1])
                run(receipt, data, [current_p], [current_t])
            except Exception as e:
                print(f"[Error] Exception during spot detection: {e}")
            cellCounts, spots, clusters, ts, foci, spots_in, spots_out, spots_no_ts = compress_data(False)

            for layer_name in ["spots", 'spots', 'clusters', 'ts', 'foci', 'spots_in', 'spots_out', 'spots_no_ts']:
                if layer_name in viewer.layers:
                    viewer.layers.remove(layer_name)

            def display_df(df, name):
                # Show spots
                if df is not None:
                    coords = df[["fov", "timepoint", "channel", "z (px)", "y (px)", "x (px)"]].values
                    viewer.add_points(
                        coords,
                        name=name,
                        size=5,
                        face_color="red",
                        scale=[1, 1, 1, voxel_size_z/voxel_size_yx, 1, 1]
                    )
                    viewer.layers[name].refresh()
                else:
                    print(f'no {name} found')

            display_df(spots, 'spots')
            display_df(clusters, 'clusters')
            display_df(ts, 'ts')
            display_df(foci, 'foci')
            display_df(spots_in, 'spots_in')
            display_df(spots_out, 'spots_out')
            display_df(spots_no_ts, 'spots_no_ts')

            if cellCounts is not None:
                mask = data.get('cyto_masks', None)
                if mask is None:
                    mask = data.get('nuc_masks', None)

                if mask is not None:
                    # Get the 2D mask (max projection)
                    mask2d = mask[current_p, current_t]
                    if mask2d.ndim == 3:
                        mask2d = np.max(mask2d, axis=0)

                    # Use regionprops to get centroids
                    from skimage.measure import regionprops
                    props = regionprops(mask2d)

                    # Create text and coordinates
                    texts = []
                    coords = []
                    for prop in props:
                        label_id = prop.label
                        centroid = prop.centroid  # (row, col)

                        # Find RNA count for this cell from DataFrame
                        row = cellCounts[(cellCounts["cell_id"] == label_id) & (spots['fov'] == current_p) & (spots['timepoint'] == current_t)]
                        if not row.empty:
                            count = int(row["nb_rna"].values[0])
                            texts.append(str(count))
                            coords.append((centroid[1], centroid[0]))  # (x, y)

                    # Display text
                    if len(coords) > 0:
                        viewer.add_text(
                            text={'string': texts, 'color': 'white', 'size': 10},
                            data=coords,
                            name='RNA counts',
                            anchor='center'
                        )


            



        viewer.window.add_dock_widget(interface, area='right')

        def on_destroyed(obj=None):
            print('cleaning up')
            compress_data()
            release_memory()

        viewer.window._qt_window.destroyed.connect(on_destroyed)

    return receipt









