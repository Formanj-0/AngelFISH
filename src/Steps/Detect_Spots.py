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


from AngelFISH.src import abstract_task, load_data
import re


class detect_spots(abstract_task):
    @classmethod
    def task_name(cls):
        return 'detect_spots'

    def extract_args(self, p, t):
        given_args = self.receipt['steps'][self.step_name]

        data_to_send = {}
        data_to_send['fig_dir'] = self.receipt['dirs']['fig_dir']
        data_to_send['image'] = self.data['images'][p, t].compute()
        data_to_send['metadata'] = self.data['metadata'](p, t).get('experimental_metadata', {})
        self.voxel_size_yx = self.data['metadata'](p, t)['PixelSizeUm'] * 1000
        data_to_send['voxel_size_yx'] = self.voxel_size_yx
        self.voxel_size_z = np.abs(self.data['metadata'](p, t, z=1)['ZPosition_um_Intended'] - self.data['metadata'](p, t, z=0)['ZPosition_um_Intended']) * 1000
        data_to_send['voxel_size_z'] = self.voxel_size_z
        nuc_masks = self.data.get('nuc_masks', None)
        if nuc_masks is not None:
            data_to_send['nuc_mask'] = nuc_masks[p,t]

        cyto_masks = self.data.get('cyto_masks', None)
        if cyto_masks is not None:
            data_to_send['cyto_mask'] = cyto_masks[p,t]

        args = {**data_to_send, **given_args}

        args['fov'] = p
        args['timepoint'] = t

        return args

    def preallocate_memmory(self):
        self.temp_dir = os.path.join(self.receipt['dirs']['analysis_dir'], self.receipt['steps'][self.step_name]['spot_name'])
        os.makedirs(self.temp_dir, exist_ok=True)
        
        if not hasattr(self, 'thresholds'):
            self.thresholds = {}

        # if not hasattr(self, 'cell_csvs'):
        #     self.cell_csvs = []
        # if not hasattr(self, 'spot_csvs'):
        #     self.spot_csvs = []
        # if not hasattr(self, 'cluster_csvs'):
        #     self.cluster_csvs = []

    @staticmethod
    def image_processing_function(
        image, 
        FISHChannel: int,  
        voxel_size_yx: float, 
        voxel_size_z: float, 
        spot_yx: float, 
        spot_z: float, 
        timepoint:int, 
        fov:int, 
        nucChannel: int = None,
        nuc_mask:np.array=None, 
        cell_mask:np.array=None, 
        threshold: Union[int, str] = None, 
        snr_threshold: float = None, 
        snr_ratio: float = None,
        alpha: float = 0.7, 
        beta:float = 1, 
        gamma:float = 5, 
        cluster_radius:int = 500, 
        min_num_spot_per_cluster:int = 4, 
        use_log_hook:bool = False, 
        verbose:bool = False, 
        display_plots: bool = False, 
        use_pca: bool = False,
        sub_pixel_fitting: bool = False, 
        minDistance:Union[float, list] = None,
        metadata:callable=None,
        **kwargs
        ):

        def _establish_threshold(c, bigfish_threshold, kwargs):
                if bigfish_threshold is not None:
                    # check if a threshold is provided
                    if isinstance(bigfish_threshold[c], (int, float)):
                        threshold = bigfish_threshold[c]
                    elif bigfish_threshold == 'mean':
                        threshold = kwargs['bigfish_mean_threshold'][c]
                    elif bigfish_threshold == 'min':
                        threshold = kwargs['bigfish_min_threshold'][c]
                    elif bigfish_threshold == 'max':
                        threshold = kwargs['bigfish_max_threshold'][c]
                    elif bigfish_threshold == 'median':
                        threshold = kwargs['bigfish_median_threshold'][c]
                    elif bigfish_threshold == 'mode':
                        threshold = kwargs['bigfish_mode_threshold'][c]
                    elif bigfish_threshold == '75th_percentile':
                        threshold = kwargs['bigfish_75_quartile'][c]
                    elif bigfish_threshold == '25th_percentile':
                        threshold = kwargs['bigfish_25_quartile'][c]
                    elif bigfish_threshold == '90th_percentile':
                        threshold = kwargs['bigfish_90_quartile'][c]
                    else: 
                        raise ValueError('IDK what to do wit this threshold type')
                else:
                    threshold = None

                return threshold
        
        def get_detected_spots(FISHChannel: int, 
                               rna:np.array, 
                               voxel_size_yx:float, 
                               voxel_size_z:float, 
                               spot_yx:float, 
                               spot_z:float, 
                               alpha:int, 
                               beta:int,
                               gamma:int, 
                               CLUSTER_RADIUS:float, 
                               MIN_NUM_SPOT_FOR_CLUSTER:int, 
                               use_log_hook:bool, 
                               verbose: bool = False, 
                               display_plots: bool = False, 
                               sub_pixel_fitting: bool = False, 
                               minimum_distance:Union[list, float] = None,
                               use_pca: bool = False, 
                               snr_threshold: float = None, 
                               snr_ratio: float = None, 
                               bigfish_threshold: Union[int, str] = None,  **kwargs):

            threshold = _establish_threshold(FISHChannel, bigfish_threshold, kwargs)

            dim_3D = len(rna.shape) == 3
            voxel_size_nm = (int(voxel_size_z), int(voxel_size_yx), int(voxel_size_yx)) if len(rna.shape) == 3 else (int(voxel_size_yx), int(voxel_size_yx))
            spot_size_nm = (int(spot_z), int(spot_yx), int(spot_yx)) if len(rna.shape) == 3 else (int(spot_yx), int(spot_yx))

            if use_log_hook:
                if minimum_distance is None:
                    spot_radius_px = detection.get_object_radius_pixel(
                            voxel_size_nm=voxel_size_nm, 
                            object_radius_nm=spot_size_nm, 
                            ndim=len(rna.shape))
                else:
                    spot_radius_px = minimum_distance
            else:
                spot_radius_px = None

            canidate_spots, individual_thershold = detection.detect_spots(
                                            images=rna, 
                                            return_threshold=True, 
                                            threshold=threshold,
                                            voxel_size=voxel_size_nm if not use_log_hook else None,
                                            spot_radius=spot_size_nm if not use_log_hook else None,
                                            log_kernel_size=spot_radius_px if use_log_hook else None,
                                            minimum_distance=spot_radius_px if use_log_hook else None,)
            
            if use_pca:
                # lets try log filter
                log = stack.log_filter(rna.copy(), 3)

                # TODO: PCA for overdetected spots
                print(canidate_spots.shape)
                valid_spots = np.ones(canidate_spots.shape[0])
                canidate_spots = np.array(canidate_spots)
                pca_data = np.zeros((canidate_spots.shape[0], 5*5))

                for i in range(canidate_spots.shape[0]-1):
                    xyz = canidate_spots[i, :] # idk why this is being mean to me 
                    if len(rna.shape) == 3:
                        x, y, z = xyz
                        try:
                            spot_kernel = log[z-2:z+3, y-2:y+3, x-2:x+3]
                            pca_data[i, :] = spot_kernel.flatten()
                            plt.imshow(spot_kernel)
                        except:
                            valid_spots[i] = 0
                    else:
                        x, y = xyz
                        try:
                            spot_kernel = log[y-2:y+3, x-2:x+3]
                            pca_data[i, :] = spot_kernel.flatten()
                            plt.imshow(spot_kernel)
                        except:
                            valid_spots[i] = 0

                plt.show()

                # z score normalization
                pca_data = (pca_data - np.mean(pca_data, axis=0)) / np.std(pca_data, axis=0)
                pca = PCA(n_components=9)
                pca.fit(pca_data)
                X = pca.transform(pca_data)

                # color the spots best on the clusters
                kmeans_pca = KMeans(n_clusters=2)
                kmeans_pca.fit(X)
                plt.scatter(X[:, 0], X[:, 1], c=kmeans_pca.labels_)
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.show()

                # remove larger varying clusters
                valid_spots[kmeans_pca.labels_ != 1] = 0
                valid_spots = valid_spots.astype(bool)
                canidate_spots = canidate_spots[valid_spots, :]

                print(canidate_spots.shape)

            # Compute SNR of the detected spots (2D array of coordinates)
            snr_spots, max_signal = compute_snr_spots(
                image=rna.astype(np.float64), 
                spots=canidate_spots.astype(np.float64), 
                voxel_size=voxel_size_nm, 
                spot_radius=spot_size_nm if not use_log_hook else spot_radius_px)
            if display_plots:
                plt.hist(snr_spots, bins=100)
                plt.xlabel('SNR')
                plt.ylabel('Frequency')
                plt.title('SNR distribution')
                plt.show()
            print(f'median SNR: {np.median(snr_spots)}')
            print(f'mean SNR: {np.mean(snr_spots)}')
            if display_plots:
                plt.scatter(max_signal, snr_spots, color='blue', alpha=0.5, s=3)
                plt.xlabel('Max signal')
                plt.xscale('log')
                plt.yscale('log')
                plt.ylabel('SNR')
                plt.title('SNR vs max signal')
                plt.show()        

            if snr_threshold is None and snr_ratio is not None:
                snr_threshold = np.median(snr_spots)*snr_ratio
            print(f'SNR threshold: {snr_threshold}')

            if snr_threshold is not None:
                # Print the number of spots before filtering
                print(f'Number of spots before SNR filtering: {canidate_spots.shape[0]}')
                good_spots = [True if snr > snr_threshold else False for snr in snr_spots]
                canidate_spots = canidate_spots[good_spots, :]
                print(f'Number of spots after SNR filtering: {canidate_spots.shape[0]}')

            # decompose dense regions
            spots_post_decomposition, dense_regions, reference_spot = detection.decompose_dense(
                                                image=rna.astype(np.uint16), 
                                                spots=canidate_spots, 
                                                voxel_size=voxel_size_nm, 
                                                spot_radius=spot_size_nm if not use_log_hook else spot_radius_px,
                                                alpha=alpha,
                                                beta=beta,
                                                gamma=gamma)

    
            spots_post_clustering, clusters = detection.detect_clusters(
                                                            spots=spots_post_decomposition, 
                                                            voxel_size=voxel_size_nm, 
                                                            radius=CLUSTER_RADIUS, 
                                                            nb_min_spots=MIN_NUM_SPOT_FOR_CLUSTER)
            
            if sub_pixel_fitting:
                spots_subpx = detection.fit_subpixel(
                                            image=rna, 
                                            spots=spots_post_clustering, 
                                            voxel_size=voxel_size_nm, 
                                            spot_radius=voxel_size_nm)
            else:
                spots_subpx = None
                
            if verbose:
                print("detected canidate spots")
                print("\r shape: {0}".format(canidate_spots.shape))
                print("\r threshold: {0}".format(individual_thershold))
                print("detected spots after decomposition")
                print("\r shape: {0}".format(spots_post_decomposition.shape))
                print("detected spots after clustering")
                print("\r shape: {0}".format(spots_post_clustering.shape))
                print("detected clusters")
                print("\r shape: {0}".format(clusters.shape))

            if display_plots:
                plot.plot_elbow(
                    images=rna, 
                    voxel_size=voxel_size_nm if not use_log_hook else None, 
                    spot_radius=spot_size_nm if not use_log_hook else None,
                    log_kernel_size=spot_radius_px if use_log_hook else None,
                    minimum_distance=spot_radius_px if use_log_hook else None, 
                    path_output=os.path.join(kwargs['fig_dir'], f'fov{fov}_tp{timepoint}_ch{FISHChannel}-elbow.png'))
                plot.plot_reference_spot(reference_spot, 
                                         rescale=True, 
                                         path_output=os.path.join(kwargs['fig_dir'], f'fov{fov}_tp{timepoint}_ch{FISHChannel}-ref_spot.png'))            
                plot.plot_detection(rna if len(rna.shape) == 2 else np.max(rna, axis=0),
                                        canidate_spots, 
                                        contrast=True)
                            
                plot.plot_detection(rna if len(rna.shape) == 2 else np.max(rna, axis=0),
                                        spots_post_decomposition, contrast=True)
                            
                plot.plot_detection(rna if len(rna.shape) == 2 else np.max(rna, axis=0), 
                                        spots=[spots_post_decomposition, clusters[:, :2] if len(rna.shape) == 2 else clusters[:, :3]], 
                                        shape=["circle", "circle"], 
                                        radius=[3, 6], 
                                        color=["red", "blue"],
                                        linewidth=[1, 2], 
                                        fill=[False, True], 
                                        contrast=True,
                                        path_output=os.path.join(kwargs['fig_dir'], f'fov{fov}_tp{timepoint}_ch{FISHChannel}-detections.png'))
                
            return spots_post_clustering, dense_regions, reference_spot, clusters, spots_subpx, individual_thershold

        def standardize_df(df_cellresults, spots_px, spots_subpx, sub_pixel_fitting, clusters, c, timepoint, fov, dim_3D):
            # get the columns
            cols_spots = []
            cols_cluster = []
            if dim_3D:
                cols_spots.append(['z_px', 'y_px', 'x_px', 'cluster_index'])
                cols_cluster.append(['z_px', 'y_px', 'x_px', 'nb_spots', 'cluster_index'])
            else:
                cols_spots.append(['y_px', 'x_px', 'cluster_index'])
                cols_cluster.append(['y_px', 'x_px', 'nb_spots', 'cluster_index'])
            
            if df_cellresults is not None:
                cols_spots.append(['is_nuc', 'cell_label', 'nuc_label'])
                cols_cluster.append(['is_nuc', 'cell_label', 'nuc_label'])
            
            cols_spots.append(['snr', 'signal'])

            if dim_3D and sub_pixel_fitting:
                spots = np.concatenate([spots_px, spots_subpx], axis=1)
                cols_spots.append(['z_nm', 'y_nm', 'x_nm'])

            elif not dim_3D and sub_pixel_fitting:
                spots = np.concatenate([spots_px, spots_subpx], axis=1)
                cols_spots.append(['y_nm', 'x_nm'])
            
            else:
                spots = spots_px

            cols_spots = [item for sublist in cols_spots for item in sublist]
            cols_cluster = [item for sublist in cols_cluster for item in sublist]

            # set the columns
            df_spotresults = pd.DataFrame(spots, columns=cols_spots)
            df_clusterresults = pd.DataFrame(clusters, columns=cols_cluster)

            # add nesscary columns
            df_spotresults['timepoint'] = [timepoint]*len(df_spotresults)
            df_spotresults['fov'] = [fov]*len(df_spotresults)
            df_spotresults['FISH_Channel'] = [c]*len(df_spotresults)

            df_clusterresults['timepoint'] = [timepoint]*len(df_clusterresults)
            df_clusterresults['fov'] = [fov]*len(df_clusterresults)
            df_clusterresults['FISH_Channel'] = [c]*len(df_clusterresults)

            return df_spotresults, df_clusterresults

        def extract_cell_level_results(image, spots, clusters, nucChannel, FISHChannel, 
                                   nuc_mask, cell_mask, timepoint, fov,
                                    verbose, display_plots) -> pd.DataFrame:
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
            else:
                df = None

            return df, spots, clusters

        def get_spot_properties(rna, spots, voxel_size_yx, voxel_size_z, spot_yx, spot_z, display_plots, **kwargs) -> pd.DataFrame:
            voxel_size_nm = (int(voxel_size_z), int(voxel_size_yx), int(voxel_size_yx)) if len(rna.shape) == 3 else (int(voxel_size_yx), int(voxel_size_yx))
            spot_size_nm = (int(spot_z), int(spot_yx), int(spot_yx)) if len(rna.shape) == 3 else (int(spot_yx), int(spot_yx))
            snr, signal = compute_snr_spots(rna, spots[:, :len(voxel_size_nm)], voxel_size_nm, spot_size_nm, display_plots)
            snr = np.array(snr).reshape(-1, 1)
            signal = np.array(signal).reshape(-1, 1)
            spots = np.hstack([spots, snr, signal])
            return spots

        if isinstance(FISHChannel, int):
            FISHChannel = [FISHChannel]

        # cycle through FISH channels
        for c in range(len(FISHChannel)):
            rna = image[FISHChannel[c], :, :, :]
            rna = rna.squeeze()
            dim_3D = len(rna.shape) == 3

            # detect spots
            print('Detecting Spots')
            spots_px, dense_regions, reference_spot, clusters, spots_subpx, threshold = get_detected_spots( FISHChannel=c,
                rna=rna, voxel_size_yx=voxel_size_yx, voxel_size_z=voxel_size_z, spot_yx=spot_yx, spot_z=spot_z, alpha=alpha,
                beta=beta, gamma=gamma, CLUSTER_RADIUS=cluster_radius, MIN_NUM_SPOT_FOR_CLUSTER=min_num_spot_per_cluster, 
                bigfish_threshold=threshold, use_log_hook=use_log_hook, verbose=verbose, display_plots=display_plots, sub_pixel_fitting=sub_pixel_fitting,
                minimum_distance=minDistance, use_pca=use_pca, snr_threshold=snr_threshold, snr_ratio=snr_ratio, **kwargs)
            
            print('Extracting Cell Results from masks')
            cell_results, spots_px, clusters = extract_cell_level_results(image, spots_px, clusters, nucChannel, FISHChannel[c], 
                                                            nuc_mask, cell_mask, timepoint, fov,
                                                            verbose, display_plots)

            print('Computing Spot Properties')
            spots_px = get_spot_properties(rna, spots_px, voxel_size_yx, voxel_size_z, spot_yx, spot_z, display_plots)

            print('Standardizing Data')
            spots, clusters = standardize_df(cell_results, spots_px, spots_subpx, sub_pixel_fitting, clusters, FISHChannel[c], timepoint, fov, dim_3D)

            print('Adding Epermental Metadata')
            expermental_metadata = metadata(p=fov, t=timepoint, z=0 ,c=c)['experimental_metadata'] # is a dictionary 
            if expermental_metadata is not None:
                for key, value in expermental_metadata.items():
                    spots[key] = [value] * len(spots)
                    clusters[key] = [value] * len(clusters)
                    if cell_results is not None:
                        cell_results[key] = [value] * len(cell_results)


            # output = SpotDetectionOutputClass(cell_results, spots, clusters, threshold)
            print('Complete Spot Detection')
        return {'cellresults': cell_results, 'spotresults': spots, 'clusterresults': clusters, 'individual_spotdetection_thresholds': threshold}

    def write_results(self, results, p, t):
        spot_name = self.receipt['steps'][self.step_name]['spot_name']

        if results['cellresults'] is not None:
            cell_path = os.path.join(self.temp_dir, f'p{p}_t{t}_{spot_name}_cellresults.csv')
            results['cellresults'].to_csv(cell_path, index=False)
            # self.cell_csvs.append(cell_path)

        if results['spotresults'] is not None:
            spot_path = os.path.join(self.temp_dir, f'p{p}_t{t}_{spot_name}_spotresults.csv')
            results['spotresults'].to_csv(spot_path, index=False)
            # self.spot_csvs.append(spot_path)

        if results['clusterresults'] is not None:
            cluster_path = os.path.join(self.temp_dir, f'p{p}_t{t}_{spot_name}_clusterresults.csv')
            results['clusterresults'].to_csv(cluster_path, index=False)
            # self.cluster_csvs.append(cluster_path)



        self.thresholds[f'{p}, {t}'] = results['individual_spotdetection_thresholds']

    def compress_and_release_memory(self):
        spot_name = self.receipt['steps'][self.step_name]['spot_name']
        results_dir = self.receipt['dirs']['results_dir']

        # List all files in the temp_dir
        all_files = os.listdir(self.temp_dir)

        # Use regex to find matching files
        cell_pattern = re.compile(r'.*_cellresults\.csv$')
        spot_pattern = re.compile(r'.*_spotresults\.csv$')
        cluster_pattern = re.compile(r'.*_clusterresults\.csv$')

        cell_files = [os.path.join(self.temp_dir, f) for f in all_files if cell_pattern.match(f)]
        spot_files = [os.path.join(self.temp_dir, f) for f in all_files if spot_pattern.match(f)]
        cluster_files = [os.path.join(self.temp_dir, f) for f in all_files if cluster_pattern.match(f)]

        final_cell_df = pd.concat([pd.read_csv(f) for f in cell_files], ignore_index=True) if cell_files else None
        final_spot_df = pd.concat([pd.read_csv(f) for f in spot_files], ignore_index=True) if spot_files else None
        final_cluster_df = pd.concat([pd.read_csv(f) for f in cluster_files], ignore_index=True) if cluster_files else None

        if final_cell_df is not None:
            final_cell_df.to_csv(os.path.join(results_dir ,f'{spot_name}_cellresults.csv'), index=False)
        if final_spot_df is not None:
            final_spot_df.to_csv(os.path.join(results_dir, f'{spot_name}_spotresults.csv'), index=False)
        if final_cluster_df is not None:
            final_cluster_df.to_csv(os.path.join(results_dir, f'{spot_name}_clusterresults.csv'), index=False)

        # Delete all files in the temp_dir
        for f in all_files:
            file_path = os.path.join(self.temp_dir, f)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

        # Delete the temp_dir itself
        try:
            os.rmdir(self.temp_dir)
        except Exception as e:
            print(f"Error deleting directory {self.temp_dir}: {e}")

    @property
    def required_keys(self):
        return ['FISHChannel']

    def write_args_to_receipt(
            self,
            spot_name,
            FISHChannel,
            nucChannel,
            spot_yx,
            spot_z,
            p,
            t,
            threshold,
            snr_threshold,
            snr_ratio,
            alpha,
            beta,
            gamma,
            cluster_radius,
            min_num_spot_per_cluster,
            use_log_hook,
            verbose,
            display_plots,
            use_pca,
            sub_pixel_fitting,
            minDistance,
            ):
        self.receipt['steps'][self.step_name]['spot_name'] = spot_name
        self.receipt['steps'][self.step_name]['FISHChannel'] = FISHChannel
        self.receipt['steps'][self.step_name]['nucChannel'] = nucChannel
        self.receipt['steps'][self.step_name]['spot_yx'] = spot_yx
        self.receipt['steps'][self.step_name]['spot_z'] = spot_z
        self.receipt['steps'][self.step_name]['p'] = p
        self.receipt['steps'][self.step_name]['t'] = t
        self.receipt['steps'][self.step_name]['threshold'] = threshold
        self.receipt['steps'][self.step_name]['snr_threshold'] = snr_threshold
        self.receipt['steps'][self.step_name]['snr_ratio'] = snr_ratio
        self.receipt['steps'][self.step_name]['alpha'] = alpha
        self.receipt['steps'][self.step_name]['beta'] = beta
        self.receipt['steps'][self.step_name]['gamma'] = gamma
        self.receipt['steps'][self.step_name]['cluster_radius'] = cluster_radius
        self.receipt['steps'][self.step_name]['min_num_spot_per_cluster'] = min_num_spot_per_cluster
        self.receipt['steps'][self.step_name]['use_log_hook'] = use_log_hook
        self.receipt['steps'][self.step_name]['verbose'] = verbose
        self.receipt['steps'][self.step_name]['display_plots'] = display_plots
        self.receipt['steps'][self.step_name]['use_pca'] = use_pca
        self.receipt['steps'][self.step_name]['sub_pixel_fitting'] = sub_pixel_fitting
        self.receipt['steps'][self.step_name]['minDistance'] = minDistance

    def run_process(self, p, t):
        print(p, t)
        self.iterate_over_data(p_range=[p], t_range=[t], run_in_parallel=False)

    @magicgui(
            call_button='Run'
    )
    def interface(self, 
                spot_name: str, 
                FISHChannel: int,  
                nucChannel: int,
                spot_yx: float, 
                spot_z: float, 
                p:int=0, 
                t:int=0, 
                threshold: Union[int, str] = None, 
                snr_threshold: float = None, 
                snr_ratio: float = None,
                alpha: float = 0.7, 
                beta:float = 1, 
                gamma:float = 5, 
                cluster_radius:int = 500, 
                min_num_spot_per_cluster:int = 4, 
                use_log_hook:bool = False, 
                verbose:bool = False, 
                display_plots: bool = False, 
                use_pca: bool = False,
                sub_pixel_fitting: bool = False, 
                minDistance:float = None,):
        try:
            self.write_args_to_receipt(
                                spot_name,
                                FISHChannel,
                                nucChannel,
                                spot_yx,
                                spot_z,
                                p,
                                t,
                                threshold,
                                snr_threshold,
                                snr_ratio,
                                alpha,
                                beta,
                                gamma,
                                cluster_radius,
                                min_num_spot_per_cluster,
                                use_log_hook,
                                verbose,
                                display_plots,
                                use_pca,
                                sub_pixel_fitting,
                                minDistance
                                )
            self.preallocate_memmory()
            self.run_process(p, t)
        except Exception as e:
            print(f"[Error] Exception during spot detection: {e}")

        # List all files in the temp_dir
        all_files = os.listdir(self.temp_dir)

        # Use regex to find matching files
        cell_pattern = re.compile(r'.*_cellresults\.csv$')
        spot_pattern = re.compile(r'.*_spotresults\.csv$')
        cluster_pattern = re.compile(r'.*_clusterresults\.csv$')

        cell_files = [os.path.join(self.temp_dir, f) for f in all_files if cell_pattern.match(f)]
        spot_files = [os.path.join(self.temp_dir, f) for f in all_files if spot_pattern.match(f)]
        cluster_files = [os.path.join(self.temp_dir, f) for f in all_files if cluster_pattern.match(f)]

        final_cell_df = pd.concat([pd.read_csv(f) for f in cell_files], ignore_index=True) if cell_files else None
        final_spot_df = pd.concat([pd.read_csv(f) for f in spot_files], ignore_index=True) if spot_files else None
        final_cluster_df = pd.concat([pd.read_csv(f) for f in cluster_files], ignore_index=True) if cluster_files else None

        c = self.receipt['steps'][self.step_name]['FISHChannel'] 

        # if self.data.get('cyto_masks', None) is not None and not 'cyto_masks' in self.viewer.layers:
        #     print('cyto_masks')
        #     mask = np.array(self.data['cyto_masks'])
        #     temp = np.zeros_like(self.data['images'])
        #     cyto_key = next((k for k in self.receipt['steps'].keys() if 'cyto' in k), None) # this is dirty but should work
        #     print(cyto_key)
        #     temp[:, :, self.receipt['steps'][cyto_key]['channel'], :, :, :] = mask
        #     self.viewer.add_labels(
        #         temp,
        #         name='cyto_masks',
        #         axis_labels=('p', 't', 'c', 'z', 'y', 'x'),
        #         scale=[1, 1, 1, self.voxel_size_z/self.voxel_size_yx, 1, 1]
        #     )

        # print('did it fail while adding')
        # if self.data.get('nuc_masks', None) is not None and not 'nuc_masks' in self.viewer.layers:
        #     print('nuc_masks')
        #     mask = np.array(self.data['nuc_masks'])
        #     temp = np.zeros_like(self.data['images'])
        #     nuc_key = next((k for k in self.receipt['steps'].keys() if 'nuc' in k), None)
        #     print(nuc_key)
        #     temp[:, :, self.receipt['steps'][nuc_key]['channel'], :, :, :] = mask
        #     self.viewer.add_labels(
        #         temp,
        #         name='nuc_masks',
        #         axis_labels=('p', 't', 'c', 'z', 'y', 'x'),
        #         scale=[1, 1, 1, self.voxel_size_z/self.voxel_size_yx, 1, 1]
        #     )
        # print('here?')

        for layer_name in ["spots", "clusters"]:
            if layer_name in self.viewer.layers:
                self.viewer.layers.remove(layer_name)

        # Show spots
        if final_spot_df is not None:
            coords = final_spot_df[["fov", "timepoint", "FISH_Channel", "z_px", "y_px", "x_px"]].values
            self.viewer.add_points(
                coords,
                name="spots",
                size=5,
                face_color="red",
                scale=[1, 1, 1, self.voxel_size_z/self.voxel_size_yx, 1, 1]
            )
            self.viewer.layers['spots'].refresh()
        else:
            print('no spots found')

        # Show clusters
        if final_cluster_df is not None:
            coords = final_cluster_df[["fov", "timepoint", "FISH_Channel", "z_px", "y_px", "x_px"]].values
            self.viewer.add_points(
                coords,
                name="clusters",
                size=10,
                face_color="blue",
                scale=[1, 1, 1, self.voxel_size_z/self.voxel_size_yx, 1, 1]
            )
            self.viewer.layers['clusters'].refresh()
        else:
            print('no clusters found')



#%% Useful Functions
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



























