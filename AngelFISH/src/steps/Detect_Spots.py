import numpy as np
from typing import Union
from bigfish import stack, detection, multistack, plot
import pandas as pd
import os

from src import abstract_task, load_data


class detect_spots(abstract_task):
    def extract_args(self):
        pass

    def preallocate_memmory(self):
        self.temp_dir = os.path.join(self.receipt['dirs']['analysis_dir'], self.self.receipt['steps'][self.step_name]['spot_name'])
        if not hasattr(self, 'thresholds'):
            self.thresholds = {}

        if not hasattr(self, 'cell_csvs'):
            self.cell_csvs = []
        if not hasattr(self, 'spot_csvs'):
            self.spot_csvs = []
        if not hasattr(self, 'cluster_csvs'):
            self.cluster_csvs = []

    @staticmethod
    def image_processing_function(
        image, FISHChannel,  nucChannel, voxel_size_yx: int, voxel_size_z: int, 
        spot_yx: int, spot_z: int, timepoint, fov, independent_params: dict, 
        spot_name: str, nuc_mask:np.array=None, cell_mask:np.array=None, 
        bigfish_threshold: Union[int, str] = None, 
        snr_threshold: float = None, snr_ratio: float = None,
        bigfish_alpha: float = 0.7, bigfish_beta:float = 1, bigfish_gamma:float = 5, 
        CLUSTER_RADIUS:int = 500, MIN_NUM_SPOT_FOR_CLUSTER:int = 4, use_log_hook:bool = False, 
        verbose:bool = False, display_plots: bool = False, bigfish_use_pca: bool = False,
        sub_pixel_fitting: bool = False, bigfish_minDistance:Union[float, list] = None
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
        
        def get_detected_spots(FISHChannel: int, rna:np.array, voxel_size_yx:float, voxel_size_z:float, spot_yx:float, spot_z:float, alpha:int, beta:int,
                               gamma:int, CLUSTER_RADIUS:float, MIN_NUM_SPOT_FOR_CLUSTER:int, use_log_hook:bool, 
                               verbose: bool = False, display_plots: bool = False, sub_pixel_fitting: bool = False, minimum_distance:Union[list, float] = None,
                               use_pca: bool = False, snr_threshold: float = None, snr_ratio: float = None, bigfish_threshold: Union[int, str] = None,  **kwargs):

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
            try: # TODO fix this try 
                spots_post_decomposition, dense_regions, reference_spot = detection.decompose_dense(
                                                    image=rna.astype(np.uint16), 
                                                    spots=canidate_spots, 
                                                    voxel_size=voxel_size_nm, 
                                                    spot_radius=spot_size_nm if not use_log_hook else spot_radius_px,
                                                    alpha=alpha,
                                                    beta=beta,
                                                    gamma=gamma)
            except RuntimeError:
                spots_post_decomposition = canidate_spots
                dense_regions = None
                reference_spot = None

            # TODO: define ts by some other metric for ts
            #         
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
                    minimum_distance=spot_radius_px if use_log_hook else None)
                plot.plot_reference_spot(reference_spot, rescale=True)            
                plot.plot_detection(rna if len(rna.shape) == 2 else np.max(rna, axis=0),
                                        canidate_spots, contrast=True)
                            
                plot.plot_detection(rna if len(rna.shape) == 2 else np.max(rna, axis=0),
                                        spots_post_decomposition, contrast=True)
                            
                plot.plot_detection(rna if len(rna.shape) == 2 else np.max(rna, axis=0), 
                                        spots=[spots_post_decomposition, clusters[:, :2] if len(rna.shape) == 2 else clusters[:, :3]], 
                                        shape=["circle", "circle"], 
                                        radius=[3, 6], 
                                        color=["red", "blue"],
                                        linewidth=[1, 2], 
                                        fill=[False, True], 
                                        contrast=True)
                
            return spots_post_clustering, dense_regions, reference_spot, clusters, spots_subpx, individual_thershold

        def standardize_df(df_cellresults, spots_px, spots_subpx, sub_pixel_fitting, clusters, c, timepoint, fov, independent_params, dim_3D):
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
                            title="Cell {0}".format(i))

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
            # rna = rna

            # detect spots
            print('Detecting Spots')
            spots_px, dense_regions, reference_spot, clusters, spots_subpx, threshold = get_detected_spots( FISHChannel=c,
                rna=rna, voxel_size_yx=voxel_size_yx, voxel_size_z=voxel_size_z, spot_yx=spot_yx, spot_z=spot_z, alpha=bigfish_alpha,
                beta=bigfish_beta, gamma=bigfish_gamma, CLUSTER_RADIUS=CLUSTER_RADIUS, MIN_NUM_SPOT_FOR_CLUSTER=MIN_NUM_SPOT_FOR_CLUSTER, 
                bigfish_threshold=bigfish_threshold, use_log_hook=use_log_hook, verbose=verbose, display_plots=display_plots, sub_pixel_fitting=sub_pixel_fitting,
                minimum_distance=bigfish_minDistance, use_pca=bigfish_use_pca, snr_threshold=snr_threshold, snr_ratio=snr_ratio)
            
            print('Extracting Cell Results from masks')
            cell_results, spots_px, clusters = extract_cell_level_results(image, spots_px, clusters, nucChannel, FISHChannel[c], 
                                                            nuc_mask, cell_mask, timepoint, fov,
                                                            verbose, display_plots)

            print('Computing Spot Properties')
            spots_px = get_spot_properties(rna, spots_px, voxel_size_yx, voxel_size_z, spot_yx, spot_z, display_plots)

            print('Standardizing Data')
            spots, clusters = standardize_df(cell_results, spots_px, spots_subpx, sub_pixel_fitting, clusters, FISHChannel[c], timepoint, fov, independent_params)

            # output = SpotDetectionOutputClass(cell_results, spots, clusters, threshold)
            print('Complete Spot Detection')
        return {'cellresults': cell_results, 'spotresults': spots, 'clusterresults': clusters, 'individual_spotdetection_thresholds': threshold}

    def write_results(self, results, p, t):
        spot_name = self.receipt['steps'][self.step_name]['spot_name']
        cell_path = os.path.join(self.temp_dir, f'p{p}_t{t}_{spot_name}_cellresults.csv')
        spot_path = os.path.join(self.temp_dir, f'p{p}_t{t}_{spot_name}_spotresults.csv')
        cluster_path = os.path.join(self.temp_dir, f'p{p}_t{t}_{spot_name}_clusterresults.csv')

        results['cellresults'].to_csv(cell_path, index=False)
        results['spotresults'].to_csv(spot_path, index=False)
        results['clusterresults'].to_csv(cluster_path, index=False)

        self.cell_csvs.append(cell_path)
        self.spot_csvs.append(spot_path)
        self.cluster_csvs.append(cluster_path)

        self.thresholds[f'{p}, {t}'] = results['individual_spotdetection_thresholds']

    def compress_and_release_memory(self):
        spot_name = self.receipt['steps'][self.step_name]['spot_name']
        results_dir = self.receipt['dirs']['results_dir']

        final_cell_df = pd.concat([pd.read_csv(f) for f in self.cell_csvs], ignore_index=True)
        final_spot_df = pd.concat([pd.read_csv(f) for f in self.spot_csvs], ignore_index=True)
        final_cluster_df = pd.concat([pd.read_csv(f) for f in self.cluster_csvs], ignore_index=True)

        final_cell_df.to_csv(os.path.join(results_dir ,f'{spot_name}_cellresults.csv'), index=False)
        final_spot_df.to_csv(os.path.join(results_dir, f'{spot_name}_spotresults.csv'), index=False)
        final_cluster_df.to_csv(os.path.join(results_dir, f'{spot_name}_clusterresults.csv'), index=False)




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



























