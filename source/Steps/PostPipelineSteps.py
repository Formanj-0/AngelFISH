import pathlib
import numpy as np
import shutil
from fpdf import FPDF
import os
import pickle
import trackpy as tp
import matplotlib.pyplot as plt
import pandas as pd

from src import Settings, Experiment, ScopeClass, DataContainer, FinalizingStepClass, NASConnection

#%% Jack
class TrackPyAnlaysis(FinalizingStepClass):
    def __init__(self) -> None:
        super().__init__()

    def main(self, list_images, FISHChannel, analysis_location, voxel_size_yx, voxel_size_z, timestep_s: float, df_spotresults: pd.DataFrame = None, 
             df_clusterresults: pd.DataFrame = None, trackpy_features: pd.DataFrame = None, display_plots: bool = False, trackpy_link_distance_um: float = 1.25,
             link_search_range: list[float] = [0.5], trackpy_memory: int = 2, trackpy_max_lagtime: int = 3,
             min_trajectory_length: int = 5, **kwargs):

        # tp.linking.Linker.MAX_SUB_NET_SIZE = 100

        # use bigfish or trackpy features
        if trackpy_features is None and df_spotresults is not None:
            trackpy_features = df_spotresults
            if 'x_subpx' in trackpy_features.columns:
                trackpy_features['xnm'] = trackpy_features['x_subpx'] * voxel_size_yx
                trackpy_features['ynm'] = trackpy_features['y_subpx'] * voxel_size_yx
                trackpy_features['znm'] = trackpy_features('z_subpx', default=0) * voxel_size_z
                trackpy_features['frame'] = trackpy_features['timepoint']
                cluster_features = df_clusterresults
                cluster_features['xum'] = cluster_features['x_subpx'] * voxel_size_yx
                cluster_features['ynm'] = cluster_features['y_subpx'] * voxel_size_yx
                cluster_features['znm'] = cluster_features.get('z_subpx', default=0) * voxel_size_z
                cluster_features['frame'] = cluster_features['timepoint']
                raise ValueError('sub pix features not supported')
            else:
                trackpy_features['xnm'] = trackpy_features['x_px'] * voxel_size_yx
                trackpy_features['ynm'] = trackpy_features['y_px'] * voxel_size_yx
                trackpy_features['znm'] = trackpy_features.get('z_px', default=0) * voxel_size_z
                trackpy_features['frame'] = trackpy_features['timepoint']
                cluster_features = df_clusterresults
                cluster_features['xnm'] = cluster_features['x_px'] * voxel_size_yx
                cluster_features['ynm'] = cluster_features['y_px'] * voxel_size_yx
                cluster_features['znm'] = cluster_features.get('z_px', default=0) * voxel_size_z
                cluster_features['frame'] = cluster_features['timepoint']

        elif trackpy_features is None and df_spotresults is None:
            raise ValueError('No spot detection results provided')

        else: 
            trackpy_features = trackpy_features

        print(trackpy_features.keys())

        # prealocate variables
        links = None
        diffusion_constants = {}
        msds = None

        # iterate over fovs
        for i, fov in enumerate(trackpy_features['fov'].unique()):
            features = trackpy_features[trackpy_features['fov'] == fov]

            if 'zum' in features.columns:
                pos_columns = ['xnm', 'ynm', 'znm']
                px_pos_columns = ['x_px', 'y_px', 'z_px']
            else:
                pos_columns = ['xnm', 'ynm']
                px_pos_columns = ['x_px', 'y_px']

            # link features
            linked = tp.link_df(features, 500, adaptive_stop=40, adaptive_step=0.95, 
                                memory=trackpy_memory, pos_columns=pos_columns)
            linked = linked.groupby('particle').filter(lambda x: len(x) >= min_trajectory_length)
            # calculate msd and diffusion constants
            fps = 1/timestep_s
            msd3D = tp.emsd(linked, mpp=(voxel_size_yx/1000, voxel_size_yx/1000, voxel_size_z/1000) if len(pos_columns) == 3 else (voxel_size_yx/1000, voxel_size_yx/1000),
                            fps=fps, max_lagtime=trackpy_max_lagtime,
                            pos_columns=px_pos_columns)
            fit = tp.utils.fit_powerlaw(msd3D)
            n, slope = float(fit['n']), float(fit['A'])
            if len(pos_columns) == 3:
                print(f'The diffusion constant is {slope/6:.2f} μm²/s')
                print(f'The anomalous exponent is {n:.2f}')
            else:
                print(f'The diffusion constant is {slope/4:.2f} μm²/s')
                print(f'The anomalous exponent is {n:.2f}')

            # int_msd3D = np.array(msd3D)
            # slope = np.linalg.lstsq(int_msd3D[:, np.newaxis], int_msd3D)[0][0]
  

            # merge dataframes from multiple fovs
            linked['fov'] = fov*len(linked)
            if links is None:
                links = linked
            else:
                links = pd.concat([links, linked])

            # merge dataframes from multiple fovs
            msd3D['fov'] = fov*len(msd3D)
            if msds is None:
                msds = msd3D    
            else:
                msds = pd.concat([msds, msd3D])

            # calculate diffusion constant for each fov in 2D or 3D
            if 'zum' in features.columns:
                diffusion_constants[f'fov num {fov}'] = slope / 6
            else:
                diffusion_constants[f'fov num {fov}'] = slope / 4

        if display_plots:
            from celluloid import Camera
            for particleIndex in linked['particle'].unique():
                if particleIndex % 100 == 0:
                    singleTrack = linked.loc[linked.particle == particleIndex]
                    xrange = (int(singleTrack['x_px'].min()-50), int(singleTrack['x_px'].max()+50))
                    yrange = (int(singleTrack['y_px'].min()-50), int(singleTrack['y_px'].max()+50))
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    camera = Camera(fig)
                    timepoints = singleTrack['img_id']

                    for i in timepoints:
                        row = singleTrack.loc[singleTrack['frame'] == i]
                        ax.imshow(np.max(list_images[i][:, :, :, FISHChannel[0]], axis=0), cmap='gray')
                        ax.plot(singleTrack['x_px'], singleTrack['y_px'], color='b')
                        try:
                            ax.scatter(row['x_px'], row['y_px'], c='r', marker='2')
                        except KeyError:
                            pass
                        plt.xlim(xrange[0], xrange[1])
                        plt.ylim(yrange[0], yrange[1])
                        camera.snap()
                        
                    animation = camera.animate(interval=500, blit=True)
                    animation.save(f'animation_track_particle{particleIndex}.mp4')

        if analysis_location is not None:
            links.to_csv(os.path.join(analysis_location,'trackpy_links.csv'))
            msd3D.to_csv(os.path.join(analysis_location,'trackpy_msd.csv'))
            # save diffusion constants to a text file
            # with open(os.path.join(analysis_location,'diffusion_constants.txt'), 'w') as f:
            #     for key in diffusion_constants.keys():
            #         f.write(f'{key}: {diffusion_constants[key]}\n')

        return links, msds, diffusion_constants



