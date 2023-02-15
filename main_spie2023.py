import os
import numpy as np
import pandas as pd
import glob
from scipy import ndimage
from tqdm import tqdm
from skimage.measure import regionprops, regionprops_table
from skimage.segmentation import clear_border
from skimage.measure import label
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
from openslide import OpenSlide, lowlevel 


from params import DATA_PATH, MORPHOLOGICAL_CSV_PATH, TOPOLOGICAL_CSV_PATH, VISUALIZATION_WSI_PATH, VISUALIZATION_GRAPH_PATH, WSI_ID, MORPHOLOGICAL_PROPERTIES, CENTROID_ID, VISUALIZATION_CENTROID, VISUALIZATION_GRAPH, GRAPH, UNET_THRESHOLD, RESULTS_PATH, EXPERIMENT_NAME
from utils import plot_centroids, plot_graph_wsi, compute_graph, extract_graph_topology, compute_graph_slidelvl
from morphology import MorphologyStudy

if __name__ == "__main__":
    
    os.makedirs(RESULTS_PATH, exist_ok=True)
    os.makedirs(MORPHOLOGICAL_CSV_PATH, exist_ok=True)
    os.makedirs(TOPOLOGICAL_CSV_PATH, exist_ok=True)

    df_topology = pd.DataFrame() # instantiate empty dataframe to store topological features all graphs

    # check the annotations in .npy for a single WSI
    df_morphology = pd.DataFrame() # instantiate empty dataframe to store morphological features of one specific wsi
    images_paths = sorted(glob.glob(DATA_PATH + '/*.npy')) # get the path to each patch of the wsi

    for i, wsi_path in tqdm(enumerate(images_paths), leave=False):
        coordinates_annotations = np.load(wsi_path, allow_pickle=True)
        
        # find the morphological 
        wsi_num = wsi_path.split('/')[-1][:-4].split('_')[1]
        wsi_name = f'wsi_{wsi_num}_stratifiad'
        morphology = MorphologyStudy(coordinates_annotations, wsi_name)
        df_morphology, gray_matter = morphology.output_all_variables(os.path.join(MORPHOLOGICAL_CSV_PATH, f'morphology_{wsi_name}.csv'))

        # print(df_morphology)

        if GRAPH['dropout'] > 0:
            df_morphology = df_morphology.sample(frac=1-GRAPH['dropout'])     

        if VISUALIZATION_CENTROID['active']:
            os.makedirs(VISUALIZATION_WSI_PATH, exist_ok=True)
            plot_centroids(df=df_morphology,
                           wsi_name=wsi_name,
                           save_path=VISUALIZATION_WSI_PATH,
                           figsize=VISUALIZATION_CENTROID['figsize'],
                           size=VISUALIZATION_CENTROID['size'],
                           title=VISUALIZATION_CENTROID['title'],
                           dpi=VISUALIZATION_CENTROID['dpi'])

        G, pos = compute_graph(df=df_morphology, weight=GRAPH['weight'])
        
        if VISUALIZATION_GRAPH['active']:
            slide_dim_lvl = 6
            wsi_path='/Users/gabriel.jimenez/Documents/phd/AT8Dataset/AT8_wsi'
            slide = OpenSlide(os.path.join(wsi_path, f'{wsi_name}.ndpi'))
            slide_levels = slide.level_dimensions
            _, factor = np.array(slide_levels[0])/np.array(slide_levels[slide_dim_lvl])

            G_slidelvl, pos_slidelvl, divider = compute_graph_slidelvl(df=df_morphology, weight=GRAPH['weight'], factor=factor)
            
            wsi = slide.read_region((0, 0), slide_dim_lvl, slide_levels[slide_dim_lvl])
            extend_factor = slide_levels[slide_dim_lvl]

            os.makedirs(VISUALIZATION_GRAPH_PATH, exist_ok=True)
            plot_graph_wsi(G=G_slidelvl, 
                       pos=pos_slidelvl, 
                       divider=divider,
                       factor=factor,
                       wsi=wsi,
                       wsi_name=wsi_name,
                       gray_matter=gray_matter,
                       extend_factor=extend_factor,
                       save_path=VISUALIZATION_GRAPH_PATH,
                       options=VISUALIZATION_GRAPH['options'],
                       figsize=VISUALIZATION_GRAPH['figsize'],
                       title=VISUALIZATION_GRAPH['title'],
                       dpi=VISUALIZATION_GRAPH['dpi'],
                       axis=VISUALIZATION_GRAPH['axis'],
                       dropout=GRAPH['dropout'])                  


        tmp_df_3 = extract_graph_topology(G=G)
        tmp_df_3.insert(0, 'Slide ID', df_morphology['Slide ID'][0] * len(tmp_df_3))
        tmp_df_3.insert(1, 'Object Group', df_morphology['Object Group'][0] * len(tmp_df_3))

        df_topology = pd.concat([df_topology, tmp_df_3], ignore_index=True)
        df_topology.to_csv(os.path.join(TOPOLOGICAL_CSV_PATH, f'topology_{EXPERIMENT_NAME}.csv'), index=False) # save dataframe with all the morphological features of one wsi
        
        
