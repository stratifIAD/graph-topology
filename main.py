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

from params import DATA_PATH, MORPHOLOGICAL_CSV_PATH, TOPOLOGICAL_CSV_PATH, VISUALIZATION_WSI_PATH, VISUALIZATION_GRAPH_PATH, WSI_ID, MORPHOLOGICAL_PROPERTIES, CENTROID_ID, VISUALIZATION_CENTROID, VISUALIZATION_GRAPH, GRAPH, UNET_THRESHOLD, RESULTS_PATH, EXPERIMENT_NAME
from utils import plot_centroids, plot_graph, compute_graph, extract_graph_topology


if __name__ == "__main__":
    
    os.makedirs(RESULTS_PATH, exist_ok=True)
    os.makedirs(MORPHOLOGICAL_CSV_PATH, exist_ok=True)
    os.makedirs(TOPOLOGICAL_CSV_PATH, exist_ok=True)
    
    if WSI_ID == "all":
        predictions_paths = sorted(glob.glob(DATA_PATH + "/*"))
    elif type(WSI_ID) == list:
        try: 
            predictions_paths = sorted(glob.glob(DATA_PATH + "/*"))
            predictions_paths = [predictions_paths[i] for i in WSI_ID]
        except:
            print('Not a valid WSI_ID list in config.yml')
    else : 
        print('Not a valid WSI_ID value in config.yml')
    
    df_topology = pd.DataFrame() # instantiate empty dataframe to store topological features all graphs
    
    for path in tqdm(predictions_paths):
        
        wsi_name = path.split('/')[-1] # get name of wsi from folder's name
        df_morphology = pd.DataFrame() # instantiate empty dataframe to store morphological features of one specific wsi
        images_paths = sorted(glob.glob(path + '/*.npy')) # get the path to each patch of the wsi
                            
        for i, patch_path in tqdm(enumerate(images_paths), leave=False):
            xcoord = patch_path.split('/')[-1][:-4].split('_')[-2] # requires to have a standard naming convention of patch_paths (data/wsi_name/wsi_name_xcoord_ycoord.png)
            ycoord = patch_path.split('/')[-1][:-4].split('_')[-1]
            
            image = np.load(patch_path) # load patch as numpy array
            image = np.where(image > UNET_THRESHOLD, 1, 0) # threshold the patch
            instance_map = label(ndimage.binary_opening(image, iterations=3)) # quick data processing to remove small objects and close small holes
            regions = regionprops(instance_map)
            tmp_df_1 = pd.DataFrame() # temporary dataframe to store properties of one region
            
            for j, props in enumerate(regions):
                tmp_df_2 = pd.DataFrame(regionprops_table(instance_map, properties=MORPHOLOGICAL_PROPERTIES))
                
                if CENTROID_ID : # give an id to each centroid if specified in config.yml
                    centroid_id = patch_path.split('/')[-1][:-4] + f'_{j}'
                    tmp_df_2['centroid_id'] = centroid_id
                    
                tmp_df_2['centroid-0'] += int(xcoord)
                tmp_df_2['centroid-1'] += int(ycoord)
                
                tmp_df_1 = pd.concat([tmp_df_1, tmp_df_2], ignore_index=True)
                
            df_morphology = pd.concat([df_morphology, tmp_df_1], ignore_index=True)
            
        df_morphology.drop_duplicates(inplace=True) # in case there are duplicated rows, remove them
        df_morphology.to_csv(os.path.join(MORPHOLOGICAL_CSV_PATH, f'morphology_{wsi_name}.csv'), index=False) # save dataframe with all the morphological features of one wsi

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
        
        tmp_df_3 = extract_graph_topology(G=G)
        df_topology = pd.concat([df_topology, tmp_df_3], ignore_index=True)
        df_topology.to_csv(os.path.join(TOPOLOGICAL_CSV_PATH, f'topology_{EXPERIMENT_NAME}.csv'), index=False) # save dataframe with all the morphological features of one wsi
        
        if VISUALIZATION_GRAPH['active']:
            os.makedirs(VISUALIZATION_GRAPH_PATH, exist_ok=True)
            plot_graph(G=G, 
                       pos=pos, 
                       wsi_name=wsi_name,
                       save_path=VISUALIZATION_GRAPH_PATH,
                       figsize=VISUALIZATION_GRAPH['figsize'],
                       title=VISUALIZATION_GRAPH['title'],
                       dpi=VISUALIZATION_GRAPH['dpi'],
                       axis=VISUALIZATION_GRAPH['axis'],
                       options=VISUALIZATION_GRAPH['options'],
                       dropout=GRAPH['dropout'])
            
                       
            