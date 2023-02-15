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
from matplotlib.lines import Line2D
import random


# k-means clustering
from numpy import unique
from numpy import where
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from params import EXPERIMENT_PATH, DATA_PATH, MORPHOLOGICAL_CSV_PATH, TOPOLOGICAL_CSV_PATH, VISUALIZATION_WSI_PATH, VISUALIZATION_GRAPH_PATH, WSI_ID, MORPHOLOGICAL_PROPERTIES, CENTROID_ID, VISUALIZATION_CENTROID, VISUALIZATION_GRAPH, GRAPH, UNET_THRESHOLD, RESULTS_PATH, EXPERIMENT_NAME
from utils import plot_centroids, plot_graph_wsi, compute_graph, extract_graph_topology, compute_graph_slidelvl
from morphology import MorphologyStudy

if __name__ == "__main__":

    random.seed(2023)
    np.random.seed(2023)

    eps = np.finfo(float).eps
    results_csv = sorted(glob.glob(MORPHOLOGICAL_CSV_PATH + '/*.csv'))
    # features = ['Surface', 'Perimeter', 'Convex hull surface', 'Convex hull perimeter', 'Circularity', 'Form factor', 'Convexity', 'Roughness', 'Proximity']
    features = ['Surface', 'Perimeter', 'Convex hull surface', 'Convex hull perimeter', 'Circularity', 'Form factor', 'Convexity', 'Roughness']
    X = []
    for csv_file in results_csv:
        results = pd.read_csv(csv_file)

        # Normalize all columns
        for column in features:
            # results[column] = (results[column] - results[column].min()) / (results[column].max() - results[column].min() + eps)
            results[column] = (np.log(results[column] / results[column].min())) / (np.log(results[column].max() / (results[column].min() + eps)))

        X.append(results.loc[:,features].mean().to_numpy())
    
    X = np.asarray(X)
    # print(X.shape)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit(X.T)
    explained_variance = np.sum(pca.explained_variance_ratio_[:2]) * 100
    print(f"The first two principal components explain the {explained_variance:.2f}% of the total variance.")

    # print(X_pca.components_)
    # print(np.expand_dims(X_pca.components_[0],axis=1).shape)
    X_components = np.concatenate((np.expand_dims(X_pca.components_[0], axis=1), np.expand_dims(X_pca.components_[1], axis=1)), axis=1)
    # print(X_components)


    model = KMeans(n_clusters=2)    # define the model
    model.fit(X_components)    # fit the model
    yhat = model.predict(X_components) # assign a cluster to each example
    clusters = unique(yhat) # retrieve unique clusters

    colors = ['#DF2020','#2095DF']
    # colors = ['#DF2020','#2095DF','#81DF20']

    labels = ['genetic AD', 'sporadic AD']
    Y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    centroids = model.cluster_centers_
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = list(range(1,16))
    cluster_map['cluster'] = model.labels_
    cluster_map['centroid'] = cluster_map.cluster.map({0:[centroids[0,0],centroids[0,1]], 1:[centroids[1,0],centroids[1,1]]})
    # cluster_map['centroid'] = cluster_map.cluster.map({0:[centroids[0,0],centroids[0,1]], 1:[centroids[1,0],centroids[1,1]], 2:[centroids[2,0],centroids[2,1]]})
    cluster_map['c'] = cluster_map.cluster.map({0:colors[0], 1:colors[1]})
    # cluster_map['c'] = cluster_map.cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})
    print(cluster_map)

    fig = plt.figure(figsize=(7, 7))
    plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color=colors, marker="^")
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        for i in row_ix[0]:
            # print(i)
            # print((X_components[i, 0], cluster_map['centroid'][i][0]),(X_components[i, 1], cluster_map['centroid'][i][1]))
            if (cluster_map['cluster'][i] == Y[i]):
                print(cluster_map['cluster'][i], Y[i])
                plt.scatter(X_components[i, 0], X_components[i, 1], c=cluster_map['c'][i], marker='o')
            else:
                plt.scatter(X_components[i, 0], X_components[i, 1], c=cluster_map['c'][i], marker='x')

            plt.annotate(cluster_map['data_index'][i], (X_components[i, 0], X_components[i, 1]), textcoords="offset points", xytext=(10,0), ha='center')
            plt.plot([X_components[i, 0], cluster_map['centroid'][i][0]],[X_components[i, 1], cluster_map['centroid'][i][1]], c=cluster_map['c'][i], alpha=0.2)
    
    # legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cluster {}'.format(i+1), markerfacecolor=mcolor, markersize=5) for i, mcolor in enumerate(colors)]
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=mlabel, markerfacecolor=mcolor, markersize=5) for mlabel, mcolor in zip(labels, colors)]
    # legend_elements.extend([Line2D([0], [0], marker='^', color='w', label='Centroid - C{}'.format(i+1), markerfacecolor=mcolor, markersize=10) for i, mcolor in enumerate(colors)])
    plt.legend(handles=legend_elements, loc='upper left', ncol=2)

    # show the plot
    plt.title('Morphological clustering of plaques in WSI')
    plt.xlabel('PC1')
    plt.ylabel('PC2')        
    plt.savefig(os.path.join(EXPERIMENT_PATH, 'morphological_analysis.png'), dpi=300)
    plt.close()

    cluster_map.to_csv(os.path.join(EXPERIMENT_PATH, 'morphological_cluster.csv'))


