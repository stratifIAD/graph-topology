import yaml
import matplotlib.pyplot as plt
import os
import networkx as nx
import numpy as np
from libpysal.cg import voronoi_frames
from libpysal import weights, examples
from scipy.spatial import distance
import pandas as pd
import pprint

def get_config():
    with open('config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def plot_centroids(df, wsi_name, save_path, figsize=20, size=2, title=True, dpi=300):
    fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    ax.scatter(x=df['centroid-0'], y=df['centroid-1'], s=size)
    if title: 
        ax.set_title(f'{wsi_name}')
    plt.savefig(os.path.join(save_path, f'{wsi_name}.png'), dpi=dpi)
    plt.close()
    
def plot_graph(G, pos, wsi_name, save_path, options, figsize=10, title=True, dpi=300, axis=False, dropout=False, ):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.patch.set_facecolor('white')
    options=options
    
    nx.draw_networkx(G, pos, ax=ax, **options)
    
    if title:
        ax.set_title(f'{wsi_name}')
    if title and dropout > 0: 
        ax.set_title(f'{wsi_name} with dropout = {dropout}')
    if not axis:    
        ax.axis('off')
        
    plt.savefig(os.path.join(save_path, f'{wsi_name}.png'), dpi=dpi)
    plt.close()

def plot_graph_wsi(G, pos, divider, factor, wsi, wsi_name, gray_matter, extend_factor, save_path, options, figsize=10, title=True, dpi=300, axis=False, dropout=False, ):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.patch.set_facecolor('white')
    options=options

    ax.imshow(wsi, extent=[0,extend_factor[0]/divider[0],0,extend_factor[1]/divider[1]],origin='lower')
    for coords in gray_matter:
        ax.plot(coords[:,0]/factor/divider[0],coords[:,1]/factor/divider[1],'--b',linewidth=1.5)
    nx.draw_networkx(G, pos, ax=ax, **options)

    if title:
        ax.set_title(f'{wsi_name}')
    if title and dropout > 0: 
        ax.set_title(f'{wsi_name} with dropout = {dropout}')
    if not axis:    
        ax.axis('off')

    plt.savefig(os.path.join(save_path, f'{wsi_name}.png'), dpi=dpi)
    plt.close()
    
def compute_graph(df, weight):
    centroids = df[['centroid-0', 'centroid-1']].values
    points = [(float(x[0]), float(x[1])) for x in centroids]
    points_array = np.array(points)
    x_max = points_array[:, 0].max()
    y_max = points_array[:, 1].max()
    divider = np.array([x_max, y_max])
    points = [(float(x[0]), float(x[1])) for x in points_array / divider]
    pos = {i: point for i, point in enumerate(points)}
    cells, generators = voronoi_frames(points)
    
    delaunay = weights.Rook.from_dataframe(cells)
    G = delaunay.to_networkx()

    weighted_G = nx.Graph()
    weighted_G.add_nodes_from(G.nodes())
    
    if weight:
        for edges in G.edges():
            weighted_G.add_edge(*edges, weight=1/distance.euclidean(centroids[edges[0]], centroids[edges[1]]))
    else:
        for edges in G.edges():
            weighted_G.add_edge(*edges, 1)

    G = weighted_G.copy()

    # Erode a graph
    '''
    eroded_G = nx.Graph()
    for edge in G.edges(data=True):
        print(edge)
        if (edge[2]['weight'] > 1e-3):
            eroded_G.add_edge(edge[0], edge[1], distance=edge[2]['weight'])
    
    adj_matrix = nx.to_numpy_matrix(G)
    eroded_graph = np.copy(adj_matrix)
    num_nodes = adj_matrix.shape[0]

    for i in range(num_nodes):
        for j in range(num_nodes):
            aux = adj_matrix[i, j]
            if (aux < 1e-3):
                eroded_graph[i, j] = 0
            else:
                eroded_graph[i, j] = aux

    eroded_graph_G = nx.from_numpy_matrix(eroded_graph)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(nx.to_numpy_array(G).shape)
    pp.pprint(nx.to_numpy_array(G))
    pp.pprint(nx.to_numpy_array(eroded_graph_G).shape)
    pp.pprint(nx.to_numpy_array(eroded_graph_G))

    counts, bins = np.histogram(adj_matrix)
    plt.figure(1)
    plt.stairs(counts, bins)
    # plt.show()

    counts, bins = np.histogram(eroded_graph)
    plt.figure(2)
    plt.stairs(counts, bins)
    plt.show()
    '''
    # return eroded_G, pos
    return G, pos

def compute_graph_slidelvl(df, weight, factor):
    centroids = df[['centroid-0', 'centroid-1']].values
    points = [(float(x[0])/factor, float(x[1])/factor) for x in centroids]
    points_array = np.array(points)
    x_max = points_array[:, 0].max()
    y_max = points_array[:, 1].max()
    divider = np.array([x_max, y_max])
    points = [(float(x[0]), float(x[1])) for x in points_array / divider]
    pos = {i: point for i, point in enumerate(points)}
    cells, generators = voronoi_frames(points)
    
    delaunay = weights.Rook.from_dataframe(cells)
    G = delaunay.to_networkx()

    weighted_G = nx.Graph()
    weighted_G.add_nodes_from(G.nodes())
    
    if weight:
        for edges in G.edges():
            weighted_G.add_edge(*edges, weight=1/distance.euclidean(centroids[edges[0]], centroids[edges[1]]))
    else:
        for edges in G.edges():
            weighted_G.add_edge(*edges, 1)

    G = weighted_G.copy()
    
    return G, pos, divider


def extract_graph_topology(G):
    
    tmp_dict = {}
    
    tmp_dict['order'] = G.order()
    tmp_dict['size'] = G.size()
    tmp_dict['diameter'] = nx.diameter(G)
    tmp_dict['radius'] = nx.radius(G)
    tmp_dict['average_shortest_path_length'] = nx.average_shortest_path_length(G)
    tmp_dict['density'] = nx.density(G)
    tmp_dict['graph_clique_number'] = nx.graph_clique_number(G)
    tmp_dict['graph_number_of_cliques'] = nx.graph_number_of_cliques(G)
    tmp_dict['transitivity'] = nx.transitivity(G)
    tmp_dict['local_efficiency'] = nx.local_efficiency(G)
    tmp_dict['global_efficiency'] = nx.global_efficiency(G)

    tmp = list(dict(G.degree()).values())
    tmp_dict['degree_avg'] = float(np.mean(tmp))
    tmp_dict['degree_min'] = float(np.min(tmp))
    tmp_dict['degree_max'] = float(np.max(tmp))
    tmp_dict['degree_qt95'] = float(np.quantile(tmp, 0.95))
    tmp_dict['degree_qt75'] = float(np.quantile(tmp, 0.75))
    tmp_dict['degree_qt66'] = float(np.quantile(tmp, 2/3))
    tmp_dict['degree_qt50'] = float(np.quantile(tmp, 0.5))
    tmp_dict['degree_qt33'] = float(np.quantile(tmp, 1/3))
    tmp_dict['degree_qt20'] = float(np.quantile(tmp, 1/5))
    tmp_dict['degree_qt10'] = float(np.quantile(tmp, 0.1))
    tmp_dict['degree_qt05'] = float(np.quantile(tmp, 0.05))

    tmp = list(nx.clustering(G).values())
    tmp_dict['clustering_avg'] = float(np.mean(tmp))
    tmp_dict['clustering_min'] = float(np.min(tmp))
    tmp_dict['clustering_max'] = float(np.max(tmp))
    tmp_dict['clustering_qt95'] = float(np.quantile(tmp, 0.95))
    tmp_dict['clustering_qt75'] = float(np.quantile(tmp, 0.75))
    tmp_dict['clustering_qt66'] = float(np.quantile(tmp, 2/3))
    tmp_dict['clustering_qt50'] = float(np.quantile(tmp, 0.5))
    tmp_dict['clustering_qt33'] = float(np.quantile(tmp, 1/3))
    tmp_dict['clustering_qt20'] = float(np.quantile(tmp, 1/5))
    tmp_dict['clustering_qt10'] = float(np.quantile(tmp, 0.1))
    tmp_dict['clustering_qt05'] = float(np.quantile(tmp, 0.05))

    tmp = list(nx.triangles(G).values())
    tmp_dict['triangles_avg'] = float(np.mean(tmp))
    tmp_dict['triangles_min'] = float(np.min(tmp))
    tmp_dict['triangles_max'] = float(np.max(tmp))
    tmp_dict['triangles_qt95'] = float(np.quantile(tmp, 0.95))
    tmp_dict['triangles_qt75'] = float(np.quantile(tmp, 0.75))
    tmp_dict['triangles_qt66'] = float(np.quantile(tmp, 2/3))
    tmp_dict['triangles_qt50'] = float(np.quantile(tmp, 0.5))
    tmp_dict['triangles_qt33'] = float(np.quantile(tmp, 1/3))
    tmp_dict['triangles_qt20'] = float(np.quantile(tmp, 1/5))
    tmp_dict['triangles_qt10'] = float(np.quantile(tmp, 0.1))
    tmp_dict['triangles_qt05'] = float(np.quantile(tmp, 0.05))

    tmp = list(nx.degree_centrality(G).values())
    tmp_dict['degree_centrality_avg'] = float(np.mean(tmp))
    tmp_dict['degree_centrality_min'] = float(np.min(tmp))
    tmp_dict['degree_centrality_max'] = float(np.max(tmp))
    tmp_dict['degree_centrality_qt95'] = float(np.quantile(tmp, 0.95))
    tmp_dict['degree_centrality_qt75'] = float(np.quantile(tmp, 0.75))
    tmp_dict['degree_centrality_qt66'] = float(np.quantile(tmp, 2/3))
    tmp_dict['degree_centrality_qt50'] = float(np.quantile(tmp, 0.5))
    tmp_dict['degree_centrality_qt33'] = float(np.quantile(tmp, 1/3))
    tmp_dict['degree_centrality_qt20'] = float(np.quantile(tmp, 1/5))
    tmp_dict['degree_centrality_qt10'] = float(np.quantile(tmp, 0.1))
    tmp_dict['degree_centrality_qt05'] = float(np.quantile(tmp, 0.05)) 

    tmp = list(nx.closeness_centrality(G).values())
    tmp_dict['closeness_centrality_avg'] = float(np.mean(tmp))
    tmp_dict['closeness_centrality_min'] = float(np.min(tmp))
    tmp_dict['closeness_centrality_max'] = float(np.max(tmp))
    tmp_dict['closeness_centrality_qt95'] = float(np.quantile(tmp, 0.95))
    tmp_dict['closeness_centrality_qt75'] = float(np.quantile(tmp, 0.75))
    tmp_dict['closeness_centrality_qt66'] = float(np.quantile(tmp, 2/3))
    tmp_dict['closeness_centrality_qt50'] = float(np.quantile(tmp, 0.5))
    tmp_dict['closeness_centrality_qt33'] = float(np.quantile(tmp, 1/3))
    tmp_dict['closeness_centrality_qt20'] = float(np.quantile(tmp, 1/5))
    tmp_dict['closeness_centrality_qt10'] = float(np.quantile(tmp, 0.1))
    tmp_dict['closeness_centrality_qt05'] = float(np.quantile(tmp, 0.05)) 
    
    tmp_df = pd.DataFrame([tmp_dict])
    
    return tmp_df