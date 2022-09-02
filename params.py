import os
import yaml
from utils import get_config


config = get_config()

EXPERIMENT_NAME = config["EXPERIMENT_NAME"]
BASE_PATH = config["BASE_PATH"]

DATA_PATH = os.path.join(BASE_PATH, "data")
RESULTS_PATH = os.path.join(BASE_PATH, "results")
EXPERIMENT_PATH = os.path.join(RESULTS_PATH, EXPERIMENT_NAME)

MORPHOLOGICAL_CSV_PATH = os.path.join(EXPERIMENT_PATH, "csv/morphology")
TOPOLOGICAL_CSV_PATH = os.path.join(EXPERIMENT_PATH, "csv/topology")

VISUALIZATION_WSI_PATH = os.path.join(EXPERIMENT_PATH, "vizualization/wsis")
VISUALIZATION_GRAPH_PATH = os.path.join(EXPERIMENT_PATH, "vizualization/graphs")

WSI_ID = config["WSI_ID"]

MORPHOLOGICAL_PROPERTIES = config['MORPHOLOGICAL_PROPERTIES']
CENTROID_ID = config['CENTROID_ID']

VISUALIZATION_CENTROID = config['VISUALIZATION_CENTROID']
VISUALIZATION_GRAPH = config['VISUALIZATION_GRAPH']

GRAPH = config['GRAPH']

UNET_THRESHOLD = config['UNET_THRESHOLD']