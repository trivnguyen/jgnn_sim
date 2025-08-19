
import os
import glob
import datetime

import numpy as np

from jeans_gnn.utils import dataset
from project_utils import envs, paths

input_root = envs.DEFAULT_GALAXIES_DIR
input_name = "gnfw_lpzhao_profiles/train_gnfw_lpzhaoCore_beta_priorlarge_pois100.hdf5"
output_root = envs.DEFAULT_GALAXIES_DIR
output_name = "gnfw_lpzhao_profiles/gnfw_lpzhaoCore_beta_priorlarge_pois100"
num_samples_per_file = 10000

os.makedirs(os.path.join(output_root, output_name), exist_ok=True)

# read in dataset
input_path = os.path.join(input_root, input_name)
node_features, graph_features, headers = dataset.read_graph_dataset(input_path)
num_graphs = len(node_features['pos'])

i_graph = 0
i_file = 0
while i_graph < num_graphs:
    out_node_features, out_graph_features = {}, {}
    out_headers = headers.copy()

    for key in node_features.keys():
        out_node_features[key] = node_features[key][
            i_graph:i_graph+num_samples_per_file]
        out_node_features[key] = np.concatenate(out_node_features[key])

    for key in graph_features.keys():
        out_graph_features[key] = graph_features[key][
            i_graph:i_graph+num_samples_per_file]

    # update headers
    out_headers['num_galaxies'] = len(out_graph_features['num_stars'])
    out_headers['date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # write dataset
    output_path = os.path.join(output_root, output_name, 'data.{:d}.hdf5'.format(i_file))
    print("Writing to {}".format(output_path))
    dataset.write_graph_dataset(
        output_path, out_node_features, out_graph_features,
        out_graph_features['num_stars'], out_headers)

    i_graph += num_samples_per_file
    i_file += 1