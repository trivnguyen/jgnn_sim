
# import modules
import argparse
import datetime
import logging
import h5py
import os
import time
from tqdm import tqdm

import pandas as pd
import agama
import astropy.units as u
import numpy as np
import yaml

DEFAULT_GALAXIES_DIR = "/mnt/ceph/users/tnguyen/jeans_gnn/datasets/raw_datasets"
DEFAULT_DATASETS_DIR = "/mnt/ceph/users/tnguyen/jeans_gnn/datasets/processed_datasets"
N_MAX_ITER = 1000

# set agama unit to be in Msun, kpc, km/s
agama.setUnits(mass=1 * u.Msun, length=1*u.kpc, velocity=1 * u.km /u.s)

# define logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to the config file')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path to the output file. If not specified, the output will be '
             'saved to the default galaxy directory with the same name. The default '
             'directory is specified in the config file')
    parser.add_argument(
        '--name', type=str, default='default',
        help='Name of galaxy sets to generate')
    parser.add_argument(
        '--num-galaxies', type=int, default=10000,
        help='Number of galaxies to sample')
    return parser.parse_args()

# IO function
def write_graph_dataset(
        path, node_features, graph_features, lengths, headers=None):
    """ Write a graph dataset with node features and graph features into HDF5
    file. The node features of all graphs are concatenated into a single
    array. The lengths of each graph is required to split the node features
    into individual graphs.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.
    node_features : dict
        Dictionary of node features. The node features of all graphs are
        concatenated into a single array. The key is the name of the feature and
        the value is a list of arrays of shape (M, ) where M is the number
        of nodes in all graphs.
    graph_features : dict
        Dictionary of graph features. The key is the name of the feature and
        the value is a list of arrays of shape (N, ) where N is the number
        of graphs.
    lengths : list
        List of lengths of each graph. The length of the list is the number of
        graphs. The sum of the lengths must be equal to M, or the total number
        of nodes in all graphs.
    headers : dict
        Dictionary of headers. The key is the name of the header and the value
        is the value of the header.
    Returns
    -------
    None
    """
    # check if total length of node features is equal to the sum of lengths
    assert np.sum(lengths) == len(list(node_features.values())[0])

    # construct pointer to each graph
    ptr = np.cumsum(lengths)

    # define headers
    default_headers ={
        "node_features": list(node_features.keys()),
        "graph_features": list(graph_features.keys()),
    }
    default_headers['all_features'] = (
        default_headers['node_features'] + default_headers['graph_features'])
    if headers is None:
        headers = default_headers
    else:
        headers.update(default_headers)

    # write dataset into HDF5 file
    with h5py.File(path, 'w') as f:
        # write pointers
        f.create_dataset('ptr', data=ptr)

        # write node features
        for key in node_features:
            dset = f.create_dataset(key, data=node_features[key])
            dset.attrs.update({'type': 'node_features'})

        # write tree features
        for key in graph_features:
            dset = f.create_dataset(key, data=graph_features[key])
            dset.attrs.update({'type': 'graph_features'})

        # write headers
        f.attrs.update(headers)

# sample parameters from a config file
def sample_parameters(config, num_galaxies=10000):
    """ Sample parameters from a list of parameter configurations
    Parameters
    ----------
    config : list
        List of parameter configurations
    num_galaxies : int
        Number of galaxies to sample
    Returns
    -------
    parameters : dict
        Dictionary of sampled parameters
    """
    parameters = {}
    for c in config:
        name = c['name']
        dist = c['dist']
        print(name, dist)
        if dist == 'uniform':
            parameters[name] = np.random.uniform(
                c['min'], c['max'], num_galaxies)
        elif dist == 'log_uniform':
            parameters[name] = 10**np.random.uniform(
                np.log10(c['min']), np.log10(c['max']), num_galaxies)
        elif dist == 'delta':
            parameters[name] = np.full(num_galaxies, c['value'])
        else:
            raise ValueError('Invalid distribution {}'.format(dist))
    # convert to DataFrame
    parameters = pd.DataFrame(parameters)
    return parameters

# parse sampled parameters into Agama format
def parse_parameters(
        dm_type, stellar_type, df_type,
        dm_params, stellar_params, df_params,
        return_params=False):
    """ Parse sampled parameters into Agama format
    Parameters
    ----------
    dm_type : str
        DM potential type
    stellar_type : str
        Stellar density profile type
    df_type : str
        DF type
    dm_params : dict
        DM potential parameters
    stellar_params : dict
        Stellar density profile parameters
    df_params : dict
        DF parameters
    return_params : bool
        Whether to return the parsed parameters
    Returns
    -------
    galaxy_model : agama.GalaxyModel
        Galaxy model
    params : dict
        Parsed parameters
    """
    # parse the DM parameters
    dm_params['densityNorm'] = dm_params.pop('rho_0')
    dm_params['scaleRadius'] = dm_params.pop('r_dm')
    dm_params['axisRatioY'] = dm_params.pop('q', 1)
    dm_params['axisRatioZ'] = dm_params.pop('p', 1)

    # parse the stellar parameters
    # if stellar_type == "Spheroid":
    #     stellar_params['gamma'] = stellar_params.pop('gamma_star')
    #     stellar_params['beta'] = stellar_params.pop('beta_star')
    #     stellar_params['alpha'] = stellar_params.pop('alpha_star')
    #     stellar_params['densityNorm'] = stellar_params.pop('rho_0_star')

    if stellar_params.get('r_star') is not None:
        stellar_params['scaleRadius'] = stellar_params.pop('r_star')
    elif stellar_params.get('r_star_r_dm') is not None:
        stellar_params['scaleRadius'] = (
            stellar_params.pop('r_star_r_dm') * dm_params['scaleRadius'])
    if stellar_params.get('q') is not None:
        stellar_params['axisRatioY'] = stellar_params.pop('q')
    elif stellar_params.get('q_star_q_dm') is not None:
        stellar_params['axisRatioY'] = (
            stellar_params.pop('q_star_q_dm') * dm_params['axisRatioY'])
    if stellar_params.get('p') is not None:
        stellar_params['axisRatioZ'] = stellar_params.pop('p')
    elif stellar_params.get('p_star_p_dm') is not None:
        stellar_params['axisRatioZ'] = (
            stellar_params.pop('p_star_p_dm') * dm_params['axisRatioZ'])

    # parse the DF parameters
    if df_params.get('r_a') is not None:
        df_params['r_a'] = df_params.pop('r_a')
    elif df_params.get('r_a_r_dm') is not None:
        df_params['r_a'] = (
            df_params.pop('r_a_r_dm') * dm_params['scaleRadius'])
    elif df_params.get('r_a_r_star') is not None:
        df_params['r_a'] = (
            df_params.pop('r_a_r_star') * stellar_params['scaleRadius'])

    # construct the DM potential, stellar density, and DF
    dm_potential = agama.Potential(
        type=dm_type, **dm_params)
    stellar_density = agama.Potential(
        type=stellar_type, mass=1, **stellar_params)  # set mass to small value
    df = agama.DistributionFunction(
        type=df_type, potential=dm_potential, density=stellar_density,
        **df_params)

    # construct galaxy model
    galaxy_model = agama.GalaxyModel(dm_potential, df)

    if return_params:
        # summarize params
        params = {'dm': dm_params, 'stellar': stellar_params, 'df': df_params}
        return galaxy_model , params
    return galaxy_model


def main():
    """ Sample the 6D stellar kinematics of dwarf galaxies """
    FLAGS = parse_args()

    # Load config file
    logger.info('Loading config file {}'.format(FLAGS.config))
    with open(FLAGS.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Sample parameters and number of stars
    logger.info('Sampling parameters for {} galaxies'.format(FLAGS.num_galaxies))
    dm_parameters = sample_parameters(
        config['dm_potential']['parameters'], FLAGS.num_galaxies)
    stellar_parameters = sample_parameters(
        config['stellar_density']['parameters'], FLAGS.num_galaxies)
    df_parameters = sample_parameters(
        config['distribution_function']['parameters'], FLAGS.num_galaxies)

    # sample number of stars
    if config['galaxy']['num_stars']['dist'] == 'uniform':
        num_stars = np.random.randint(
            config['galaxy']['num_stars']['min'],
            config['galaxy']['num_stars']['max'], FLAGS.num_galaxies)
    elif config['galaxy']['num_stars']['dist'] == 'poisson':
        num_stars = np.random.poisson(
            lam=config['galaxy']['num_stars']['mean'], size=FLAGS.num_galaxies)
    elif config['galaxy']['num_stars']['dist'] == 'delta':
        num_stars = np.full(
            FLAGS.num_galaxies, config['galaxy']['num_stars']['value'])
    else:
        raise ValueError('Invalid distribution for stars {}'.format(
            config['galaxy']['num_stars']['dist']))

    # Iterate over galaxies and sample the stellar kinematics
    logger.info('Sampling stellar kinematics')

    posvels = []
    index_failed = []
    # iterate over galaxies
    for i in tqdm(range(FLAGS.num_galaxies), mininterval=1):
        # construct galaxy model
        success = False
        iter = 0
        while (not success) and (iter < N_MAX_ITER):
            try:
                gal = parse_parameters(
                    config['dm_potential']['type'], config['stellar_density']['type'],
                    config['distribution_function']['type'],
                    dm_parameters.iloc[i].to_dict(),
                    stellar_parameters.iloc[i].to_dict(),
                    df_parameters.iloc[i].to_dict())
                posvel_gal, _ = gal.sample(num_stars[i])
                success = True

                # add parsed parameters and kinematics to list
                posvels.append(posvel_gal)

            except Exception as e:
                iter += 1
                # print(e, gal, num_stars[i])
                pass
        if not success:
            logger.warning('Failed to sample galaxy {} after {} iterations'.format(
                i, N_MAX_ITER))
            index_failed.append(i)
            posvels.append(np.full((num_stars[i], 6), np.nan))

        # NOTE: there may be a memory leak in Agama, so we need to explicitly
        # delete the galaxy model
        del gal

    # convert kinematics to Numpy array
    posvels = np.array(posvels, dtype='object')

    # Prepare data for HDF5 file
    # create node features
    node_features = {}
    node_features['pos'] = np.concatenate(
        [posvels[i][:, :3] for i in range(len(posvels))])
    node_features['vel'] = np.concatenate(
        [posvels[i][:, 3:] for i in range(len(posvels))])

    # create graph features from parameters
    # Adding prefix to DataFrame column avoid name collisions
    dm_parameters = dm_parameters.add_prefix('dm_')
    stellar_parameters = stellar_parameters.add_prefix('stellar_')
    df_parameters = df_parameters.add_prefix('df_')
    parameters = pd.concat(
        [dm_parameters, stellar_parameters, df_parameters], axis=1)

    graph_features = parameters.to_dict('list')
    for k in graph_features:
        graph_features[k] = np.array(graph_features[k])
    graph_features['num_stars'] = num_stars

    # create headers
    headers = {
        'name': FLAGS.name,
        'node_features': list(node_features.keys()),
        'graph_features': list(graph_features.keys()),
        'dm_type': config['dm_potential']['type'],
        'stellar_type': config['stellar_density']['type'],
        'df_type': config['distribution_function']['type'],
        'num_galaxies': FLAGS.num_galaxies,
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Write kinematics to HDF5 file
    if FLAGS.output is None:
        # if no output file is specified, use the name from the config file
        # get config file name without extension
        config_name = os.path.splitext(os.path.basename(FLAGS.config))[0]

        # create output file name as "{name}_{config_name}.hdf5"
        FLAGS.output = os.path.join(
            DEFAULT_GALAXIES_DIR,
            '{}_{}.hdf5'.format(FLAGS.name, config_name)
        )

    logger.info('Writing kinematics to {}'.format(FLAGS.output))
    write_graph_dataset(
        FLAGS.output, node_features, graph_features, num_stars,
        headers=headers)

if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    logger.info('Time taken: {:.2f} seconds'.format(t2-t1))
