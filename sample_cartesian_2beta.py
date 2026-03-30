
# import modules
import argparse
import datetime
import logging
import h5py
import os
import time
from tqdm import tqdm

import importlib.util

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
            try:
                dset = f.create_dataset(key, data=node_features[key])
                # dset.attrs.update({'type': 'node_features'})
            except Exception as e:
                print(e)
                print(key, node_features[key].shape, node_features[key])
                print(node_features[key])
                raise e

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


def _sample_one(spec, num_galaxies):
    """Draw `num_galaxies` samples from a single distribution spec.

    Works with both plain dicts (YAML) and ml_collections ConfigDicts.
    """
    dist = spec['dist']
    if dist == 'uniform':
        return np.random.uniform(spec['min'], spec['max'], num_galaxies)
    elif dist == 'log_uniform':
        return 10 ** np.random.uniform(
            np.log10(spec['min']), np.log10(spec['max']), num_galaxies)
    elif dist == 'delta':
        return np.full(num_galaxies, spec['value'])
    else:
        raise ValueError('Unknown distribution: {}'.format(dist))


def sample_parameters_configdict(params_config, num_galaxies):
    """Sample parameters from an ml_collections ConfigDict.

    Unlike the YAML-style ``sample_parameters`` (which takes a *list* of
    ``{name, dist, …}`` dicts), this function accepts a ConfigDict whose
    *keys* are parameter names and whose *values* are distribution specs::

        params_config.gamma  = ConfigDict({'dist': 'uniform', 'min': -1, 'max': 2})
        params_config.rho_0  = ConfigDict({'dist': 'log_uniform', 'min': 1e3, 'max': 1e10})

    This makes per-parameter CLI overrides possible, e.g.
    ``--config.dm_potential.params.gamma.min=-2``.

    Parameters
    ----------
    params_config : ml_collections.ConfigDict
        ConfigDict where every item is a distribution spec.
    num_galaxies : int

    Returns
    -------
    pd.DataFrame
    """
    parameters = {}
    for name, spec in params_config.items():
        print(name, spec['dist'])
        parameters[name] = _sample_one(spec, num_galaxies)
    return pd.DataFrame(parameters)


def load_config(path):
    """Load a config file and return ``(config, fmt)``.

    Supports two formats:

    * **YAML** (``*.yaml`` / ``*.yml``) — existing format; returns a plain
      ``dict`` and ``fmt='yaml'``.
    * **Python** (``*.py``) — new ml_collections format; the file must expose
      ``get_config() -> ml_collections.ConfigDict``.  Returns the ConfigDict
      and ``fmt='py'``.

    Parameters
    ----------
    path : str

    Returns
    -------
    config : dict | ml_collections.ConfigDict
    fmt : str   ``'yaml'`` or ``'py'``
    """
    if path.endswith('.py'):
        spec = importlib.util.spec_from_file_location('_galaxy_config', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.get_config(), 'py'
    else:
        with open(path, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader), 'yaml'


def _get(config, key, fmt, default=None):
    """Uniform attribute access for both plain dicts and ConfigDicts."""
    if fmt == 'py':
        return getattr(config, key, default)
    else:
        return config.get(key, default)


def _get_type(section, fmt):
    """Return the agama type string from a config section."""
    if fmt == 'py':
        return section.type
    else:
        return section['type']


def _get_params(section, fmt):
    """Return the parameter sub-config from a config section."""
    if fmt == 'py':
        return section.params
    else:
        return section['parameters']


def _sample_section(section, num_galaxies, fmt):
    """Sample parameters for one config section (DM, stellar, or DF)."""
    params = _get_params(section, fmt)
    if fmt == 'py':
        return sample_parameters_configdict(params, num_galaxies)
    else:
        return sample_parameters(params, num_galaxies)


def _parse_dm_stellar_params(dm_type, stellar_type, dm_params, stellar_params):
    """Parse and build a DM potential and stellar density from raw parameter dicts.

    Parameters
    ----------
    dm_type : str
    stellar_type : str
    dm_params : dict   — copy is made internally; original is not mutated
    stellar_params : dict

    Returns
    -------
    dm_potential : agama.Potential
    stellar_density : agama.Density
    dm_params : dict    — parsed (agama-ready) DM params
    stellar_params : dict — parsed (agama-ready) stellar params
    """
    dm_params = dm_params.copy()
    stellar_params = stellar_params.copy()

    # rename DM params to agama convention
    dm_params['densityNorm'] = dm_params.pop('rho_0')
    dm_params['scaleRadius'] = dm_params.pop('r_dm')
    dm_params['axisRatioY'] = dm_params.pop('q', 1)
    dm_params['axisRatioZ'] = dm_params.pop('p', 1)

    # resolve stellar scale radius
    if stellar_params.get('r_star') is not None:
        stellar_params['scaleRadius'] = stellar_params.pop('r_star')
    elif stellar_params.get('r_star_r_dm') is not None:
        stellar_params['scaleRadius'] = (
            stellar_params.pop('r_star_r_dm') * dm_params['scaleRadius'])

    # resolve stellar axis ratios
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

    dm_potential = agama.Potential(type=dm_type, **dm_params)
    stellar_density = agama.Density(type=stellar_type, mass=1, **stellar_params)

    return dm_potential, stellar_density, dm_params, stellar_params


def _parse_df_params(df_type, dm_potential, stellar_density,
                     df_params, dm_params, stellar_params):
    """Build a DistributionFunction from raw DF params plus parsed DM/stellar dicts.

    Parameters
    ----------
    df_type : str
    dm_potential : agama.Potential
    stellar_density : agama.Density
    df_params : dict   — copy is made internally
    dm_params : dict   — parsed DM params (agama-ready keys)
    stellar_params : dict — parsed stellar params (agama-ready keys)

    Returns
    -------
    df : agama.DistributionFunction
    df_params : dict  — parsed DF params
    """
    df_params = df_params.copy()

    # resolve anisotropy radius
    if df_params.get('r_a_r_dm') is not None:
        df_params['r_a'] = df_params.pop('r_a_r_dm') * dm_params['scaleRadius']
    elif df_params.get('r_a_r_star') is not None:
        df_params['r_a'] = df_params.pop('r_a_r_star') * stellar_params['scaleRadius']

    df = agama.DistributionFunction(
        type=df_type, potential=dm_potential, density=stellar_density, **df_params)

    return df, df_params


# parse sampled parameters into Agama format (single-population, backward-compat)
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
    params : dict  (only if return_params=True)
    """
    dm_potential, stellar_density, dm_params, stellar_params = (
        _parse_dm_stellar_params(dm_type, stellar_type, dm_params, stellar_params))
    df, df_params = _parse_df_params(
        df_type, dm_potential, stellar_density, df_params, dm_params, stellar_params)

    galaxy_model = agama.GalaxyModel(dm_potential, df)

    if return_params:
        params = {'dm': dm_params, 'stellar': stellar_params, 'df': df_params}
        return galaxy_model, params
    return galaxy_model


def main():
    """ Sample the 6D stellar kinematics of dwarf galaxies """
    FLAGS = parse_args()

    # Load config — supports both YAML and ml_collections Python configs
    logger.info('Loading config file {}'.format(FLAGS.config))
    config, fmt = load_config(FLAGS.config)
    logger.info('Config format: {}'.format(fmt))

    # Detect two-population mode.
    # YAML uses dict keys; Python configs use ConfigDict attributes.
    if fmt == 'py':
        is_two_pop = hasattr(config, 'df1') and hasattr(config, 'df2')
        dm_section = config.dm_potential
        stellar_section = config.stellar_density
        df1_section = config.df1 if is_two_pop else None
        df2_section = config.df2 if is_two_pop else None
        df_section = getattr(config, 'df', None)
        galaxy_section = config.galaxy
    else:
        is_two_pop = (
            'distribution_function_1' in config and
            'distribution_function_2' in config)
        dm_section = config['dm_potential']
        stellar_section = config['stellar_density']
        df1_section = config.get('distribution_function_1')
        df2_section = config.get('distribution_function_2')
        df_section = config.get('distribution_function')
        galaxy_section = config['galaxy']

    logger.info('Mode: {}'.format('two-population' if is_two_pop else 'single-population'))

    # Sample DM and stellar parameters (shared across both populations)
    logger.info('Sampling parameters for {} galaxies'.format(FLAGS.num_galaxies))
    dm_parameters = _sample_section(dm_section, FLAGS.num_galaxies, fmt)
    stellar_parameters = _sample_section(stellar_section, FLAGS.num_galaxies, fmt)

    if is_two_pop:
        df1_parameters = _sample_section(df1_section, FLAGS.num_galaxies, fmt)
        df2_parameters = _sample_section(df2_section, FLAGS.num_galaxies, fmt)

        # Enforce beta0_1 <= beta0_2 by sorting the independently drawn values.
        # This gives a uniform joint prior on the ordered pair while keeping
        # each marginal uniform.
        if 'beta0' in df1_parameters.columns and 'beta0' in df2_parameters.columns:
            b1 = df1_parameters['beta0'].values.copy()
            b2 = df2_parameters['beta0'].values.copy()
            df1_parameters['beta0'] = np.minimum(b1, b2)
            df2_parameters['beta0'] = np.maximum(b1, b2)

        # Sample mixture fraction f = fraction of stars from population 1
        if fmt == 'py':
            mix_spec = galaxy_section.mixture
            mixture_fractions = _sample_one(mix_spec, FLAGS.num_galaxies)
        else:
            mix_cfg = galaxy_section.get(
                'mixture', {'dist': 'uniform', 'min': 0.0, 'max': 1.0})
            mixture_fractions = sample_parameters(
                [{'name': 'f', **mix_cfg}], FLAGS.num_galaxies)['f'].values
    else:
        df_parameters = _sample_section(df_section, FLAGS.num_galaxies, fmt)

    # Sample number of stars per galaxy
    star_cfg = galaxy_section['num_stars'] if fmt == 'yaml' else galaxy_section.num_stars
    if star_cfg['dist'] == 'uniform':
        num_stars = np.random.randint(
            star_cfg['min'], star_cfg['max'], FLAGS.num_galaxies)
    elif star_cfg['dist'] == 'poisson':
        num_stars = np.random.poisson(
            lam=star_cfg['mean'], size=FLAGS.num_galaxies)
    elif star_cfg['dist'] == 'delta':
        num_stars = np.full(FLAGS.num_galaxies, star_cfg['value'])
    else:
        raise ValueError('Invalid distribution for stars {}'.format(star_cfg['dist']))

    # Iterate over galaxies and sample the stellar kinematics
    logger.info('Sampling stellar kinematics')

    dm_type = _get_type(dm_section, fmt)
    stellar_type = _get_type(stellar_section, fmt)
    df1_type = _get_type(df1_section, fmt) if is_two_pop else None
    df2_type = _get_type(df2_section, fmt) if is_two_pop else None
    df_type = _get_type(df_section, fmt) if not is_two_pop else None

    posvels = []
    index_failed = []

    for i in tqdm(range(FLAGS.num_galaxies), mininterval=1):
        success = False
        iter_count = 0

        while (not success) and (iter_count < N_MAX_ITER):
            try:
                dm_pot, stellar_dens, dm_p, stellar_p = _parse_dm_stellar_params(
                    dm_type, stellar_type,
                    dm_parameters.iloc[i].to_dict(),
                    stellar_parameters.iloc[i].to_dict())

                if is_two_pop:
                    df1, _ = _parse_df_params(
                        df1_type, dm_pot, stellar_dens,
                        df1_parameters.iloc[i].to_dict(), dm_p, stellar_p)
                    df2, _ = _parse_df_params(
                        df2_type, dm_pot, stellar_dens,
                        df2_parameters.iloc[i].to_dict(), dm_p, stellar_p)

                    gal1 = agama.GalaxyModel(dm_pot, df1)
                    gal2 = agama.GalaxyModel(dm_pot, df2)

                    f = mixture_fractions[i]
                    n1 = int(np.round(f * num_stars[i]))
                    n2 = num_stars[i] - n1

                    parts = []
                    if n1 > 0:
                        pv1, _ = gal1.sample(n1)
                        parts.append(pv1)
                    if n2 > 0:
                        pv2, _ = gal2.sample(n2)
                        parts.append(pv2)
                    posvel_gal = np.vstack(parts) if parts else np.zeros((0, 6))

                    del gal1, gal2

                else:
                    df, _ = _parse_df_params(
                        df_type, dm_pot, stellar_dens,
                        df_parameters.iloc[i].to_dict(), dm_p, stellar_p)
                    gal = agama.GalaxyModel(dm_pot, df)
                    posvel_gal, _ = gal.sample(num_stars[i])
                    del gal

                success = True
                posvels.append(posvel_gal)

            except Exception:
                iter_count += 1

        if not success:
            logger.warning('Failed to sample galaxy {} after {} iterations'.format(
                i, N_MAX_ITER))
            index_failed.append(i)
            posvels.append(np.full((num_stars[i], 6), np.nan))

    # Convert kinematics to Numpy array
    posvels = np.array(posvels, dtype='object')

    # Prepare node features
    node_features = {}
    node_features['pos'] = np.concatenate(
        [posvels[i][:, :3] for i in range(len(posvels))]).astype(np.float32)
    node_features['vel'] = np.concatenate(
        [posvels[i][:, 3:] for i in range(len(posvels))]).astype(np.float32)

    # Prepare graph features from sampled parameters
    dm_parameters = dm_parameters.add_prefix('dm_')
    stellar_parameters = stellar_parameters.add_prefix('stellar_')

    if is_two_pop:
        df1_parameters = df1_parameters.add_prefix('df1_')
        df2_parameters = df2_parameters.add_prefix('df2_')
        parameters = pd.concat(
            [dm_parameters, stellar_parameters, df1_parameters, df2_parameters], axis=1)
    else:
        df_parameters = df_parameters.add_prefix('df_')
        parameters = pd.concat(
            [dm_parameters, stellar_parameters, df_parameters], axis=1)

    graph_features = {k: np.array(v) for k, v in parameters.to_dict('list').items()}
    graph_features['num_stars'] = num_stars
    if is_two_pop:
        graph_features['mixture_f'] = mixture_fractions

    # Build headers
    headers = {
        'name': FLAGS.name,
        'config_format': fmt,
        'dm_type': dm_type,
        'stellar_type': stellar_type,
        'df_type': df_type if not is_two_pop else '{}/{}'.format(df1_type, df2_type),
        'two_population': is_two_pop,
        'num_galaxies': FLAGS.num_galaxies,
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Write to HDF5
    if FLAGS.output is None:
        config_name = os.path.splitext(os.path.basename(FLAGS.config))[0]
        FLAGS.output = os.path.join(
            DEFAULT_GALAXIES_DIR,
            '{}_{}.hdf5'.format(FLAGS.name, config_name))

    logger.info('Writing kinematics to {}'.format(FLAGS.output))
    write_graph_dataset(
        FLAGS.output, node_features, graph_features, num_stars, headers=headers)

if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    logger.info('Time taken: {:.2f} seconds'.format(t2-t1))
