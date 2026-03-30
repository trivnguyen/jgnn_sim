"""
sample_galaxies_target.py

Generate mock kinematic catalogs matched to a specific observational target.

Unlike sample_galaxies.py (abstract Cartesian phase-space), this script:
  1. Loads the target's observed catalog and metadata.
  2. Places each mock galaxy at the target's sky position / distance.
  3. Transforms agama phase-space samples to ICRS observables.
  4. Resamples mock stars to match the observed projected-radius distribution.
  5. Draws velocity errors from the empirical error distribution of the data.
  6. Adds Gaussian noise to line-of-sight velocities.
  7. Saves everything to an HDF5 file.

Config format: Python ml_collections (*.py exposing get_config()).
"""

# import modules
import argparse
import datetime
import importlib.util
import logging
import os
import time
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import h5py
import numpy as np
import pandas as pd
import scipy.special as sc
import scipy.stats as stats
import agama
import astropy.units as u
from astropy.coordinates import SkyCoord
from tqdm import tqdm

from dsph_analysis import kinematic_io

DEFAULT_GALAXIES_DIR = "/mnt/ceph/users/tnguyen/jeans_gnn/datasets/processed_datasets"
N_MAX_ITER = 1000
R_WOLF_FACTOR = 1.74  # r_wolf = R_WOLF_FACTOR * r_half (Wolf et al. 2010)

agama.setUnits(mass=1 * u.Msun, length=1*u.kpc, velocity=1 * u.km/u.s)

# define logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


# ---------------------------------------------------------------------------
# Dark matter mass profile
# ---------------------------------------------------------------------------

def M_enclosed_normalized(r, r_s, alpha, beta, gamma):
    """Enclosed mass of a generalised NFW profile with rho_s = 1 (Msun/kpc^3).

    M(r) = rho_s * M_enclosed_normalized(r, r_s, alpha, beta, gamma)

    Parameters
    ----------
    r     : float   evaluation radius (kpc)
    r_s   : float   DM scale radius (kpc)
    alpha : float   transition sharpness
    beta  : float   outer slope
    gamma : float   inner slope
    """
    r_n = r / r_s
    a1 = (3.0 - gamma) / alpha
    a2 = (beta - gamma) / alpha
    a3 = 1.0 + (3.0 - gamma) / alpha
    a4 = -(r_n ** alpha)
    c1 = (4.0 * np.pi * r_s**3) / (3.0 - gamma)
    c2 = r_n ** (3.0 - gamma)
    return c1 * c2 * sc.hyp2f1(a1, a2, a3, a4)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Sample mock target-matched galaxy catalogs.')
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to an ml_collections Python config file (*.py).')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path to the output HDF5 file. Defaults to '
             'DEFAULT_GALAXIES_DIR/<name>_<config_stem>.hdf5.')
    parser.add_argument(
        '--name', type=str, default='default',
        help='Name tag for this galaxy set.')
    parser.add_argument(
        '--num-galaxies', type=int, default=10000,
        help='Number of mock galaxies to generate.')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path):
    """Load a Python ml_collections config file and return the ConfigDict."""
    if not path.endswith('.py'):
        raise ValueError(
            'sample_galaxies_target.py only supports Python ml_collections '
            'configs (*.py). Got: {}'.format(path))
    spec = importlib.util.spec_from_file_location('_galaxy_config', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------

def _sample_one(spec, num_galaxies, meta=None):
    """Draw num_galaxies samples from a single distribution spec.

    Meta-aware distribution types (require meta != None):
      meta_rhalf                   fixed to meta.rhalf_kpc
      meta_rhalf_gaussian          N(rhalf, 0.5*(ep+em)), clipped > 0
      meta_log_uniform_around_rhalf  log-uniform in
                                   [rhalf * 10^dex_min, rhalf * 10^dex_max]
    """
    dist = spec['dist']
    if dist == 'uniform':
        return np.random.uniform(spec['min'], spec['max'], num_galaxies)
    elif dist == 'log_uniform':
        return 10 ** np.random.uniform(
            np.log10(spec['min']), np.log10(spec['max']), num_galaxies)
    elif dist == 'delta':
        return np.full(num_galaxies, spec['value'])
    elif dist == 'meta_rhalf':
        if meta is None:
            raise ValueError(
                'dist: meta_rhalf requires target metadata to be loaded first.')
        return np.full(num_galaxies, meta.rhalf_kpc.value)
    elif dist == 'meta_rhalf_gaussian':
        if meta is None:
            raise ValueError(
                'dist: meta_rhalf_gaussian requires target metadata.')
        rhalf_mean = meta.rhalf_kpc.value
        rhalf_err = 0.5 * (meta.rhalf_kpc_ep.value + meta.rhalf_kpc_em.value)
        return np.abs(np.random.normal(rhalf_mean, rhalf_err, num_galaxies))
    elif dist == 'meta_log_uniform_around_rhalf':
        if meta is None:
            raise ValueError(
                'dist: meta_log_uniform_around_rhalf requires target metadata.')
        log_rhalf = np.log10(meta.rhalf_kpc.value)
        return 10 ** np.random.uniform(
            log_rhalf + spec['dex_min'], log_rhalf + spec['dex_max'], num_galaxies)
    else:
        raise ValueError('Unknown distribution: {}'.format(dist))


def sample_parameters_configdict(params_config, num_galaxies, meta=None):
    """Sample parameters from an ml_collections ConfigDict.

    Keys are parameter names; values are distribution specs.
    meta is passed through to _sample_one so that dist: meta_rhalf works.
    """
    parameters = {}
    for name, spec in params_config.items():
        print(name, spec['dist'])
        parameters[name] = _sample_one(spec, num_galaxies, meta=meta)
    return pd.DataFrame(parameters)


def _resample_row(i, dm_parameters, stellar_parameters, df_parameters,
                  dm_section, stellar_section, df_section, meta, derive_rho0_cfg):
    """Resample all parameters for galaxy i in-place (used on retry)."""
    for df_out, section in [
        (dm_parameters, dm_section),
        (stellar_parameters, stellar_section),
        (df_parameters, df_section),
    ]:
        new = sample_parameters_configdict(section.params, 1, meta=meta)
        for col in new.columns:
            df_out.at[i, col] = new.iloc[0][col]

    if derive_rho0_cfg is not None and derive_rho0_cfg.method == 'wolf_mass':
        rhalf_i = abs(np.random.normal(
            meta.rhalf_kpc.value,
            0.5 * (meta.rhalf_kpc_ep.value + meta.rhalf_kpc_em.value)))
        mwolf_i = 10 ** np.random.normal(
            meta.log_mass_wolf,
            0.5 * (meta.log_mass_wolf_ep + meta.log_mass_wolf_em))
        rho_0_i = mwolf_i / M_enclosed_normalized(
            R_WOLF_FACTOR * rhalf_i,
            dm_parameters.at[i, 'r_dm'],
            dm_parameters.at[i, 'alpha'],
            dm_parameters.at[i, 'beta'],
            dm_parameters.at[i, 'gamma'])
        dm_parameters.at[i, 'rho_0'] = rho_0_i
        stellar_parameters.at[i, 'r_star'] = rhalf_i


# ---------------------------------------------------------------------------
# Agama model construction helpers (mirrors sample_galaxies.py)
# ---------------------------------------------------------------------------

def _parse_dm_stellar_params(dm_type, stellar_type, dm_params, stellar_params):
    """Parse and build a DM potential and stellar density from raw parameter dicts."""
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
    """Build a DistributionFunction from raw DF params plus parsed DM/stellar dicts."""
    df_params = df_params.copy()

    # resolve anisotropy radius
    if df_params.get('r_a') is not None:
        pass  # already in agama units
    elif df_params.get('r_a_r_dm') is not None:
        df_params['r_a'] = df_params.pop('r_a_r_dm') * dm_params['scaleRadius']
    elif df_params.get('r_a_r_star') is not None:
        df_params['r_a'] = df_params.pop('r_a_r_star') * stellar_params['scaleRadius']

    df = agama.DistributionFunction(
        type=df_type, potential=dm_potential, density=stellar_density, **df_params)

    return df, df_params


# ---------------------------------------------------------------------------
# Coordinate transformation
# ---------------------------------------------------------------------------

def transform_to_icrs(pos_cartesian, vel_cartesian, meta):
    """Transform agama Cartesian phase-space samples to ICRS observables.

    Parameters
    ----------
    pos_cartesian : ndarray, shape (N, 3)   kpc, galaxy-centric Cartesian
    vel_cartesian : ndarray, shape (N, 3)   km/s
    meta : DwarfMeta

    Returns
    -------
    mock_icrs_true : SkyCoord   full 3-D ICRS (true distance kept)
    mock_icrs_obs  : SkyCoord   distance collapsed to systemic
    """
    target_icrs = SkyCoord(
        ra=meta.ra,
        dec=meta.dec,
        distance=meta.distance,
        pm_ra_cosdec=meta.pmra,
        pm_dec=meta.pmdec,
        radial_velocity=meta.vlos_systemic,
        frame='icrs',
    )
    target_gala = target_icrs.galactocentric

    mock_gala = SkyCoord(
        x=pos_cartesian[:, 0] * u.kpc + target_gala.x,
        y=pos_cartesian[:, 1] * u.kpc + target_gala.y,
        z=pos_cartesian[:, 2] * u.kpc + target_gala.z,
        v_x=vel_cartesian[:, 0] * u.km/u.s + target_gala.v_x,
        v_y=vel_cartesian[:, 1] * u.km/u.s + target_gala.v_y,
        v_z=vel_cartesian[:, 2] * u.km/u.s + target_gala.v_z,
        frame='galactocentric',
        representation_type='cartesian',
    )
    mock_icrs_true = mock_gala.transform_to('icrs')

    # collapse individual star distances to the systemic distance
    mock_icrs_obs = SkyCoord(
        ra=mock_icrs_true.ra,
        dec=mock_icrs_true.dec,
        distance=meta.distance,
        pm_ra_cosdec=mock_icrs_true.pm_ra_cosdec,
        pm_dec=mock_icrs_true.pm_dec,
        radial_velocity=mock_icrs_true.radial_velocity,
        frame='icrs',
    )
    return mock_icrs_true, mock_icrs_obs


# ---------------------------------------------------------------------------
# Projected-radius resampling
# ---------------------------------------------------------------------------

def resample_rproj(Rproj_mock, Rproj_data, num_stars, bins=None):
    """Resample mock stars to match the observed projected-radius distribution.

    Parameters
    ----------
    Rproj_mock : ndarray   projected radii of oversampled mock stars (kpc)
    Rproj_data : ndarray   projected radii of observed member stars (kpc)
    num_stars  : int       number of stars to select
    bins       : array_like, optional
        Bin edges (kpc). Default: 20 log-spaced bins from 0.001 to 10 kpc.

    Returns
    -------
    indices : ndarray, shape (num_stars,)
    """
    if bins is None:
        bins = np.logspace(np.log10(0.001), np.log10(10), 20)

    hist_mock, bin_edges = np.histogram(Rproj_mock, bins=bins, density=True)
    hist_data, _ = np.histogram(Rproj_data, bins=bins, density=True)
    hist_ratio = np.where(hist_mock > 0, hist_data / hist_mock, 0.0)

    bin_indices = np.digitize(Rproj_mock, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(hist_ratio) - 1)
    weights = hist_ratio[bin_indices]

    weight_sum = weights.sum()
    if weight_sum == 0:
        raise ValueError(
            'All resampling weights are zero — '
            'check Rproj range against bin edges.')
    weights /= weight_sum

    # use replace=True only as fallback if the oversampled pool is too small
    replace = len(Rproj_mock) < num_stars
    if replace:
        logger.warning(
            'n_oversample (%d) < num_stars (%d); sampling with replacement.',
            len(Rproj_mock), num_stars)

    return np.random.choice(len(Rproj_mock), size=num_stars,
                            replace=replace, p=weights)


# ---------------------------------------------------------------------------
# HDF5 writer (same as sample_galaxies.py)
# ---------------------------------------------------------------------------

def write_graph_dataset(path, node_features, graph_features, lengths,
                        headers=None):
    assert np.sum(lengths) == len(list(node_features.values())[0])
    ptr = np.cumsum(lengths)

    default_headers = {
        'node_features': list(node_features.keys()),
        'graph_features': list(graph_features.keys()),
    }
    default_headers['all_features'] = (
        default_headers['node_features'] + default_headers['graph_features'])
    if headers is None:
        headers = default_headers
    else:
        headers.update(default_headers)

    with h5py.File(path, 'w') as f:
        f.create_dataset('ptr', data=ptr)

        for key, arr in node_features.items():
            try:
                f.create_dataset(key, data=arr)
            except Exception as e:
                print(e, key, arr.shape)
                raise

        for key, arr in graph_features.items():
            dset = f.create_dataset(key, data=arr)
            dset.attrs.update({'type': 'graph_features'})

        f.attrs.update(headers)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FLAGS = parse_args()

    # Load config
    logger.info('Loading config: {}'.format(FLAGS.config))
    config = load_config(FLAGS.config)

    dm_section = config.dm_potential
    stellar_section = config.stellar_density
    df_section = config.df
    galaxy_section = config.galaxy
    target_cfg = config.target

    dm_type = dm_section.type
    stellar_type = stellar_section.type
    df_type = df_section.type

    n_oversample = int(galaxy_section.get('n_oversample', 100_000))

    # Load target metadata
    logger.info('Loading target metadata: key={}'.format(target_cfg.key))
    meta_path = target_cfg.get('meta_path', None)
    if meta_path:
        meta = kinematic_io.load_meta(target_cfg.key, meta_path=meta_path)
    else:
        meta = kinematic_io.load_meta(target_cfg.key)

    # Load observational catalog
    logger.info('Loading observational catalog: {}'.format(target_cfg.catalog_path))
    loader_kwargs = dict(target_cfg.loader_kwargs)
    data = kinematic_io.load_kinematic_data(
        target_cfg.catalog_path,
        meta=meta,
        source=target_cfg.source,
        **loader_kwargs,
    )
    Rproj_data = data.R_proj.to_value(u.kpc)
    verr_data = data.vlos_err.to_value(u.km/u.s)
    logger.info('Observed catalog: {} member stars'.format(len(Rproj_data)))

    # Fit empirical velocity-error distribution
    hist_verr, bins_verr = np.histogram(verr_data, bins=20, density=True)
    rv_verr = stats.rv_histogram((hist_verr, bins_verr))

    # SkyCoord of target center (used for Rproj calculation)
    target_icrs_center = SkyCoord(
        ra=meta.ra, dec=meta.dec, distance=meta.distance, frame='icrs')

    # Sample model parameters for all galaxies
    logger.info('Sampling parameters for {} galaxies'.format(FLAGS.num_galaxies))
    dm_parameters = sample_parameters_configdict(
        dm_section.params, FLAGS.num_galaxies, meta=meta)
    stellar_parameters = sample_parameters_configdict(
        stellar_section.params, FLAGS.num_galaxies, meta=meta)
    df_parameters = sample_parameters_configdict(
        df_section.params, FLAGS.num_galaxies, meta=meta)

    # Derive rho_0 from Wolf mass (if requested by config)
    log_mwolf_samples = None
    derive_rho0_cfg = getattr(dm_section, 'derive_rho0', None)
    if derive_rho0_cfg is not None and derive_rho0_cfg.method == 'wolf_mass':
        # shared rhalf sample: used for both stellar r_star and the Wolf radius
        rhalf_mean = meta.rhalf_kpc.value
        rhalf_err = 0.5 * (meta.rhalf_kpc_ep.value + meta.rhalf_kpc_em.value)
        rhalf_samples = np.abs(
            np.random.normal(rhalf_mean, rhalf_err, FLAGS.num_galaxies))

        # sample log10(M_wolf) within observational uncertainty
        log_mwolf_mean = meta.log_mass_wolf
        log_mwolf_err = 0.5 * (meta.log_mass_wolf_ep + meta.log_mass_wolf_em)
        log_mwolf_samples = np.random.normal(
            log_mwolf_mean, log_mwolf_err, FLAGS.num_galaxies)
        mwolf_samples = 10 ** log_mwolf_samples

        # rho_0 = M_wolf / M_enclosed(r_wolf, r_dm, alpha, beta, gamma)
        rwolf_samples = R_WOLF_FACTOR * rhalf_samples
        rho_0_samples = np.array([
            mwolf_samples[j] / M_enclosed_normalized(
                rwolf_samples[j],
                dm_parameters.iloc[j]['r_dm'],
                dm_parameters.iloc[j]['alpha'],
                dm_parameters.iloc[j]['beta'],
                dm_parameters.iloc[j]['gamma'])
            for j in range(FLAGS.num_galaxies)])

        dm_parameters['rho_0'] = rho_0_samples
        stellar_parameters['r_star'] = rhalf_samples
        logger.info('rho_0 derived from Wolf mass; r_star set to sampled rhalf')

    # Sample number of stars
    star_cfg = galaxy_section.num_stars
    if star_cfg['dist'] == 'match_data':
        num_stars = np.full(FLAGS.num_galaxies, len(Rproj_data), dtype=int)
    elif star_cfg['dist'] == 'uniform':
        num_stars = np.random.randint(
            star_cfg['min'], star_cfg['max'], FLAGS.num_galaxies)
    elif star_cfg['dist'] == 'poisson':
        num_stars = np.random.poisson(
            lam=star_cfg['mean'], size=FLAGS.num_galaxies)
    elif star_cfg['dist'] == 'delta':
        num_stars = np.full(FLAGS.num_galaxies, int(star_cfg['value']), dtype=int)
    else:
        raise ValueError('Unknown num_stars dist: {}'.format(star_cfg['dist']))

    # Main sampling loop
    logger.info('Sampling stellar kinematics (n_oversample={})'.format(n_oversample))

    acc = {k: [] for k in (
        'ra', 'dec', 'pmra_cosdec', 'pmdec',
        'vlos', 'vlos_err', 'vlos_true',
        'distance', 'distance_true', 'R_proj')}
    index_failed = []

    for i in tqdm(range(FLAGS.num_galaxies), mininterval=1):
        n_stars = int(num_stars[i])
        success = False

        for attempt in range(N_MAX_ITER):
            # resample fresh parameters on every retry (first attempt uses pre-sampled)
            if attempt > 0:
                _resample_row(
                    i, dm_parameters, stellar_parameters, df_parameters,
                    dm_section, stellar_section, df_section, meta, derive_rho0_cfg)

            try:
                dm_pot, stellar_dens, dm_p, stellar_p = _parse_dm_stellar_params(
                    dm_type, stellar_type,
                    dm_parameters.iloc[i].to_dict(),
                    stellar_parameters.iloc[i].to_dict())

                df, _ = _parse_df_params(
                    df_type, dm_pot, stellar_dens,
                    df_parameters.iloc[i].to_dict(), dm_p, stellar_p)

                gal = agama.GalaxyModel(dm_pot, df)
                samples_over, _ = gal.sample(n_oversample)
                del gal

                # coordinate transformation
                mock_icrs_true, mock_icrs_obs = transform_to_icrs(
                    samples_over[:, :3], samples_over[:, 3:], meta)

                # projected radius (works because distance is collapsed to systemic)
                Rproj_mock = mock_icrs_obs.separation_3d(
                    target_icrs_center).to_value(u.kpc)

                # resample to match observed Rproj distribution
                idx = resample_rproj(Rproj_mock, Rproj_data, n_stars)

                mock_obs = mock_icrs_obs[idx]
                mock_true = mock_icrs_true[idx]

                # velocity errors sampled from the empirical distribution
                verr_mock = rv_verr.rvs(size=n_stars)

                # add Gaussian noise to vlos
                vlos_true = mock_obs.radial_velocity.to_value(u.km/u.s)
                vlos_noisy = vlos_true + np.random.normal(0, verr_mock)

                # collect
                acc['ra'].append(
                    mock_obs.ra.to_value(u.deg).astype(np.float32))
                acc['dec'].append(
                    mock_obs.dec.to_value(u.deg).astype(np.float32))
                acc['pmra_cosdec'].append(
                    mock_obs.pm_ra_cosdec.to_value(u.mas/u.yr).astype(np.float32))
                acc['pmdec'].append(
                    mock_obs.pm_dec.to_value(u.mas/u.yr).astype(np.float32))
                acc['vlos'].append(vlos_noisy.astype(np.float32))
                acc['vlos_err'].append(verr_mock.astype(np.float32))
                acc['vlos_true'].append(vlos_true.astype(np.float32))
                acc['distance'].append(
                    mock_obs.distance.to_value(u.kpc).astype(np.float32))
                acc['distance_true'].append(
                    mock_true.distance.to_value(u.kpc).astype(np.float32))
                acc['R_proj'].append(
                    Rproj_mock[idx].astype(np.float32))

                success = True
                break

            except Exception:
                continue

        if not success:
            logger.warning(
                'Galaxy {} skipped after {} resamples.'.format(i, N_MAX_ITER))
            index_failed.append(i)

    # Drop failed galaxies from graph-level parameter arrays
    if index_failed:
        logger.warning('{}/{} galaxies skipped (removed from output).'.format(
            len(index_failed), FLAGS.num_galaxies))
        mask = np.ones(FLAGS.num_galaxies, dtype=bool)
        mask[index_failed] = False
        dm_parameters   = dm_parameters.loc[mask].reset_index(drop=True)
        stellar_parameters = stellar_parameters.loc[mask].reset_index(drop=True)
        df_parameters   = df_parameters.loc[mask].reset_index(drop=True)
        num_stars       = num_stars[mask]
        if log_mwolf_samples is not None:
            log_mwolf_samples = log_mwolf_samples[mask]

    n_succeeded = len(num_stars)

    # Build node feature arrays
    node_features = {k: np.concatenate(v) for k, v in acc.items()}

    # Build graph feature arrays from sampled parameters
    dm_parameters      = dm_parameters.add_prefix('dm_')
    stellar_parameters = stellar_parameters.add_prefix('stellar_')
    df_parameters      = df_parameters.add_prefix('df_')
    parameters = pd.concat(
        [dm_parameters, stellar_parameters, df_parameters], axis=1)

    graph_features = {k: np.array(v) for k, v in parameters.to_dict('list').items()}
    graph_features['num_stars'] = num_stars
    if log_mwolf_samples is not None:
        graph_features['log_mwolf'] = log_mwolf_samples

    # Derived graph features — log-transformed versions matching the Cartesian
    # pre-processed convention so the same config labels work for both streams.
    graph_features['dm_log_r_dm'] = np.log10(graph_features['dm_r_dm'])
    graph_features['dm_log_rho_0'] = np.log10(graph_features['dm_rho_0'])
    graph_features['stellar_log_r_star'] = np.log10(graph_features['stellar_r_star'])
    # absolute anisotropy radius r_a [kpc] and its log
    df_r_a = graph_features['df_r_a_r_star'] * graph_features['stellar_r_star']
    graph_features['df_r_a'] = df_r_a
    graph_features['df_log_r_a'] = np.log10(df_r_a)
    graph_features['df_log_r_a_r_star'] = np.log10(graph_features['df_r_a_r_star'])

    # Build headers
    headers = {
        'name': FLAGS.name,
        'num_galaxies': n_succeeded,
        'n_oversample': n_oversample,
        'dm_type': dm_type,
        'stellar_type': stellar_type,
        'df_type': df_type,
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        # target metadata
        'target_key': meta.key,
        'target_ra_deg': meta.ra.to_value(u.deg),
        'target_dec_deg': meta.dec.to_value(u.deg),
        'target_distance_kpc': meta.distance.to_value(u.kpc),
        'target_pmra_masyr': meta.pmra.to_value(u.mas/u.yr),
        'target_pmdec_masyr': meta.pmdec.to_value(u.mas/u.yr),
        'target_vlos_systemic_kms': meta.vlos_systemic.to_value(u.km/u.s),
        'target_rhalf_kpc': meta.rhalf_kpc.to_value(u.kpc),
        'target_catalog': target_cfg.catalog_path,
        'target_source': target_cfg.source,
        'target_n_obs_stars': len(Rproj_data),
    }

    # Write to HDF5
    if FLAGS.output is None:
        config_name = os.path.splitext(os.path.basename(FLAGS.config))[0]
        FLAGS.output = os.path.join(
            DEFAULT_GALAXIES_DIR,
            '{}_{}.hdf5'.format(FLAGS.name, config_name))

    logger.info('Writing to {}'.format(FLAGS.output))
    write_graph_dataset(
        FLAGS.output, node_features, graph_features, num_stars, headers=headers)

    logger.info('Wrote {} / {} galaxies.'.format(n_succeeded, FLAGS.num_galaxies))


if __name__ == '__main__':
    t0 = time.time()
    main()
    logger.info('Time: {:.1f} s'.format(time.time() - t0))
