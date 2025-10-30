
import sys
import datetime
import logging
import os
import time

import numpy as np
# import scipy.optimize as opt
import yaml
from tqdm import tqdm
from ml_collections import config_dict
from ml_collections import config_flags
from absl import flags

import utils

try:
    from jgnn_utils import physics
except:
    print('cannot import jgnn_utils, skipping physics')
    physics = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def get_light_profile(rad, n_bins=None):

    # calculate the light profile
    Sigma, Sigma_lo, Sigma_hi, log_R_lo, log_R_hi = physics.particles.calc_Sigma(
        rad, n_bins=n_bins, return_bounds=True)
    log_R = 0.5 * (log_R_lo + log_R_hi)
    log_Sigma = np.log10(Sigma)
    log_Sigma_lo = np.log10(Sigma_lo)
    log_Sigma_hi = np.log10(Sigma_hi)
    log_Sigma_errlo = log_Sigma - log_Sigma_lo
    log_Sigma_errhi = log_Sigma_hi - log_Sigma
    log_Sigma_err = 0.5 * (log_Sigma_errlo + log_Sigma_errhi)

    return log_Sigma, log_Sigma_err, log_R

def fit_lp(rad, fn, p0, bounds, n_bins=None):

    # create fitting function
    if fn == 'zhao':
        def lp_fn(x, log_rho_s, log_r_s, alpha, beta, gamma):
            rho_s = 10**log_rho_s
            r_s = 10**log_r_s
            return np.log10(physics.profiles.Sigma_Zhao(
                x, rho_s, r_s, alpha, beta, gamma))
    elif fn == 'plummer':
        def lp_fn(x, logL, logr_star):
            L = 10**logL
            r_star = 10**logr_star
            return physics.profiles.log_Plummer2d(x, L, r_star)
    else:
        raise ValueError('fn_type not recognized')
    log_Sigma, log_Sigma_err, log_R = get_light_profile(rad, n_bins=n_bins)

    popt, cov = opt.curve_fit(
        lp_fn, 10**log_R, log_Sigma, sigma=log_Sigma_err, p0=p0,
        maxfev=10000, bounds=bounds)

    # calculate the chi2 of the fit averaged over number of bins
    chi2 = np.sum((lp_fn(10**log_R, *popt) - log_Sigma)**2 / log_Sigma_err**2)
    chi2 /= len(log_R)

    return popt, cov, chi2


def main(config: config_dict.ConfigDict,):

    np.random.seed(config.seed)

    input_path = os.path.join(config.input_root, config.input_name)
    output_path = os.path.join(config.output_root, config.output_name)

    node_features, graph_features, headers = utils.read_graph_dataset(
        input_path, to_array=True)

    num_galaxies  = len(node_features['pos'])
    loop = tqdm(
        range(num_galaxies), miniters=num_galaxies//100, desc='Preprocessing galaxies')

    processed_node_features = {
        'pos': [],
        'vel': [],
        'vel_true': [],
        'vel_error': [],
    }
    processed_graph_features = {k: [] for k in graph_features.keys()}
    processed_graph_features['cond'] = []
    processed_graph_features['cond_chi2'] = []

    for i in loop:
        # extract galaxy
        nodes, graph = utils.get_graph(node_features, graph_features, i)
        pos = nodes['pos'].astype(np.float32)
        vel = nodes['vel'].astype(np.float32)
        stellar_rstar = graph['stellar_r_star_r_dm'] * graph['dm_r_dm']

        # apply velocity cut on the 3d velocity
        vel3d = np.linalg.norm(vel, axis=1)
        mask = (vel3d > config.min_v) & (vel3d < config.max_v)
        if np.sum(mask) < len(pos) * 0.5:
            # if half of the stars are outside the velocity range, skip this galaxy
            continue
        pos = pos[mask]
        vel = vel[mask]

        # apply velocity dispersion cut on the 3d velocity dispersion
        vdisp = np.std(vel, axis=0)
        vdisp = np.linalg.norm(vdisp)
        if vdisp < config.min_vdisp or vdisp > config.max_vdisp:
            continue

        # project onto 2d plane
        if config.projection is None:
            projection = [None, ]
        else:
            projection = config.projection

        for ip, project in enumerate(projection):
            for _ in range(config.num_projection[ip]):
                pos_proj_true, vel_proj_true = utils.project2d(
                    pos, vel, axis=project, use_proper_motions=config.use_proper_motions)
                rad_proj_true = np.linalg.norm(pos_proj_true, axis=1)

                # select only the stars within the radius range
                min_radius = max(config.min_radius_rstar * stellar_rstar, config.min_radius_kpc)
                max_radius = min(config.max_radius_rstar * stellar_rstar, config.max_radius_kpc)
                mask = (rad_proj_true > min_radius) & (rad_proj_true < max_radius)
                if np.sum(mask) < len(pos_proj_true) * 0.5:
                    # if half of the stars are outside the radius range, skip this galaxy
                    continue
                pos_proj_true = pos_proj_true[mask]
                vel_proj_true = vel_proj_true[mask]
                rad_proj_true = rad_proj_true[mask]

                # conditioning vectors
                cond = []
                cond_chi2 = []
                if config.condition.use_rstar:
                    cond.append(np.log10(stellar_rstar))
                # if config.condition.fit_lp:
                #     popt, pcov, chi2 = fit_lp(
                #         rad_proj_true, config.lp_fit_param.fn, config.lp_fit_param.p0, config.lp_fit_param.bounds)
                #     if chi2 > config.lp_fit_param.chi2_max:
                #         continue
                #     cond.append(popt)
                #     if config.lp_fit_param.use_cov:
                #         cond.append(pcov.flatten())
                #     cond_chi2.append(chi2)
                # if config.condition.use_lp_bins:
                #     log_Sigma, log_Sigma_err, log_R = get_light_profile(
                #         rad_proj_true, n_bins=config.condition.num_lp_bins)
                #     cond.append(log_R)
                #     cond.append(log_Sigma)
                #     if config.condition.use_lp_bins_errors:
                #         cond.append(log_Sigma_err)
                #     cond_chi2.append(np.zeros_like(log_Sigma))  # no chi2 without fit

                if len(cond) > 0:
                    cond = np.array(cond)
                    cond_chi2 = np.array(cond_chi2)
                    processed_graph_features['cond'].append(cond)
                    processed_graph_features['cond_chi2'].append(cond_chi2)

                # normalize the radius
                if config.norm_rstar:
                    pos_proj_true /= stellar_rstar

                # print(pos_proj_true)

                # add to the new dataset
                processed_node_features['pos'].append(pos_proj_true)
                processed_node_features['vel'].append(vel_proj_true)
                processed_node_features['vel_true'].append(vel_proj_true)
                processed_node_features['vel_error'].append(np.zeros_like(vel_proj_true))

                # add graph features
                for k in graph:
                    processed_graph_features[k].append(graph[k])

    num_galaxies = len(processed_node_features['pos'])
    num_stars = np.array([len(x) for x in processed_node_features['pos']])
    processed_graph_features['num_stars'] = num_stars

    for k in processed_node_features.keys():
        processed_node_features[k] = np.concatenate(processed_node_features[k])

    # calculate all relevant labels
    processed_graph_features = utils.parse_graph_features(
        processed_graph_features, norm_rstar=config.norm_rstar)

    # write the dataset
    default_headers = headers.copy()
    default_headers.update({
        'seed': config.seed,
        'projection': str(config.projection) or 'random',
        'num_projection': config.num_projection,
        'use_proper_motions': config.use_proper_motions,
        'min_vdisp': config.min_vdisp,
        'max_vdisp': config.max_vdisp,
        'min_v': config.min_v,
        'max_v': config.max_v,
        'min_radius_rstar': config.min_radius_rstar,
        'max_radius_rstar': config.max_radius_rstar,
        'min_radius_kpc': config.min_radius_kpc,
        'max_radius_kpc': config.max_radius_kpc,
        'norm_rstar': config.norm_rstar,
        'fit_lp': config.condition.fit_lp,
        'use_lp_bins': config.condition.use_lp_bins,
        'use_lp_bins_errors': config.condition.use_lp_bins_errors,
        'num_lp_bins': config.condition.num_lp_bins or 0,
        'lp_fn': config.lp_fit_param.fn or 'plummer',
        'lp_p0': config.lp_fit_param.p0 or 'default',
        'lp_bounds': config.lp_fit_param.bounds or 'default',
        'use_cov': config.lp_fit_param.use_cov or False,
    })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    utils.write_graph_dataset(
        output_path, processed_node_features, processed_graph_features,
        num_stars, default_headers)


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the preprocessing configuration",
        lock_config=True,
    )
    # Parse flags
    FLAGS(sys.argv)

    # Start training run
    main(config=FLAGS.config)
