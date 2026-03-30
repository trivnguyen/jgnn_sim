"""Draco I target config for sample_galaxies_target.py (ml_collections).

Usage
-----
Run with default settings::

    python sample_galaxies_target.py \
        --config galaxy_configs_new/draco1_gnfw_beta_target_config.py \
        --num-galaxies 10000 \
        --name draco1_gnfw_beta

Override individual fields from the CLI (ml_collections style)::

    python sample_galaxies_target.py \
        --config galaxy_configs_new/draco1_gnfw_beta_target_config.py \
        --config.galaxy.n_oversample=200000 \
        --config.galaxy.num_stars.dist=delta \
        --config.galaxy.num_stars.value=300

Notes
-----
- ``num_stars.dist: match_data`` sets the number of mock stars equal to
  the number of observed member stars in the catalog.
- ``r_dm`` is sampled log-uniformly in [rhalf * 10^dex_min, rhalf * 10^dex_max].
- ``derive_rho0.method: wolf_mass`` causes the script to derive rho_0 from
  the Wolf mass estimator rather than sampling it freely.  rhalf is drawn
  from N(meta.rhalf_kpc, uncertainty) and shared between the Wolf mass
  calculation and the stellar Plummer scale radius.
- ``n_oversample`` controls how many raw agama samples are drawn before
  the Rproj resampling step; must be larger than num_stars.
"""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    # Galaxy settings
    config.galaxy = ml_collections.ConfigDict()
    config.galaxy.num_stars = ml_collections.ConfigDict({
        'dist': 'poisson',
        'mean': 100
    })
    config.galaxy.n_oversample = 200

    # Target observational data
    config.target = ml_collections.ConfigDict()
    config.target.key = 'bootes_1'
    config.target.catalog_path = (
        '/mnt/home/tnguyen/projects/jeans_gnn/dsph_datasets/vr_catalogs/'
        'bootes1-ting/bootes1.ecsv')
    config.target.source = 'bootes1_ting'
    config.target.loader_kwargs = ml_collections.ConfigDict({
        'mem_prob_min': 0.8,
        'vlos_abs_max': 50.0,
        'instrument': 'avg',
        'apply_perspective_corr': True,
    })

    # DM potential
    # rho_0 is NOT sampled here — it is derived from the Wolf mass (see derive_rho0).
    config.dm_potential = ml_collections.ConfigDict()
    config.dm_potential.type = 'Spheroid'
    config.dm_potential.params = ml_collections.ConfigDict({
        'alpha': {'dist': 'delta', 'value': 1},
        'beta':  {'dist': 'delta', 'value': 3},
        'gamma': {'dist': 'uniform', 'min': -1, 'max': 2},
        'r_dm':  {'dist': 'meta_log_uniform_around_rhalf', 'dex_min': -1, 'dex_max': 3},
    })

    # Derive rho_0 from the Wolf mass estimator.
    # The script samples log10(M_wolf) ~ N(meta.log_mass_wolf, uncertainty)
    # and rhalf ~ N(meta.rhalf_kpc, uncertainty), then solves
    # rho_0 = M_wolf / M_enclosed(1.74 * rhalf, r_dm, alpha, beta, gamma).
    # The same rhalf draw is used as the Plummer scale radius (r_star).
    config.dm_potential.derive_rho0 = ml_collections.ConfigDict({
        'method': 'wolf_mass',
    })

    # Stellar density — r_star is set by the derive_rho0 block (no params needed)
    config.stellar_density = ml_collections.ConfigDict()
    config.stellar_density.type = 'Plummer'
    config.stellar_density.params = ml_collections.ConfigDict()

    # Distribution function (Cuddeford-Osipkov-Merritt)
    config.df = ml_collections.ConfigDict()
    config.df.type = 'QuasiSpherical'
    config.df.params = ml_collections.ConfigDict({
        'beta0': {'dist': 'uniform', 'min': -0.499, 'max': 0.999},
        'r_a_r_star': {'dist': 'log_uniform', 'min': 0.1, 'max': 1000},
    })

    return config