"""Two-population velocity anisotropy config (ml_collections).

Usage
-----
Run with the default settings::

    python sample_galaxies.py \
        --config galaxy_configs/gnfw_profiles/gnfw_2pop_beta_pois100_config.py \
        --num-galaxies 10000

Override individual fields from the CLI (ml_collections style)::

    python sample_galaxies.py \
        --config galaxy_configs/gnfw_profiles/gnfw_2pop_beta_pois100_config.py \
        --config.galaxy.num_stars.mean=500 \
        --config.dm_potential.params.gamma.min=0 \
        --config.df1.params.beta0.max=0.5

Design
------
- **Population 1** is always the *less* radially anisotropic component:
  beta0_1 = min(raw_beta0_1, raw_beta0_2), enforced by sorting in the script.
- **Population 2** is always the *more* radially anisotropic component:
  beta0_2 = max(raw_beta0_1, raw_beta0_2).
- ``r_a`` (anisotropy radius, expressed as a multiple of r_star) is sampled
  independently for each population.
- ``f`` is the fraction of stars drawn from population 1; population 2 gets
  the remaining ``1 - f`` fraction.
"""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.galaxy = ml_collections.ConfigDict()
    config.galaxy.num_stars = ml_collections.ConfigDict({
        'dist': 'poisson',
        'mean': 100,
    })

    # DM potential
    config.dm_potential = ml_collections.ConfigDict()
    config.dm_potential.type = 'Spheroid'
    config.dm_potential.params = ml_collections.ConfigDict({
        'alpha': {'dist': 'uniform', 'min': 0.5, 'max': 3.0},
        'beta':  {'dist': 'uniform', 'min': 1.0, 'max': 10.0},
        'gamma': {'dist': 'uniform', 'min': -1.0, 'max': 2.0},
        'r_dm':  {'dist': 'log_uniform', 'min': 1e-2,  'max': 1e2},
        'rho_0': {'dist': 'log_uniform', 'min': 1e3,   'max': 1e10},
    })

    # Stellar density profiles
    config.stellar_density = ml_collections.ConfigDict()
    config.stellar_density.type = 'Plummer'
    config.stellar_density.params = ml_collections.ConfigDict({
        'r_star_r_dm': {'dist': 'log_uniform', 'min': 0.001, 'max': 1.0},
    })

    # Distribution function (Cuddeford-Osipkov-Merritt)
    config.df = ml_collections.ConfigDict()
    config.df.type = 'QuasiSpherical'
    config.df.params = ml_collections.ConfigDict({
        'beta0': {'dist': 'uniform', 'min': -0.499, 'max': 0.999},
        'r_a_r_star': {'dist': 'log_uniform', 'min': 0.1, 'max': 1000},
    })

    return config
