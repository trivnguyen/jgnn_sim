
from ml_collections import config_dict

def get_config():

    config = config_dict.ConfigDict()

    config.input_root = "/mnt/ceph/users/tnguyen/jeans_gnn/datasets/raw_datasets"
    config.input_name = "gnfw_profiles/gnfw_beta_priorlarge_uni/data.0.hdf5"
    config.output_root = "/mnt/ceph/users/tnguyen/jeans_gnn/datasets/processed_datasets"
    config.output_name = "gnfw_profiles/gnfw_beta_priorlarge_uni/data.0.hdf5"
    config.seed = 310
    config.num_max_samples = 1000

    config.projection = ['random', ]
    config.num_projection = [1, ]
    config.use_proper_motions = False
    config.min_vdisp = 1e-10  # minimum velocity dispersion in km/s
    config.max_vdisp = 1e10   # maximum velocity dispersion in km/s
    config.min_radius_rstar = 0.  # minimum radius in units of r_star
    config.max_radius_rstar = 10   # maximum radius in units of r_star
    config.min_radius_kpc = 0.  # minimum radius in kpc
    config.max_radius_kpc = 100   # maximum radius in kpc
    config.norm_rstar = False

    config.condition = config_dict.ConfigDict()
    config.condition.fit_lp = False
    config.condition.use_lp_bins = False
    config.condition.use_lp_bins_errors = False
    config.condition.num_lp_bins = 0
    config.lp = config_dict.ConfigDict()
    config.lp.fn = 'plummer'
    config.lp.p0 = [2, 0]  # logL and log_rstar
    config.lp.bounds = ([0, -5], [4, 5])
    config.lp.chi2_max = 100
    config.lp.use_cov = False

    return config
