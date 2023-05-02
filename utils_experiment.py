fractions = [0.001, 0.004, 0.016, 0.063, 0.251, 1]  # np.geomspace(0.001,1,7) np.linspace(0.001,1,7)

ACS_dataset_names = [
    'ACSPublicCoverage',  # 1138289
    'ACSEmployment',  # 3236107
    'ACSIncomePovertyRatio',  # 3236107
    'ACSHealthInsurance',  # 3236107
    'ACSIncome',  # 1664500
    'ACSMobility',  # 620937
    'ACSTravelTime',  # 1466648
    'ACSEmploymentFiltered'  # 2590315
]

dataset_names = ['adult'] + ACS_dataset_names

sigmod_datasets = ['compas', 'german', 'adult_sigmod']

sigmod_dataset_map = dict(zip(['compas', 'german', 'adult_sigmod'], ['CompasDataset', 'GermanDataset', 'AdultDataset']))

sigmod_datasets_aif360 = [x + '_aif360' for x in sigmod_datasets]

sample_variation = range(2)

fixed_sample_frac = 0.1

eps = 0.01  # , 0.001] # 0.05 old value
eps_values = [.005, 0.01, 0.02, 0.05, 0.10, 0.2]
# eps_values = [0.0001, 0.001, 0.01, 0.05]

index_cols = ['random_seed', 'train_test_fold', 'sample_seed', 'train_test_seed', 'base_model_code', 'constraint_code',
              'iterations', 'dataset']

experiment_configurations = [
    {'experiment_id': 's_h_1.0.TEST',
     'dataset_names': ['adult_sigmod', 'german', 'compas', ],
     'model_names': ['hybrids'],
     'params': ['--exp_subset', '--redo_tuning'],
     'eps': [0.001, 0.01],
     'sample_seeds': [0],
     'exp_fractions': [0.251],
     'grid_fractions': [1],
     'base_model_code': ['lr', 'lgbm'],
     'random_seeds': 0,
     'train_test_seeds': [0],
     'constraint_code': 'dp'},
    {'experiment_id': 's_o_1.0',
     'dataset_names': ['german_aif360', 'compas_aif360', 'adult_sigmod_aif360'],
     'model_names': ['Calmon', 'ZafarDI'],
     'random_seeds': 0,
     'train_test_seeds': [0, 1],
     'base_model_code': ['lr', 'lgbm'],
     },
    {'experiment_id': 's_h_1.0',
     'dataset_names': ['compas', 'german', 'adult_sigmod'],
     'model_names': ['hybrids'],
     'params': ['--exp_subset', '--redo_tuning'],
     'eps': [0.001, .005, 0.01, 0.02, 0.05, 0.10],
     'sample_seeds': range(2),
     'exp_fractions': [0.251, 1],
     'grid_fractions': [1],
     'base_model_code': ['lr', 'lgbm'],
     'random_seeds': range(3),
     'train_test_seeds': range(3),
     'constraint_code': 'dp'},
    {'experiment_id': 's_c_1.0',
     'dataset_names': ['german_aif360', 'compas_aif360', 'adult_sigmod_aif360'],
     'model_names': ['Calmon'],
     'random_seeds': range(3),
     'train_test_seeds': range(3),
     'base_model_code': ['lr', 'lgbm'],
     },
    {'experiment_id': 's_zDI_1.0',
     'dataset_names': ['german_aif360', 'compas_aif360', 'adult_sigmod_aif360'],
     'model_names': ['ZafarDI'],
     'random_seeds': range(3),
     'train_test_seeds': range(3),
     'base_model_code': ['lr', 'lgbm'],
     },
    {'experiment_id': 's_tr_1.0',
     'dataset_names': ['german', 'compas', 'adult_sigmod'],
     'model_names': ['ThresholdOptimizer'],
     'random_seeds': range(3),
     'train_test_seeds': range(3),
     'base_model_code': ['lr', 'lgbm'],
     },
    {'experiment_id': 's_hardt_1.0',
     'dataset_names': ['german_aif360', 'compas_aif360', 'adult_sigmod_aif360'],
     'model_names': ['Hardt'],
     'random_seeds': range(3),
     'train_test_seeds': range(3),
     'base_model_code': ['lr', 'lgbm'],
     },
]
