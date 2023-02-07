fractions = [0.001, 0.004, 0.016, 0.063, 0.251, 1]  # np.geomspace(0.001,1,7) np.linspace(0.001,1,7)

ACS_dataset_names = ['ACSPublicCoverage', # 1138289
                     'ACSEmployment', # 3236107
                     'ACSHealthInsurance', # 3236107
                     'ACSIncomePovertyRatio', # 3236107
                     'ACSIncome', # 1664500
                     'ACSMobility', # 620937
                     'ACSTravelTime', # 1466648
                     'ACSEmploymentFiltered' # 2590315
                     ]

dataset_names = ['adult'] + ACS_dataset_names

sample_variation = range(2)

fixed_sample_frac = 0.1

eps = 0.05