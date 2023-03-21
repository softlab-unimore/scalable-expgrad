import sys
import numpy as np
from run import execute_experiment
import utils_values

if __name__ == "__main__":
    original_argv = sys.argv.copy()

    for dataset_name in utils_values.dataset_names[:]:
        args = [dataset_name, 'ThresholdOptimizer',
                '--redo_exp',
                # '--redo_tuning'
                ]
        kwargs = {}

        # vary exp sample
        kwargs.update(**{
            '--sample_variations': utils_values.sample_variation,
            '--base_model_code':  'lr', #'lgbm', #
        })
        execute_experiment(args, kwargs, original_argv)
