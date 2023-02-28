import sys
import numpy as np
from run import execute_experiment
import utils_values

if __name__ == "__main__":
    original_argv = sys.argv.copy()
    for dataset_name in utils_values.other_small_dataset_names:
        args = [dataset_name, 'hybrids',
                '--exp_subset',
                '--redo_exp',
                # '--redo_tuning'
                ]
        kwargs = {}

        kwargs.update(**{
            '--eps': utils_values.eps_values,
            '--sample_variations': utils_values.sample_variation,
            '--exp_fractions': 1,
            '--grid_fractions': 1,
            '--base_model_code': 'lr',
        })
        execute_experiment(args, kwargs, original_argv)
