import numpy as np

from experiment_routine import execute_experiment
from test.test_synthetic_data import test_get_synthetic_data
import sys

original_argv = sys.argv.copy()


if __name__ == "__main__":
    dataset_name = 'adult'
    args = [dataset_name, 'model_name']
    kwargs = {}

    fractions = [0.0001, 0.0001] # np.geomspace(0.001,1,7) np.linspace(0.001,1,7)
    fixed_sample_frac = 0.001
    args[1] = 'hybrids'
    # make grid-sample vary
    kwargs.update(**{
        '--eps': 0.05,
        '--sample-variations': np.arange(3),
        '--exp-fractions': fixed_sample_frac,
        '--grid-fractions': fractions})
    execute_experiment(args, kwargs, original_argv)
    # vary exp sample
    kwargs.update(**{
        '--eps': 0.05,
        '--sample-variations': np.arange(3),
        '--exp-fractions': fractions,
        '--grid-fractions': fixed_sample_frac})
    execute_experiment(args, kwargs, original_argv)


    args[1] = 'fairlearn'
    kwargs = {'--eps': 0.05}
    execute_experiment(args, kwargs, original_argv)

    args[1] = 'unmitigated'
    execute_experiment(args, kwargs, original_argv)


# if __name__ == '__main__':
#     test_get_synthetic_data()
