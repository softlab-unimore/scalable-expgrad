import sys
import numpy as np
from run import execute_experiment

original_argv = sys.argv.copy()

if __name__ == "__main__":
    original_argv = sys.argv.copy()
    for dataset_name in ['ACSPublicCoverage',
                          # 'adult',
                          ]:
        args = [dataset_name, 'model_name']
        kwargs = {}
        sample_seeds = np.arange(3)
        fixed_sample_frac = 1 / 2 ** 3
        eps_list = [10**-x for x in [8,5]] + np.linspace(0.001, .12, 3).tolist() # np.geomspace(0.001, .12, 5)
        # np.geomspace(0.001, .12, 5)

        args[1] = 'fairlearn'
        kwargs = {'--eps': eps_list}
        execute_experiment(args, kwargs, original_argv)

        args[1] = 'hybrids'
        kwargs.update(**{
            '--eps': eps_list,
            '--sample_seeds': sample_seeds,
            '--exp_fractions': fixed_sample_frac,
            '--grid_fractions': fixed_sample_frac})
        execute_experiment(args, kwargs, original_argv)





