import sys
from run import execute_experiment
from utils_experiment_parameters import fractions, sample_variation, eps, dataset_names


if __name__ == "__main__":
    original_argv = sys.argv.copy()

    for dataset_name in dataset_names:
        args = [dataset_name, 'model_name',
                '--exp_subset','--redo_exp',
                '--no_run_linprog_step',
                ]
        kwargs = {}

        args[1] = 'hybrids'
        kwargs.update(**{
            '--eps': eps,
            '--sample_seeds': sample_variation,
            '--exp_fractions': fractions,
            '--exp_grid_ratio': 'sqrt',

            # '--grid_fractions': fixed_sample_frac
            # '--base_model_code': 'gbm',  # 'lgmb
        })
        execute_experiment(args, kwargs, original_argv)
