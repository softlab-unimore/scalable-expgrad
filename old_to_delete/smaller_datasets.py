import sys
from run import execute_experiment
import utils_experiment_parameters

if __name__ == "__main__":
    original_argv = sys.argv.copy()
    for dataset_name in utils_experiment.other_small_dataset_names:
        args = [dataset_name, 'hybrids',
                '--exp_subset',
                '--redo_exp',
                '--redo_tuning'
                ]
        kwargs = {}

        kwargs.update(**{
            '--eps': utils_experiment.eps_values,
            '--sample_seeds': utils_experiment.sample_variation,
            '--train_fractions': utils_experiment.fractions[-2:],
            '--grid_fractions': 1,
            '--base_model_code': 'lr',
            '--random_seed': 2,
            '--train_test_seed': 1,
        })
        execute_experiment(args, kwargs, original_argv)
