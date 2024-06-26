import sys
from run import execute_experiment
import utils_experiment_parameters

if __name__ == "__main__":
    original_argv = sys.argv.copy()

    for dataset_name in utils_experiment.dataset_names:
        args = [dataset_name, 'model_name',
                '--exp_subset','--redo_exp',
                ]
        kwargs = {}

        args[1] = 'hybrids'

        # make grid-sample vary
        kwargs.update(**{
            '--eps': utils_experiment.eps,
            '--sample_seeds': utils_experiment.sample_variation,
            '--train_fractions': 0.001,
            '--grid_fractions': utils_experiment.fractions})
        kwargs.pop('--exp_grid_ratio')
        execute_experiment(args, kwargs, original_argv)
        # TODO try different sensitive attr

    # dataset_name = 'synth'
    # base_kwargs = {'--eps': eps, '-f': 3, '-t': 0.5, '-t0': 0.3, '-t1': 0.6, '-v': 1,
    #                '--test_ratio': 0.3
    #                }
    # args = [dataset_name, 'model_name']
    # for n in [
    #     10000,
    #     100000,
    #     1000000,
    #     10000000
    # ]:
    #     args[1] = 'fairlearn'
    #     base_kwargs['-n'] = n
    #     kwargs = base_kwargs.copy()
    #     execute_experiment(args, kwargs, original_argv)
    #
    #     args[1] = 'hybrids'
    #     for g in [.5, .2, .1]:
    #         kwargs = base_kwargs.copy()
    #         kwargs.update(**{
    #             '--sample_seeds': sample_variation,
    #             '--train_fractions': 0.016,
    #             '--grid-fraction': g})
    #         execute_experiment(args, kwargs, original_argv)
    #
    #     args[1] = 'unmitigated'
    #     kwargs = base_kwargs.copy()
    #     execute_experiment(args, kwargs, original_argv)
