import numpy as np

from run import execute_experiment
import sys

original_argv = sys.argv.copy()

if __name__ == "__main__": # TODO ⚠ ✍ ⚍⚍⚍⚍⚍⚍⚍⚍⚍⚍⚍⚍⚍⚍⚍⚍⚍ ╠╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╬╣ ◝◟⌒⌒⌒⌒⌒◞◜
    states = ['MA']  # CA AL, MA, MI
    # dataset_name = 'ACSHealthInsurance'
    dataset_name = 'adult'
    args = [dataset_name, 'model_name',
            # '--redo_tuning',
            '--redo_exp',
            '--exp_subset'
            ]
    kwargs = {'--save': 'True', '--base_model_code': 'lr',
              # '--states': states,
              }

    fractions = [0.01, 0.001]  # np.geomspace(0.001,1,7) np.linspace(0.001,1,7)
    sample_variation = [1]  # np.arange(2)
    fixed_sample_frac = 0.001
    args[1] = 'hybrids'
    kwargs.update(**{
        '--eps': [0.001, .01],
        '--sample_variations': sample_variation,
        '--exp_fractions': fractions,
        '--exp_grid_ratio': 'sqrt',
        # '--grid_fractions': fixed_sample_frac
    })
    execute_experiment(args, kwargs, original_argv)


    # make grid-sample vary

    # args[1] = 'fairlearn'
    # kwargs = {'--eps': 0.05, '--states': states
    #           }
    # execute_experiment(args, kwargs, original_argv)

    # args[1] = 'unmitigated'
    # execute_experiment(args, kwargs, original_argv)



    # dataset_name = 'synth'
    # args = [dataset_name, 'model_name', '--exp_subset', #'--run_linprog_step'
    #         ]
    # kwargs = {'--save': 'True',
    #           '--base_model': 'lgbm',
    #           '--num_data_points': 100000,
    #           '--num_features': 3,
    #           '--theta': 0.55,
    #           '--groups': range(3),
    #           '--group_prob': [.3, .3, .4],
    #           '--y_prob': [.7, .6, .65],
    #           '--switch_pos': [.1, .2, .15],
    #           '--switch_neg': [.2, .15, .2],
    #           '--random_seed': 0,
    #           }
    #
    # fractions = [0.01, 0.1]  # np.geomspace(0.001,1,7) np.linspace(0.001,1,7)
    # sample_variation = range(2)
    # fixed_sample_frac = 0.005
    #
    # args[1] = 'hybrids'
    # kwargs.update(**{
    #     '--eps': [0.05, .01],
    #     '--sample_variations': sample_variation,
    #     '--exp_fractions': fractions,
    #     # '--grid_fractions': fixed_sample_frac
    #     '--exp_grid_ratio':'sqrt'
    # })
    # # kwargs.pop('--grid_fractions')
    # execute_experiment(args, kwargs, original_argv)
    #
    # args[1] = 'fairlearn'
    # kwargs.update(**{'--eps': 0.05, })
    # execute_experiment(args, kwargs, original_argv)
    #
    # args[1] = 'unmitigated'
    # execute_experiment(args, kwargs, original_argv)


# if __name__ == '__main__':
#     test_get_synthetic_data()
