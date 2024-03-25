import sys
import run
import utils_experiment_parameters

if __name__ == "__main__":
    run.launch_experiment_by_id('acs_h_gs1_1.0')
    original_argv = sys.argv.copy()

    for dataset_name in utils_experiment.dataset_names[:]:
        for base_model_code in ['lr', 'lgbm']:
            args = [dataset_name, 'hybrids',
                    '--exp_subset',
                    '--redo_exp',
                    # '--redo_tuning'
                    ]
            kwargs = {}

            args[1] = 'hybrids'

            # vary exp sample
            kwargs.update(**{
                '--eps': utils_experiment.eps,
                '--sample_seeds': utils_experiment.sample_variation,
                '--train_fractions': utils_experiment.fractions,
                # '--exp_grid_ratio': 'sqrt',
                # '--grid_fractions': fixed_sample_frac,
                '--grid_fractions': 1,
                '--base_model_code': base_model_code,  # 'lr', #'lgbm', #
            })
            execute_experiment(args, kwargs, original_argv)
        # # make grid-sample vary
        # kwargs.update(**{
        #     '--eps': eps,
        #     '--sample_seeds': sample_variation,
        #     '--train_fractions': 0.001,
        #     '--grid_fractions': fractions})
        # kwargs.pop('--exp_grid_ratio')
        # execute_experiment(args, kwargs, original_argv)
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
