import sys

import numpy as np

from run import execute_experiment

if __name__ == "__main__":
    original_argv = sys.argv.copy()
    fractions = [0.001, 0.004, 0.016, 0.063, 0.251, 1
                 ]  # np.geomspace(0.001,1,7) np.linspace(0.001,1,7)
    # np.geomspace(0.001,1,6)
    fixed_sample_frac = 0.1
    sample_variation = range(1)
    eps = 0.05



    for dataset_name in ['ACSPublicCoverage', 'ACSIncome', 'ACSMobility', 'ACSEmployment', 'ACSTravelTime',
                         'ACSHealthInsurance', 'ACSEmploymentFiltered' 'ACSIncomePovertyRatio']:
        args = [dataset_name, 'model_name']
        kwargs = {}

        args[1] = 'hybrids'

        # vary exp sample
        kwargs.update(**{
            '--eps': eps,
            '--sample_variations': sample_variation,
            '--exp_fractions': fractions,
            '--grid_fractions': fixed_sample_frac})
        execute_experiment(args, kwargs, original_argv)
        # make grid-sample vary
        kwargs.update(**{
            '--eps': eps,
            '--sample_variations': sample_variation,
            '--exp_fractions': 0.001,
            '--grid_fractions': fractions})
        execute_experiment(args, kwargs, original_argv)
        # TODO try different sensitive attr

        args[1] = 'fairlearn'
        kwargs = {'--eps': eps}
        execute_experiment(args, kwargs, original_argv)

        args[1] = 'unmitigated'
        execute_experiment(args, kwargs, original_argv)

    # Adult dataset
    dataset_name = 'adult'
    args = [dataset_name, 'model_name']
    kwargs = {}

    args[1] = 'hybrids'
    # make grid-sample vary
    kwargs.update(**{
        '--eps': eps,
        '--sample_variations': sample_variation,
        '--exp_fractions': fixed_sample_frac,
        '--grid_fractions': fractions})
    execute_experiment(args, kwargs, original_argv)
    # vary exp sample
    kwargs.update(**{
        '--eps': eps,
        '--sample_variations': sample_variation,
        '--exp_fractions': fractions,
        '--grid_fractions': fixed_sample_frac})
    execute_experiment(args, kwargs, original_argv)
    # TODO try different sensitive attr

    args[1] = 'fairlearn'
    kwargs = {'--eps': eps}
    execute_experiment(args, kwargs, original_argv)

    args[1] = 'unmitigated'
    execute_experiment(args, kwargs, original_argv)

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
    #             '--sample_variations': sample_variation,
    #             '--exp_fractions': 0.016,
    #             '--grid-fraction': g})
    #         execute_experiment(args, kwargs, original_argv)
    #
    #     args[1] = 'unmitigated'
    #     kwargs = base_kwargs.copy()
    #     execute_experiment(args, kwargs, original_argv)
