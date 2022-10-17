import sys

import numpy as np

import run as run

original_argv = sys.argv.copy()


def to_arg(list_p, dict_p, original_argv=original_argv):
    res_string = original_argv + list_p
    for key, value in dict_p.items():
        flag = False
        try:
            flag = len(value) > 0
        except:
            pass
        if flag is True:
            value = ','.join([str(x) for x in value])
        res_string += [f'{key}={value}']
    return res_string


def execute_experiment(list_p, dict_p, original_argv=original_argv):
    sys.argv = to_arg(list_p, dict_p, original_argv)
    run.main()


if __name__ == "__main__":
    dataset_name = 'adult'
    args = [dataset_name, 'model_name']
    kwargs = {}

    fractions = [0.001, 0.004, 0.016, 0.063, 0.251, 1] # np.geomspace(0.001,1,7) np.linspace(0.001,1,7)
    fixed_sample_frac = 0.1
    args[1] = 'hybrids'
    # make grid-sample vary
    kwargs.update(**{
        '--eps': 0.05,
        '--sample-variations': np.arange(10),
        '--exp-fractions': fixed_sample_frac,
        '--grid-fractions': fractions})
    execute_experiment(args, kwargs, original_argv)
    # vary exp sample
    kwargs.update(**{
        '--eps': 0.05,
        '--sample-variations': np.arange(10),
        '--exp-fractions': fractions,
        '--grid-fractions': fixed_sample_frac})
    execute_experiment(args, kwargs, original_argv)
    # TODO try different sensitive attr


    args[1] = 'fairlearn'
    kwargs = {'--eps': 0.05}
    execute_experiment(args, kwargs, original_argv)

    args[1] = 'unmitigated'
    execute_experiment(args, kwargs, original_argv)



    # dataset_name = 'synth'
    # base_kwargs = {'--eps': 0.05, '-f': 3, '-t': 0.5, '-t0': 0.3, '-t1': 0.6, '-v': 1,
    #                '--test-ratio': 0.3
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
    #             '--sample-variations': np.arange(10),
    #             '--exp-fractions': 0.016,
    #             '--grid-fraction': g})
    #         execute_experiment(args, kwargs, original_argv)
    #
    #     args[1] = 'unmitigated'
    #     kwargs = base_kwargs.copy()
    #     execute_experiment(args, kwargs, original_argv)
