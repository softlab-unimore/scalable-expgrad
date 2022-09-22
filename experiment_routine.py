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

    args[1] = 'fairlearn'
    kwargs = {'--eps': 0.05}
    execute_experiment(args, kwargs, original_argv)

    args[1] = 'unmitigated'
    execute_experiment(args, kwargs, original_argv)

    args[1] = 'hybrids'
    kwargs.update(**{
        '--eps': 0.05,
        '--sample-variations': np.arange(10),
        '--sample-fractions': [0.001, 0.004, 0.016, 0.063, 0.251, 1],
        '--grid-fraction': 0.1})  # TODO -g 0.5???
    execute_experiment(args, kwargs, original_argv)


    dataset_name = 'synth'
    base_kwargs = {'--eps': 0.05, '-f': 3, '-t': 0.5, '-t0': 0.3, '-t1': 0.6, '-v': 1,
                   '--test-ratio': 0.3
                   }
    args = [dataset_name, 'model_name']
    for n in [
        10000,
        100000,
        1000000,
        10000000
    ]:
        args[1] = 'fairlearn'
        base_kwargs['-n'] = n
        kwargs = base_kwargs.copy()
        execute_experiment(args, kwargs, original_argv)

        args[1] = 'hybrids'
        for g in [.5, .2, .1]:
            kwargs = base_kwargs.copy()
            kwargs.update(**{
                '--sample-variations': np.arange(10),
                '--sample-fractions': 0.016,
                '--grid-fraction': g})
            execute_experiment(args, kwargs, original_argv)

        args[1] = 'unmitigated'
        kwargs = base_kwargs.copy()
        execute_experiment(args, kwargs, original_argv)
