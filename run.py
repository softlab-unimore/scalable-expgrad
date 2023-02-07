import gc
import itertools
from copy import deepcopy
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import folktables
import sys
import os
import socket
from argparse import ArgumentParser
from datetime import datetime
from warnings import simplefilter
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from fairlearn.reductions import DemographicParity, EqualizedOdds, UtilityParity
from synthetic_data import get_synthetic_data, data_split, get_miro_synthetic_data
from utils_prepare_data import load_data, load_transform_ACS, fix_nan
import inspect
from hybrid_models import Hybrid5, Hybrid1, Hybrid2, Hybrid3, Hybrid4, ExponentiatedGradientPmf, finetune_model, \
    get_base_model
from metrics import default_metrics_dict
from utils_prepare_data import get_data_from_expgrad


def to_arg(list_p, dict_p, original_argv):
    res_string = original_argv + list_p
    for key, value in dict_p.items():
        if isinstance(value, list) or isinstance(value, range):
            value = ','.join([str(x) for x in value])
        res_string += [f'{key}={value}']
    return res_string


def execute_experiment(list_p, dict_p, original_argv):
    sys.argv = to_arg(list_p, dict_p, original_argv)
    exp_run = ExpreimentRun()
    exp_run.run()


params_initials_map = {'d': 'dataset', 'm': 'method', 'e': 'eps', 'ndp': 'num_data_points', 'nf': 'num_features',
                       't': 'theta', 'g': 'groups', 'gp': 'group_prob', 'yp': 'y_prob', 'sp': 'switch_pos',
                       'sn': 'switch_neg', 'sv': 'sample_variations', 'ef': 'exp_fractions', 'gf': 'grid_fractions',
                       'egr': 'exp_grid_ratio', 'es': 'exp_subset', 's': 'states', 'rs': 'random_seed',
                       'rls': 'run_linprog_step', 'rt': 'redo_tuning', 're': 'redo_exp', 'bmc': 'base_model_code',
                       'cc': 'constraint_code'}


def get_constraint(constraint_code='dp', eps=.05):
    code_to_constraint = {'dp': DemographicParity,
                          'eo': EqualizedOdds}
    if constraint_code not in code_to_constraint.keys():
        assert False, f'available constraint_code are: {list(code_to_constraint.keys())}'
    constraint: UtilityParity = code_to_constraint[constraint_code]
    return constraint(difference_bound=eps)


class ExpreimentRun:

    def __init__(self):
        host_name = socket.gethostname()
        if "." in host_name:
            host_name = host_name.split(".")[-1]
        self.host_name = host_name
        self.base_result_dir = f'results/{host_name}/'

    def run(self):
        simplefilter(action='ignore', category=FutureWarning)
        arg_parser = ArgumentParser()

        arg_parser.add_argument("dataset")
        arg_parser.add_argument("method")

        # For Fairlearn and Hybrids
        arg_parser.add_argument("--eps")
        arg_parser.add_argument("--constraint_code", choices=['dp', 'eo'], default='dp')

        # For synthetic data
        arg_parser.add_argument("--num_data_points", type=int)
        arg_parser.add_argument("--num_features", type=int)
        arg_parser.add_argument("--theta", type=float, default=0.5)
        arg_parser.add_argument('--groups')
        arg_parser.add_argument('--group_prob')
        arg_parser.add_argument('--y_prob')
        arg_parser.add_argument('--switch_pos')
        arg_parser.add_argument('--switch_neg')
        # arg_parser.add_argument("--test_ratio", type=float, default=0.3)

        # For hybrid methods
        arg_parser.add_argument("--sample_variations")
        arg_parser.add_argument("--exp_fractions")
        arg_parser.add_argument("--grid_fractions")
        arg_parser.add_argument("--exp_grid_ratio", choices=['sqrt', None], default=None)
        arg_parser.add_argument("--exp_subset", action="store_true")

        # Others
        arg_parser.add_argument("--save", default=True)
        arg_parser.add_argument("-v", "--random_seed", type=int, default=0)
        arg_parser.add_argument("--no_run_linprog_step", default=True, dest='run_linprog_step', action='store_false')
        arg_parser.add_argument("--redo_tuning", action="store_true", default=False)
        arg_parser.add_argument("--redo_exp", action="store_true", default=False)
        arg_parser.add_argument("--states")
        arg_parser.add_argument("--base_model_code", default='lr')

        args = arg_parser.parse_args()
        params_to_initials_map = {"".join([x[0] for x in key.split("_")]): key for key in args.__dict__.keys()}

        self.args = args
        if args.grid_fractions is not None:
            assert args.exp_grid_ratio is None, '--exp_grid_ratio must not be set if using --grid_fractions'
        ### Parse parameters
        states = None
        if args.states is not None:
            states = [x for x in args.states.split(',')]
        method_str = args.method
        for key, value in args.__dict__.items():
            if key in ['save', 'method', 'dataset', 'eps', 'exp_fractions', 'grid_fractions',
                       'sample_variations', 'redo_tuning', 'redo_exp'] or value is None:
                continue
            method_str += f'_{"".join([x[0] for x in key.split("_")])}({value})'

        prm = {}
        for key, t_type in zip(['exp_fractions', 'grid_fractions', 'eps', 'sample_variations'],
                               [float] * 3 + [int]):
            if hasattr(args, key) and getattr(args, key) is not None:
                prm[key] = [t_type(x) for x in getattr(args, key).split(",")]
            else:
                prm[key] = None
        if 'exp_fractions' not in prm.keys() or prm['exp_fractions'] is None:
            prm['exp_fractions'] = [1]

        varying_values = ''
        for key, value in prm.items():
            if value is not None:
                if key in ['exp_fractions', 'grid_fractions', 'eps']:
                    if type(value) is list and len(value) > 1:
                        method_str += f'_{key[:3]}VARY'
                else:
                    method_str += f'_{key[:3]}{value}'


        ### Load dataset
        self.dataset_str = args.dataset

        def skip(dataset_str):
            path = os.path.join(self.base_result_dir, self.dataset_str, f'last_{varying_values}.csv')
            if os.path.exists(path) and args.redo_exp is False:
                print(f'\n\nFile {path} already exist. \nSkipping it.\n\n')
                return True
            print(f'Start data loading {dataset_str}')
            return False

        if self.dataset_str == "adult":
            if skip(dataset_str=self.dataset_str) is True:
                return 2
            X, y, A = load_data()
        elif self.dataset_str in ['ACSIncome', 'ACSPublicCoverage', 'ACSMobility', 'ACSEmployment', 'ACSTravelTime',
                                  'ACSHealthInsurance', 'ACSEmploymentFiltered' 'ACSIncomePovertyRatio']:
            if skip(dataset_str=self.dataset_str) is True:
                return 2
            loader_method = getattr(folktables, self.dataset_str)
            X, y, A = load_transform_ACS(loader_method=loader_method, states=states)
            # X, y, A = fix_nan(X, y, A, mode='mean')
        elif "synth" in self.dataset_str:
            print(self.dataset_str)
            ratios = {}
            ratios['group'] = [x for x in args.groups.split(",")]
            for key in ['group_prob', 'y_prob', 'switch_pos', 'switch_neg']:
                ratios[key] = [float(x) for x in getattr(args, key).split(",")]
            self.dataset_str = f"synth_n{args.num_data_points}_f{args.num_features}_v{args.random_seed}_t{args.theta}"
            if skip(dataset_str=self.dataset_str) is True:
                return 2
            synth_info = pd.DataFrame(ratios)
            for key in ['num_data_points', 'num_features', 'random_seed', 'theta']:
                synth_info[key] = getattr(args, key)

            self.save_result(synth_info, 'synth_info', additional_dir='dataset')
            X, y, A = get_miro_synthetic_data(
                num_data_points=args.num_data_points,
                num_features=args.num_features,
                ratios=ratios,
                random_seed=args.random_seed,
                theta=args.theta)
            for df, name in zip([X, y, A], ['X', 'y', 'A']):
                self.save_result(df, name, additional_dir='dataset')
        else:
            raise ValueError(self.dataset_str)

        self.tuning_step(base_model_code=args.base_model_code, X=X, y=y, fractions=prm['exp_fractions'],
                         random_seed=args.random_seed, redo_tuning=args.redo_tuning)

        results = []
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        # if isinstance(A, pd.DataFrame):
        #     A = A.iloc[:,0]
        to_stratify = pd.Series(A.astype(str)) + '_' + y.astype(int).astype(str)
        for train_test_fold, (train_index, test_index) in tqdm(list(enumerate(skf.split(X, to_stratify)))):
            print('')
            datasets_divided = []
            for turn_index in [train_index, test_index]:
                for turn_df in [X, y, A]:
                    datasets_divided.append(turn_df.iloc[turn_index])
            if "hybrids" in method_str:
                print(
                    f"\nRunning Hybrids with sample variations {prm['sample_variations']} and fractions {prm['exp_fractions']}, "
                    f"and grid-fraction={prm['grid_fractions']}...\n")
                for exp_frac, sampleseed in itertools.product(prm['exp_fractions'], prm['sample_variations']):
                    turn_results = self.run_hybrids(*datasets_divided, eps=prm['eps'], sample_seeds=[sampleseed],
                                                    exp_fractions=[exp_frac], grid_fractions=prm['grid_fractions'],
                                                    train_test_fold=train_test_fold, exp_subset=args.exp_subset,
                                                    exp_grid_ratio=args.exp_grid_ratio,
                                                    base_model_code=args.base_model_code,
                                                    run_linprog_step=args.run_linprog_step,
                                                    random_seed=args.random_seed,
                                                    constraint_code=args.constraint_code)
                    self.save_result(df=pd.DataFrame(turn_results), additional_dir=method_str,
                                     name=f'train_test_fold{train_test_fold}_sampleseed{sampleseed}_exp_frac{exp_frac}')

            elif "unmitigated" in method_str:
                turn_results = self.run_unmitigated(*datasets_divided, train_test_fold=train_test_fold,
                                                    random_seed=args.random_seed, base_model_code=args.base_model_code)
                self.save_result(df=pd.DataFrame(turn_results), additional_dir=method_str,
                                 name=f'train_test_fold{train_test_fold}')
            elif "fairlearn" in method_str:
                turn_results = self.run_fairlearn_full(*datasets_divided, eps=prm['eps'],
                                                       train_test_fold=train_test_fold,
                                                       run_linprog_step=args.run_linprog_step,
                                                       random_seed=args.random_seed,
                                                       base_model_code=args.base_model_code, )
                self.save_result(df=pd.DataFrame(turn_results),
                                 additional_dir=method_str + '_LP_off' if args.run_linprog_step is False else '',
                                 name=f'train_test_fold{train_test_fold}')
            else:
                raise ValueError(method_str)
            results += turn_results
        results_df = pd.DataFrame(results)

        self.save_result(df=results_df, name=method_str)

    def save_result(self, df, name, additional_dir=None):
        if self.args.save is not None and self.args.save == 'False':
            print('Not saving...')
            return 0
        assert self.dataset_str is not None
        directory = os.path.join(self.base_result_dir, self.dataset_str)
        if additional_dir is not None:
            directory = os.path.join(directory, additional_dir)
        os.makedirs(directory, exist_ok=True)
        current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for prefix in [  # f'{current_time_str}',
            f'last']:
            path = os.path.join(directory, f'{prefix}_{name}.csv')
            df.to_csv(path, index=False)
            print(f'Saving results in: {path}')

    def run_hybrids(self, X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all, eps,
                    sample_seeds, exp_fractions, grid_fractions, train_test_fold=None, base_model_code='lr',
                    exp_subset=True, exp_grid_ratio=None, run_linprog_step=True, random_seed=0,
                    constraint_code='dp'):
        assert train_test_fold is not None
        simplefilter(action='ignore', category=FutureWarning)

        # Combine all training data into a single data frame
        train_all = pd.concat([X_train_all, y_train_all, A_train_all], axis=1)
        self.data_dict = {"frac": 0, 'model_name': 'model', 'time': 0, 'phase': 'model name',
                          'random_seed': random_seed, 'base_model_code': base_model_code,
                          'constraint_code': constraint_code,
                          'train_test_fold': train_test_fold, 'iterations': 0,
                          'total_train_size': X_train_all.shape[0], 'total_test_size': X_test_all.shape[0]}
        run_lp_suffix = '_LP_off' if run_linprog_step is False else ''
        eval_dataset_dict = {'train': [X_train_all, y_train_all, A_train_all],
                             'test': [X_test_all, y_test_all, A_test_all]}
        all_params = dict(X=X_train_all, y=y_train_all, sensitive_features=A_train_all)
        if exp_grid_ratio is not None:
            assert grid_fractions is None
            grid_fractions = [exp_grid_ratio]

        results = []
        to_iter = list(itertools.product(eps, exp_fractions, grid_fractions, sample_seeds))
        # Iterations on difference fractions
        for i, (turn_eps, exp_f, grid_f, sample_seed) in tqdm(list(enumerate(to_iter))):
            print('')
            gc.collect()
            self.turn_results = []
            self.data_dict['eps'] = turn_eps
            self.data_dict['exp_frac'] = exp_f
            self.data_dict["sample_seed"] = sample_seed
            if type(grid_f) == str:
                if grid_f == 'sqrt':
                    grid_f = np.sqrt(exp_f)
            self.data_dict["grid_frac"] = grid_f
            # self.data_dict['exp_size'] = int(n_data * exp_f)
            # self.data_dict['grid_size'] = int(n_data * grid_f)
            constraint = get_constraint(constraint_code=constraint_code, eps=turn_eps)

            print(f"Processing: fraction {exp_f: <5}, sample {sample_seed: ^10} GridSearch fraction={grid_f:0<5}")

            base_model = self.load_base_model_best_param(base_model_code=base_model_code, fraction=exp_f,
                                                         random_seed=random_seed)
            self.data_dict['model_name'] = 'unconstrained'
            unconstrained_model = deepcopy(base_model)
            metrics_res, time_unconstrained_dict, time_eval_dict = self.fit_evaluate_model(
                unconstrained_model, dict(X=X_train_all, y=y_train_all), eval_dataset_dict)
            time_unconstrained_dict['phase'] = 'unconstrained'
            self.add_turn_results(metrics_res, [time_eval_dict, time_unconstrained_dict])

            # GridSearch data fraction
            grid_sample = train_all.sample(frac=grid_f, random_state=sample_seed + 60, replace=False)
            grid_sample = grid_sample.reset_index(drop=True)
            grid_params = dict(X=grid_sample.iloc[:, :-2],
                               y=grid_sample.iloc[:, -2],
                               sensitive_features=grid_sample.iloc[:, -1])

            if exp_subset:
                exp_sample = grid_sample.sample(frac=exp_f / grid_f, random_state=sample_seed + 20, replace=False)
            else:
                exp_sample = train_all.sample(frac=exp_f, random_state=sample_seed + 20, replace=False)
            exp_sample = exp_sample.reset_index(drop=True)
            exp_params = dict(X=exp_sample.iloc[:, :-2],
                              y=exp_sample.iloc[:, -2],
                              sensitive_features=exp_sample.iloc[:, -1])
            # Unconstrained on sample
            self.data_dict['model_name'] = 'unconstrained_frac'
            unconstrained_model_frac = deepcopy(base_model)
            metrics_res, time_uncons_frac_dict, time_eval_dict = self.fit_evaluate_model(
                unconstrained_model_frac, dict(X=exp_params['X'], y=exp_params['y']), eval_dataset_dict)
            time_unconstrained_dict['phase'] = 'unconstrained'
            self.add_turn_results(metrics_res, [time_eval_dict, time_uncons_frac_dict])

            # Expgrad on sample
            self.data_dict['model_name'] = f'expgrad_fracs{run_lp_suffix}'
            expgrad_frac = ExponentiatedGradientPmf(estimator=deepcopy(base_model), run_linprog_step=run_linprog_step,
                                                    constraints=deepcopy(constraint), eps=turn_eps, nu=1e-6)
            metrics_res, time_exp_dict, time_eval_dict = self.fit_evaluate_model(expgrad_frac, exp_params,
                                                                                 eval_dataset_dict)
            exp_data_dict = get_data_from_expgrad(expgrad_frac)
            self.data_dict.update(**exp_data_dict)
            time_exp_dict['phase'] = 'expgrad_fracs'
            print(f"ExponentiatedGradient on subset done in {time_exp_dict['time']}")
            self.add_turn_results(metrics_res, [time_eval_dict, time_exp_dict])

            #################################################################################################
            # 7
            #################################################################################################
            self.data_dict['model_name'] = f'hybrid_7{run_lp_suffix}'
            print(f"Running {self.data_dict['model_name']}")
            subsample_size = int(X_train_all.shape[0] * exp_f)
            expgrad_subsample = ExponentiatedGradientPmf(estimator=deepcopy(base_model),
                                                         run_linprog_step=run_linprog_step,
                                                         constraints=deepcopy(constraint), eps=turn_eps, nu=1e-6,
                                                         subsample=subsample_size, random_state=random_seed)
            metrics_res, time_exp_dict, time_eval_dict = self.fit_evaluate_model(expgrad_subsample, all_params,
                                                                                 eval_dataset_dict)
            time_exp_dict['phase'] = 'expgrad_fracs'
            exp_data_dict = get_data_from_expgrad(expgrad_subsample)
            self.data_dict.update(**exp_data_dict)
            self.add_turn_results(metrics_res, [time_eval_dict, time_exp_dict])

            for turn_expgrad, prefix in [(expgrad_frac, ''), (expgrad_subsample, 'sub_')]:
                #################################################################################################
                # Hybrid 5: Run LP with full dataset on predictors trained on partial dataset only
                # Get rid
                #################################################################################################
                self.data_dict['model_name'] = f'{prefix}hybrid_5{run_lp_suffix}'
                print(f"Running {self.data_dict['model_name']}")
                model1 = Hybrid5(turn_expgrad, eps=turn_eps, constraint=deepcopy(constraint))
                metrics_res, time_lp_dict, time_eval_dict = self.fit_evaluate_model(model1, all_params,
                                                                                    eval_dataset_dict)
                time_lp_dict['phase'] = 'lin_prog'
                self.add_turn_results(metrics_res, [time_eval_dict, time_exp_dict, time_lp_dict])

                #################################################################################################
                # H5 + unconstrained
                #################################################################################################
                self.data_dict['model_name'] = f'{prefix}hybrid_5_U{run_lp_suffix}'
                print(f"Running {self.data_dict['model_name']}")
                model1.unconstrained_model = unconstrained_model
                metrics_res, time_lp_dict, time_eval_dict = self.fit_evaluate_model(model1, all_params,
                                                                                    eval_dataset_dict)
                time_lp_dict['phase'] = 'lin_prog'
                self.add_turn_results(metrics_res,
                                      [time_eval_dict, time_exp_dict, time_unconstrained_dict, time_lp_dict,
                                       ])

                #################################################################################################
                # Hybrid 1: Just Grid Search -> expgrad partial + grid search
                #################################################################################################
                self.data_dict['model_name'] = f'{prefix}hybrid_1{run_lp_suffix}'
                print(f"Running {self.data_dict['model_name']}")
                grid_subsample_size = int(X_train_all.shape[0] * grid_f)
                model = Hybrid1(expgrad=turn_expgrad, eps=turn_eps, constraint=deepcopy(constraint),
                                base_model=deepcopy(base_model), subsample=grid_subsample_size)
                metrics_res, time_grid_dict, time_eval_dict = self.fit_evaluate_model(model, grid_params,
                                                                                      eval_dataset_dict)
                time_grid_dict['phase'] = 'grid_frac'
                time_grid_dict['grid_oracle_times'] = model.grid_search_frac.oracle_execution_times_
                self.add_turn_results(metrics_res, [time_eval_dict, time_exp_dict, time_grid_dict])
                grid_search_frac = model.grid_search_frac

                #################################################################################################
                # Hybrid 2: pmf_predict with exp grid weights in grid search
                # Keep this, remove Hybrid 1.
                #################################################################################################
                self.data_dict['model_name'] = f'{prefix}hybrid_2{run_lp_suffix}'
                print(f"Running {self.data_dict['model_name']}")
                model = Hybrid2(expgrad=turn_expgrad, grid_search_frac=grid_search_frac, eps=turn_eps,
                                constraint=deepcopy(constraint))
                metrics_res, _, time_eval_dict = self.fit_evaluate_model(model, grid_params, eval_dataset_dict)
                self.add_turn_results(metrics_res, [time_eval_dict, time_exp_dict, time_grid_dict])

                #################################################################################################
                # Hybrid 3: re-weight using LP
                #################################################################################################
                self.data_dict['model_name'] = f'{prefix}hybrid_3{run_lp_suffix}'
                print(f"Running {self.data_dict['model_name']}")
                model = Hybrid3(grid_search_frac=grid_search_frac, eps=turn_eps, constraint=deepcopy(constraint))
                metrics_res, time_lp3_dict, time_eval_dict = self.fit_evaluate_model(model, all_params,
                                                                                     eval_dataset_dict)
                time_lp3_dict['phase'] = 'lin_prog'
                self.add_turn_results(metrics_res, [time_eval_dict, time_exp_dict, time_grid_dict, time_lp3_dict])

                #################################################################################################
                # Hybrid 3 +U: re-weight using LP + unconstrained
                #################################################################################################
                self.data_dict['model_name'] = f'{prefix}hybrid_3_U{run_lp_suffix}'
                print(f"Running {self.data_dict['model_name']}")
                model = Hybrid3(grid_search_frac=grid_search_frac, eps=turn_eps, constraint=deepcopy(constraint),
                                unconstrained_model=unconstrained_model)
                metrics_res, time_lp3_dict, time_eval_dict = self.fit_evaluate_model(model, all_params,
                                                                                     eval_dataset_dict)
                time_lp3_dict['phase'] = 'lin_prog'
                self.add_turn_results(metrics_res, [time_eval_dict, time_exp_dict, time_grid_dict, time_lp3_dict,
                                                    time_unconstrained_dict])

                #################################################################################################
                # Hybrid 4: re-weight only the non-zero weight predictors using LP
                #################################################################################################
                self.data_dict['model_name'] = f'{prefix}hybrid_4{run_lp_suffix}'
                print(f"Running {self.data_dict['model_name']}")
                model = Hybrid4(expgrad=turn_expgrad, grid_search_frac=grid_search_frac, eps=turn_eps,
                                constraint=deepcopy(constraint))
                metrics_res, time_lp4_dict, time_eval_dict = self.fit_evaluate_model(model, all_params,
                                                                                     eval_dataset_dict)
                time_lp4_dict['phase'] = 'lin_prog'
                self.add_turn_results(metrics_res, [time_eval_dict, time_exp_dict, time_grid_dict, time_lp4_dict])

                #################################################################################################
                # Hybrid 6: exp + grid predictors
                #################################################################################################
                self.data_dict['model_name'] = f'{prefix}hybrid_6{run_lp_suffix}'
                print(f"Running {self.data_dict['model_name']}")
                model = Hybrid3(add_exp_predictors=True, grid_search_frac=grid_search_frac, expgrad=turn_expgrad,
                                eps=turn_eps, constraint=deepcopy(constraint))
                metrics_res, time_lp_dict, time_eval_dict = self.fit_evaluate_model(model, all_params,
                                                                                    eval_dataset_dict)
                time_lp_dict['phase'] = 'lin_prog'
                self.add_turn_results(metrics_res, [time_eval_dict, time_exp_dict, time_grid_dict, time_lp_dict])

                #################################################################################################
                # Hybrid 6 + U: exp + grid predictors + unconstrained
                #################################################################################################
                self.data_dict['model_name'] = f'{prefix}hybrid_6_U{run_lp_suffix}'
                print(f"Running {self.data_dict['model_name']}")
                model = Hybrid3(add_exp_predictors=True, grid_search_frac=grid_search_frac, expgrad=turn_expgrad,
                                eps=turn_eps, constraint=deepcopy(constraint), unconstrained_model=unconstrained_model)
                metrics_res, time_lp_dict, time_eval_dict = self.fit_evaluate_model(model, all_params,
                                                                                    eval_dataset_dict)
                time_lp_dict['phase'] = 'lin_prog'
                self.add_turn_results(metrics_res, [time_eval_dict, time_exp_dict, time_grid_dict, time_lp_dict,
                                                    time_unconstrained_dict])

            #################################################################################################
            # End models
            #################################################################################################
            results += self.turn_results
            print("Fraction processing complete.\n")

        return results

    def run_unmitigated(self, X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all,
                        base_model_code, train_test_fold, random_seed=0):
        assert train_test_fold is not None
        assert base_model_code is not None
        self.turn_results = []
        self.data_dict = {'model_name': 'unmitigated', 'time': 0, 'phase': 'model name', 'random_seed': random_seed,
                          'train_test_fold': train_test_fold, 'base_model_code': base_model_code}
        eval_dataset_dict = {'train': [X_train_all, y_train_all, A_train_all],
                             'test': [X_test_all, y_test_all, A_test_all]}
        base_model = self.load_base_model_best_param(base_model_code=base_model_code, random_seed=random_seed)
        train_data = dict(X=X_train_all, y=y_train_all)
        metrics_res, time_exp_dict, time_eval_dict = self.fit_evaluate_model(base_model, train_data, eval_dataset_dict)
        time_exp_dict['phase'] = 'unconstrained'
        self.add_turn_results(metrics_res, [time_eval_dict, time_exp_dict])
        return self.turn_results

    # Fairlearn on full dataset
    def run_fairlearn_full(self, X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all, eps,
                           base_model_code, train_test_fold,
                           random_seed=0, run_linprog_step=True):
        assert train_test_fold is not None
        assert base_model_code is not None
        self.turn_results = []
        self.data_dict = {'eps': 0, 'model_name': 'fairlearn_full', 'time': 0, 'phase': 'fairlearn_full',
                          'random_seed': random_seed, 'train_test_fold': train_test_fold,
                          'base_model_code': base_model_code}
        eval_dataset_dict = {'train': [X_train_all, y_train_all, A_train_all],
                             'test': [X_test_all, y_test_all, A_test_all]}
        num_samples = 1
        to_iter = list(itertools.product(eps, [num_samples]))

        for i, (turn_eps, n) in tqdm(list(enumerate(to_iter))):
            print('')
            self.data_dict['eps'] = turn_eps
            base_model = self.load_base_model_best_param(base_model_code=base_model_code, random_seed=random_seed)
            expgrad_X_logistic = ExponentiatedGradientPmf(base_model,
                                                          constraints=DemographicParity(difference_bound=turn_eps),
                                                          eps=turn_eps, nu=1e-6,
                                                          run_linprog_step=run_linprog_step)
            print("Fitting Exponentiated Gradient on full dataset...")
            train_data = dict(X=X_train_all, y=y_train_all, sensitive_features=A_train_all)
            metrics_res, time_exp_dict, time_eval_dict = self.fit_evaluate_model(expgrad_X_logistic,
                                                                                 train_data,
                                                                                 eval_dataset_dict)
            time_exp_dict['phase'] = 'expgrad'
            exp_data_dict = get_data_from_expgrad(expgrad_X_logistic)
            self.data_dict.update(**exp_data_dict)
            self.add_turn_results(metrics_res, [time_eval_dict, time_exp_dict])

            print(f'Exponentiated Gradient on full dataset : ')
            for key, value in self.data_dict.items():
                if key not in ['model_name', 'phase']:
                    print(f'{key} : {value}')
        return self.turn_results

    def add_turn_results(self, metrics_res, time_dict_list):
        base_dict = self.data_dict
        base_dict.update(**metrics_res)
        for t_time_dict in time_dict_list:
            turn_dict = deepcopy(base_dict)
            turn_dict.update(**t_time_dict)
            self.turn_results.append(turn_dict)

    @staticmethod
    def get_metrics(dataset_dict: dict, predict_method, metrics_methods=default_metrics_dict, return_times=False):
        metrics_res = {}
        time_list = []
        time_dict = {}
        for phase, dataset_list in dataset_dict.items():
            X, Y, S = dataset_list
            a = datetime.now()
            y_pred = predict_method(X)
            b = datetime.now()
            time_dict.update(metric=f'{phase}_prediction', time=(b - a).total_seconds())
            time_list.append(deepcopy(time_dict))
            for name, eval_method in metrics_methods.items():
                turn_name = f'{phase}_{name}'
                params = inspect.signature(eval_method).parameters.keys()
                a = datetime.now()
                if 'predict_method' in params:
                    turn_res = eval_method(*dataset_list, predict_method=predict_method)
                elif 'y_pred' in params:
                    turn_res = eval_method(*dataset_list, y_pred=y_pred)
                else:
                    raise AssertionError('Metric method has not y_pred or predict_method. This is not allowed!')
                b = datetime.now()
                time_dict.update(metric=turn_name, time=(b - a).total_seconds())
                time_list.append(deepcopy(time_dict))
                metrics_res[turn_name] = turn_res
        if return_times:
            return metrics_res, time_list
        return metrics_res

    @staticmethod
    def fit_evaluate_model(model, train_dataset, evaluate_dataset_dict):
        a = datetime.now()
        model.fit(**train_dataset)
        b = datetime.now()
        time_fit_dict = {'time': (b - a).total_seconds(), 'phase': 'train'}

        # Training violation & error of hybrid 4
        a = datetime.now()
        metrics_res, metrics_time = ExpreimentRun.get_metrics(evaluate_dataset_dict, model.predict, return_times=True)
        b = datetime.now()
        time_eval_dict = {'time': (b - a).total_seconds(), 'phase': 'evaluation', 'metrics_time': metrics_time}
        return metrics_res, time_fit_dict, time_eval_dict

    def load_base_model_best_param(self, base_model_code=None, random_seed=None, fraction=1):
        if random_seed is None:
            random_seed = self.args.random_seed
        best_params = self.load_best_param(base_model_code, fraction=fraction,
                                           random_seed=random_seed)
        model = get_base_model(base_model_code=base_model_code, random_seed=random_seed)
        model.set_params(**best_params)
        return model

    def tuning_step(self, base_model_code, X, y, fractions, random_seed=0, redo_tuning=False):
        print(f'Starting finetuning of {base_model_code}')
        for turn_frac in (pbar := tqdm(fractions)):
            pbar.set_description(f'fraction: {turn_frac: <5}')
            directory = os.path.join(self.base_result_dir, self.dataset_str, 'tuned_models')
            os.makedirs(directory, exist_ok=True)
            name = f'grid_search_{base_model_code}_rnd{random_seed}_frac{turn_frac}'
            path = os.path.join(directory, name + '.pkl')
            if redo_tuning or not os.path.exists(path):
                size = X.shape[0]
                sample_mask = np.arange(size)
                if turn_frac != 1:
                    sample_mask, _ = train_test_split(sample_mask, train_size=turn_frac, stratify=y,
                                                      random_state=random_seed, shuffle=True)
                a = datetime.now()
                clf = finetune_model(base_model_code, X.iloc[sample_mask], y[sample_mask], random_seed=random_seed)
                b = datetime.now()
                joblib.dump(clf, path, compress=1)

                finetuning_time_df = pd.DataFrame([{'phase': 'grid_searh_finetuning', 'time': (b - a).total_seconds()}])
                self.save_result(finetuning_time_df, 'time_' + name, additional_dir='tuned_models')

    def load_best_param(self, base_model_code, fraction, random_seed=0):
        directory = os.path.join(self.base_result_dir, self.dataset_str, 'tuned_models')
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f'grid_search_{base_model_code}_rnd{random_seed}_frac{fraction}.pkl')
        grid_clf = joblib.load(path)
        return grid_clf.best_params_


if __name__ == "__main__":
    exp_run = ExpreimentRun()
    exp_run.run()
