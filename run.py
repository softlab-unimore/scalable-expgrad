import sys
import json
import os
import socket
from argparse import ArgumentParser
from datetime import datetime

from baselines import run_unmitigated, run_fairlearn_full
from hybrid_methods import run_hybrids
from synthetic_data import get_data, data_split
from utils import load_data


def main(*args, **kwargs):
    arg_parser = ArgumentParser()

    arg_parser.add_argument("dataset")
    arg_parser.add_argument("method")

    # For Fairlearn and Hybrids
    arg_parser.add_argument("--eps", type=float)

    # For synthetic data
    arg_parser.add_argument("-n", "--num-data-points", type=int)
    arg_parser.add_argument("-f", "--num-features", type=int)
    arg_parser.add_argument("-t", "--type-ratio", type=float)
    arg_parser.add_argument("-t0", "--t0-ratio", type=float)
    arg_parser.add_argument("-t1", "--t1-ratio", type=float)
    arg_parser.add_argument("-v", "--data-random-variation", type=int)
    arg_parser.add_argument("--test-ratio", type=float)

    # For hybrid methods
    arg_parser.add_argument("--sample-variations")
    arg_parser.add_argument("--sample-fractions")
    arg_parser.add_argument("--grid-fraction", type=float)

    args = arg_parser.parse_args()

    ####

    host_name = socket.gethostname()
    if "." in host_name:
        host_name = host_name.split(".")[-1]

    dataset = args.dataset
    method = args.method
    eps = args.eps

    if dataset == "adult":
        dataset_str = f"adult"
        X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all = load_data()

    elif dataset == "synth":
        num_data_pts = args.num_data_points
        num_features = args.num_features
        type_ratio = args.type_ratio
        t0_ratio = args.t0_ratio
        t1_ratio = args.t1_ratio
        random_variation = args.data_random_variation
        test_ratio = args.test_ratio
        dataset_str = f"synth_n{num_data_pts}_f{num_features}_t{type_ratio}_t0{t0_ratio}_t1{t1_ratio}_tr{test_ratio}_v{random_variation}"

        print(f"Generating synth data "
              f"(n={num_data_pts}, f={num_features}, t={type_ratio}, t0={t0_ratio}, t1={t1_ratio}, "
              f"v={random_variation})...")
        All = get_data(
            num_data_pts=num_data_pts,
            num_features=num_features,
            type_ratio=type_ratio,
            t0_ratio=t0_ratio,
            t1_ratio=t1_ratio,
            random_seed=random_variation + 40)

        assert 0 < test_ratio < 1
        print(f"Splitting train/test with test_ratio={test_ratio}")
        X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all = data_split(All, test_ratio)

    else:
        raise ValueError(dataset)

    if method == "hybrids":
        # Subsampling process
        # num_samples = 1  # 10  # 20
        # num_fractions = 6  # 20
        # fractions = np.logspace(-3, 0, num=num_fractions)
        # fractions = [0.004]
        sample_variations = [int(x) for x in args.sample_variations.split(",")]
        sample_fractions = [float(x) for x in args.sample_fractions.split(",")]
        grid_fraction = args.grid_fraction
        assert 0 <= grid_fraction <= 1
        method_str = f"hybrids_e{eps}_g{grid_fraction}"

        print(f"Running Hybrids with sample variations {sample_variations} and fractions {sample_fractions}, "
              f"and grid-fraction={grid_fraction}...\n")
        results = run_hybrids(X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all, eps,
                              sample_indices=sample_variations, fractions=sample_fractions, grid_fraction=grid_fraction)

    elif method == "unmitigated":
        results = run_unmitigated(X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all)
        method_str = f"unmitigated"

    elif method == "fairlearn":
        results = run_fairlearn_full(X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all, eps)
        method_str = f"fairlearn_e{eps}"

    else:
        raise ValueError(method)

    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_file_name = f'results/{host_name}/{dataset_str}/{current_time_str}_{method_str}.json'
    print(f"Storing results in '{results_file_name}'")

    # Store results
    base_dir = os.path.dirname(results_file_name)
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    with open(results_file_name, 'w') as _file:
        json.dump(results, _file, indent=2)


if __name__ == "__main__":
    main()
