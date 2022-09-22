from run_hybrids import run_hybrids
from utils import load_data


def test_run_hybrids():
    dataset_str = f"adult"
    X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all = load_data()
    sample_variations = [int(x) for x in [1,2]]
    sample_fractions = [float(x) for x in [.2,.5]]
    grid_fraction = 0.1
    eps = 0.05
    method_str = f"hybrids_e{eps}_g{grid_fraction}"

    print(f"Running Hybrids with sample variations {sample_variations} and fractions {sample_fractions}, "
          f"and grid-fraction={grid_fraction}...\n")
    results = run_hybrids(X_train_all, y_train_all, A_train_all, X_test_all, y_test_all, A_test_all, eps,
                          sample_indices=sample_variations, fractions=sample_fractions, grid_fraction=grid_fraction)

    assert True


