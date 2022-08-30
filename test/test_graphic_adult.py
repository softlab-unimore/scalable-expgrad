from graphic_adult import *

def test_load_data_adult():
    base_dir = os.path.join("../","results", "yeeha", "adult")
    unmitigated_results_file_name = f"{base_dir}/0.05_2021-01-25_09-33-57_unmitigated.json"
    fairlearn_results_file_name = f"{base_dir}/0.05_2021-01-25_09-33-57_fairlearn.json"
    # hybrid_results_file_name = f"{base_dir}/0.05_2021-02-23_05-29-19_hybrids.json"
    hybrid_results_file_name = f"{base_dir}/0.05_2021-01-25_09-59-57_hybrid.json"
    with open(unmitigated_results_file_name, 'r') as _file:
        unmitigated_results = json.load(_file)
    with open(fairlearn_results_file_name, 'r') as _file:
        fairlearn_results = json.load(_file)
    with open(hybrid_results_file_name, 'r') as _file:
        hybrid_results = json.load(_file)

    data = load_data_adult(unmitigated_results, fairlearn_results, hybrid_results)
    data_old = load_data_adult_old(unmitigated_results, fairlearn_results, hybrid_results) # No return
    assert True

