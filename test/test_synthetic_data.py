from synthetic_data import *


def test_get_synthetic_data():
    num_data_pts, num_features, type_ratio, t0_ratio, t1_ratio, random_seed = 1000, 4, .5, .3, .6, 42
    synth_new = get_synthetic_data(num_data_pts, num_features, type_ratio, t0_ratio, t1_ratio, random_seed)
    synth_old = get_synthetic_data_old(num_data_pts, num_features, type_ratio, t0_ratio, t1_ratio, random_seed)
    # assert (synth_old.values == synth_new.values).all()
    assert synth_new.shape == (1000, num_features + 1)


if __name__ == '__main__':
    test_get_synthetic_data()
