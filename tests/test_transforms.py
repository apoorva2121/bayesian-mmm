import numpy as np

from mmm.simulate import load_config, simulate
from mmm.transforms import geometric_adstock_np, hill_np


def test_adstock_weights_sum_to_one_and_carry_over():
    x = np.zeros(20); x[5] = 100.0
    y = geometric_adstock_np(x, decay=0.5, max_lag=4)
    assert np.isclose(y.sum(), 100.0)
    assert y[5] > y[6] > y[7] > 0 and y[4] == 0


def test_hill_bounds_and_half_saturation():
    assert hill_np(np.array([0.0]), 10.0, 2.0)[0] == 0.0
    assert np.isclose(hill_np(np.array([10.0]), 10.0, 2.0)[0], 0.5)
    assert hill_np(np.array([1e9]), 10.0, 2.0)[0] < 1.0


def test_simulation_is_reproducible_and_positive():
    cfg = load_config()
    a, ta = simulate(cfg); b, tb = simulate(cfg)
    assert a.equals(b) and ta == tb
    assert (a["sales"] > 0).all()
    assert abs(sum(ta["contribution_share"].values()) - 1) < 1e-9
