"""The model pipeline end to end on a tiny synthetic problem."""

import numpy as np
import torch

from marketlab.model import train
from marketlab.model.cnn import MarketCNN
from marketlab.model.dataset import Scaler, make_windows


def test_windows_end_at_their_row():
    matrix = np.arange(40, dtype=np.float32).reshape(10, 4)   # 10 days, 4 channels
    x = make_windows(matrix, np.array([4, 9]), window=3)
    assert tuple(x.shape) == (2, 4, 3)
    assert x[0, 0].tolist() == [8.0, 12.0, 16.0]              # channel 0 on days 2, 3, 4
    assert x[1, 3].tolist() == [31.0, 35.0, 39.0]


def test_scaler_uses_training_rows_and_neutralises_missing_values():
    train_part = np.array([[1.0, 10.0], [3.0, np.nan], [5.0, 30.0]])
    scaler = Scaler.fit(train_part, clip=2.0)
    out = scaler.transform(np.array([[3.0, np.nan], [1000.0, 20.0]]))
    assert out[0].tolist() == [0.0, 0.0]            # mean -> 0, missing -> 0
    assert out[1, 0] == 2.0 and out[1, 1] == 0.0    # outliers are clipped


def test_cnn_cannot_see_beyond_the_last_day():
    """Causal convolutions: the output for a window must not change if we append future days."""
    torch.manual_seed(0)
    model = MarketCNN({"a": [0, 1], "b": [2]}, n_horizons=2, hidden=4).eval()
    x = torch.randn(3, 3, 30)
    with torch.no_grad():
        direction, _ = model(x)
        branch = model.branches["a"]
        hidden_short = branch.blocks(branch.mix(x[:, :2, :20]))
        hidden_long = branch.blocks(branch.mix(x[:, :2, :]))
    assert direction.shape == (3, 2)
    assert torch.allclose(hidden_short, hidden_long[:, :, :20], atol=1e-6)


def test_walk_forward_runs_and_keeps_test_years_out_of_training(bundle, cfg):
    pred = train.walk_forward(bundle, cfg, kind="cnn", years=[2010, 2011], workers=1)
    assert set(pred["fold"]) == {2010, 2011}
    assert pred.index.is_monotonic_increasing and pred.index.year.isin([2010, 2011]).all()
    for column in ("p_5", "mu_5", "ref_mean_5", "ref_std_5", "base_p_5", "p_5_wo_market"):
        assert column in pred.columns
    assert pred["p_5"].between(0, 1).all() and pred["ref_std_5"].notna().all()
    # each fold was trained only on rows whose labels ended before its first test day
    first_2010 = bundle.panel.index.get_loc(pred.index[pred["fold"] == 2010][0])
    assert pred.loc[pred["fold"] == 2010, "train_rows"].iloc[0] <= first_2010 - 6 - (cfg["model"]["window"] - 1)


def test_training_is_reproducible_whatever_ran_before(bundle, cfg):
    first = train.walk_forward(bundle, cfg, kind="cnn", years=[2011], workers=1)
    torch.randn(1000)  # disturb the global random state, as an earlier fold in the same worker would
    second = train.walk_forward(bundle, cfg, kind="cnn", years=[2011], workers=1)
    np.testing.assert_allclose(first["p_5"].to_numpy(), second["p_5"].to_numpy(), atol=1e-6)
