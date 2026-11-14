"""Information coefficient: how each feature lines up with what the index did next.

For every feature we take the rank correlation between its value on day t and the forward
return of the index, separately in an early and a late period. A feature that only "works"
in one of the two periods is most likely noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.panel import PanelBundle
from ..model.dataset import make_labels
from ..trading.metrics import spearman


def feature_ic(bundle: PanelBundle, cfg: dict, split_year: int) -> pd.DataFrame:
    horizons = [int(h) for h in cfg["model"]["horizons"]]
    labels = make_labels(bundle.price, horizons, int(cfg["model"]["execution_lag"]))
    panel = bundle.panel
    early = panel.index.year <= split_year
    rows = []
    for j, h in enumerate(horizons):
        if h == 1:
            continue
        forward = pd.Series(labels.ret[:, j], index=panel.index)
        for column in panel.columns:
            feature = panel[column]
            record = {"feature": column, "group": bundle.groups[column], "horizon": h}
            for name, mask in (("early", early), ("late", ~early)):
                x, y = feature[mask], forward[mask]
                ok = x.notna() & y.notna()
                n = int(ok.sum())
                ic = spearman(x[ok], y[ok]) if n > 100 else np.nan
                record[f"ic_{name}"] = ic
                # overlapping h-day labels: roughly n / h independent observations
                record[f"t_{name}"] = ic * np.sqrt(max(n / h - 2, 1)) if np.isfinite(ic) else np.nan
            record["same_sign"] = bool(np.sign(record["ic_early"]) == np.sign(record["ic_late"]))
            record["stable_score"] = (min(abs(record["ic_early"]), abs(record["ic_late"]))
                                      if record["same_sign"] else 0.0)
            rows.append(record)
    table = pd.DataFrame(rows)
    return table.sort_values(["horizon", "stable_score"], ascending=[True, False]).reset_index(drop=True)
