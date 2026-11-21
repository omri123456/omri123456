"""The daily answer in plain words: buy, add, reduce, exit or wait, plus market breadth.

The model only speaks about the traded index (NASDAQ-100) as a whole. Market breadth comes from
the day-trader scanner (share of liquid US stocks in each setup) and is recorded every day, so it
builds a history that can later become a model input. Breadth is shown, never traded on.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths

log = logging.getLogger(__name__)

DECISIONS = {
    "buy": "קנה: להיכנס למדד",
    "add": "הגדל חשיפה",
    "reduce": "הקטן חשיפה",
    "exit": "צא מהמדד לגמרי",
    "hold": "החזק / חכה: בלי שינוי",
}
SETUPS = ["oversold", "oversold_bounce", "breakout_20d", "bull_flag", "reclaim_ma20"]


def decide(current: float, target: float, band: float) -> str:
    """Compare the weight held now with the weight the model wants after the next close."""
    if target <= 1e-6 and current > 0.05:
        return "exit"
    if current <= 0.05 and target > band:
        return "buy"
    if target - current > band:
        return "add"
    if current - target > band:
        return "reduce"
    return "hold"


def build_signal(state: dict, pred: pd.DataFrame, cfg: dict, breadth: dict | None) -> dict:
    from .paper import summary  # local import keeps the module light for the report

    info = summary(state)
    pending = info.get("pending") or {}
    horizon = int(cfg["model"]["trade_horizon"])
    last = pred.iloc[-1]
    p = float(last[f"p_{horizon}"])
    ref_mean = float(last.get(f"ref_mean_{horizon}", np.nan))
    ref_std = float(last.get(f"ref_std_{horizon}", np.nan))
    score = (p - ref_mean) / ref_std if ref_std and np.isfinite(ref_std) and ref_std > 0 else np.nan
    target = float(pending.get("target_weight", info["weight"]))
    code = decide(info["weight"], target, float(cfg["trading"]["rebalance_band"]))
    return {
        "date": str(pred.index[-1].date()),
        "instrument": cfg["project"]["target"],
        "decision": code,
        "decision_he": DECISIONS[code],
        "weight_now": round(info["weight"], 4),
        "weight_target": round(target, 4),
        "p_up": round(p, 4),
        "p_usual": round(ref_mean, 4) if np.isfinite(ref_mean) else None,
        "score": round(float(score), 2) if np.isfinite(score) else None,
        "horizon_days": horizon,
        "executes_at": "סגירת יום המסחר הבא",
        "account": {k: info[k] for k in ("equity", "return_pct", "index_return_pct", "trades", "last_mark")},
        "breadth": breadth,
    }


def append_row(path: Path, row: dict, key: str = "date") -> None:
    """Add or replace one dated row in a small CSV history."""
    frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(frame) and key in frame.columns:
        frame = frame[frame[key].astype(str) != str(row[key])]
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True).sort_values(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def save_signal(paths: Paths, signal: dict) -> None:
    paths.state.mkdir(parents=True, exist_ok=True)
    (paths.state / "latest_signal.json").write_text(json.dumps(signal, ensure_ascii=False, indent=1), encoding="utf-8")
    flat = {k: v for k, v in signal.items() if k not in ("account", "breadth", "decision_he", "executes_at")}
    flat["equity"] = signal["account"]["equity"]
    append_row(paths.state / "signals.csv", flat)


# ---------------------------------------------------------------------------------------------
# Market breadth from the day-trader scanner (optional: skipped when the skill is not installed)
# ---------------------------------------------------------------------------------------------

def scanner_path(paths: Paths) -> Path:
    return paths.root / ".claude" / "skills" / "trader-scanner" / "scan.py"


def run_scanner(paths: Paths, timeout: int = 900) -> bool:
    """Refresh the stock list weekly and run the pre-market daily scan. Never fatal."""
    scan = scanner_path(paths)
    if not scan.exists():
        return False
    universe = paths.root / "trader" / "universe.txt"
    try:
        if not universe.exists() or time.time() - universe.stat().st_mtime > 7 * 86400:
            subprocess.run([sys.executable, str(scan), "universe"], cwd=paths.root, timeout=300,
                           capture_output=True, check=False)
        done = subprocess.run([sys.executable, str(scan), "daily", "--show", "0"], cwd=paths.root,
                              timeout=timeout, capture_output=True, text=True, check=False)
        log.info("scanner: %s", (done.stdout or done.stderr).strip().splitlines()[-1:] or "")
        return (paths.root / "trader" / "watchlist.json").exists()
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("scanner did not run: %s", exc)
        return False


def read_breadth(paths: Paths) -> dict | None:
    """Summarise trader/watchlist.json written by the daily scan."""
    path = paths.root / "trader" / "watchlist.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    if data.get("mode") != "daily" or not data.get("scanned"):
        return None
    hits = data.get("hits", [])
    scanned = int(data["scanned"])
    counts = {s: sum(1 for h in hits if s in h.get("setups", [])) for s in SETUPS}
    as_of = str(data.get("as_of", ""))[:10]
    if not as_of or not as_of[0].isdigit():
        as_of = str(hits[0].get("close_date", "")) if hits else ""
    top = sorted(hits, key=lambda h: -(h.get("price") or 0) * (h.get("volume") or 0))[:12]
    return {
        "as_of": as_of, "scanned": scanned, "universe": data.get("universe"),
        "counts": counts, "pct": {s: round(100.0 * c / scanned, 1) for s, c in counts.items()},
        "top": [{"symbol": h["symbol"], "setups": h.get("setups", []), "price": h.get("price"),
                 "rsi": h.get("rsi")} for h in top],
    }


def save_breadth(paths: Paths, breadth: dict) -> None:
    row = {"date": breadth["as_of"], "scanned": breadth["scanned"]}
    row.update({f"n_{s}": breadth["counts"][s] for s in SETUPS})
    row.update({f"pct_{s}": breadth["pct"][s] for s in SETUPS})
    append_row(paths.state / "breadth.csv", row)


def describe(signal: dict) -> str:
    """The daily message, in Hebrew."""
    lines = [f"אות יומי ל-{signal['instrument']} לפי סגירת {signal['date']}: {signal['decision_he']}.",
             f"חשיפה עכשיו {signal['weight_now']:.0%}, יעד {signal['weight_target']:.0%}, לביצוע ב{signal['executes_at']}."]
    if signal.get("p_usual") is not None:
        lines.append(f"הסתברות לעלייה ב-{signal['horizon_days']} ימים: {signal['p_up']:.1%} "
                     f"(הרגיל של המודל בשנה האחרונה: {signal['p_usual']:.1%}, ציון {signal['score']:+.2f}).")
    acc = signal["account"]
    lines.append(f"החשבון הוירטואלי: {acc['equity']:,.0f}$ ({acc['return_pct']:+.2f}%, המדד {acc['index_return_pct']:+.2f}%).")
    b = signal.get("breadth")
    if b:
        lines.append(f"רוחב שוק ({b['scanned']:,} מניות נזילות, סגירת {b['as_of']}): "
                     f"{b['pct']['oversold']}% במכירת יתר, {b['pct']['breakout_20d']}% בפריצה לשיא 20 יום.")
    if signal.get("provisional_close"):
        lines.append("מחיר הסגירה של היום הזה זמני (Yahoo); FRED עוד לא פרסם אותו.")
    lines.append("זה פלט של ניסוי, לא ייעוץ השקעות; המודל לא הוכח כטוב יותר מקנה-והחזק.")
    return "\n".join(lines)
