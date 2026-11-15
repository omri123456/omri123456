"""Build reports/latest.html from runs/latest, runs/research and the paper account.

The page is self-contained (inline CSS/JS, data embedded as JSON) and written in Hebrew for
its reader. Every sentence that states a result is generated from the numbers, never typed in.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths
from ..trading import paper

TEMPLATE = Path(__file__).with_name("template.html")

STRATEGY_HE = {"cnn_all": "המודל (CNN, כל המידע)", "cnn_market": "CNN על מחירים בלבד",
               "linear_all": "מודל ליניארי (כל המידע)", "buy_and_hold": "קנה והחזק",
               "trend_200d": "כלל מגמה (ממוצע 200 יום)", "constant_mix": "חשיפה קבועה זהה"}
PERIOD_HE = {"development": "פיתוח", "holdout": "מבחן נקי (holdout)", "full": "כל התקופה"}
GROUP_HE = {"market": "מחיר ותנודתיות", "rates": "ריביות ואשראי", "cross": "מט\"ח ונפט",
            "liquidity": "נזילות ותנאים פיננסיים", "macro": "מאקרו", "forecasts": "תחזיות וציפיות",
            "news_idx": "מדדי חדשות", "wiki": "אירועי עולם (ויקיפדיה)", "calendar": "לוח שנה ו-FOMC"}
EVENT_HE = {
    "news: war": "חדשות: מלחמה", "news: terror": "חדשות: טרור", "news: disaster": "חדשות: אסון טבע/תאונה",
    "news: health": "חדשות: בריאות ומגפות", "news: economy": "חדשות: כלכלה", "news: central_bank": "חדשות: בנקים מרכזיים",
    "news: fin_crisis": "חדשות: משבר פיננסי", "news: markets": "חדשות: שווקים", "news: corporate": "חדשות: חברות",
    "news: energy": "חדשות: אנרגיה", "news: trade": "חדשות: סחר וסנקציות", "news: politics": "חדשות: פוליטיקה ובחירות",
    "news: us_policy": "חדשות: מדיניות ארה\"ב", "news: geopolitics": "חדשות: גיאופוליטיקה", "news: unrest": "חדשות: מחאות ואי-שקט",
    "news: crime_law": "חדשות: פשיעה ומשפט", "news: tech": "חדשות: טכנולוגיה", "news: mass casualties": "חדשות: אירוע רב-נפגעים",
    "index: geopolitical risk spike": "קפיצה במדד הסיכון הגיאופוליטי", "index: policy uncertainty spike": "קפיצה באי-ודאות המדיניות",
    "index: equity uncertainty spike": "קפיצה באי-ודאות שוק המניות", "index: news sentiment drop": "צניחה בסנטימנט החדשות",
    "fed: scheduled FOMC decision": "החלטת ריבית מתוכננת (FOMC)", "fed: unscheduled FOMC action": "צעד חירום של הפד",
    "market: one-day drop over 3%": "ירידה יומית של יותר מ-3%", "market: VIX jump over 20%": "קפיצה של יותר מ-20% ב-VIX",
}


def _read(path: Path, **kwargs) -> pd.DataFrame | None:
    return pd.read_csv(path, **kwargs) if path.exists() else None


def _pct(value, digits: int = 1, signed: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value * 100:{'+' if signed else ''}.{digits}f}%"


def _num(value, digits: int = 2, signed: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:{'+' if signed else ''}.{digits}f}"


def _table(headers: list[str], rows: list[list[str]], highlight: int | None = None) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for i, row in enumerate(rows):
        cls = ' class="lead"' if highlight is not None and i == highlight else ""
        cells = "".join(f"<t{'h scope=\"row\"' if j == 0 else 'd'}>{html.escape(str(c))}</t{'h' if j == 0 else 'd'}>"
                        for j, c in enumerate(row))
        body.append(f"<tr{cls}>{cells}</tr>")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _strategy_rows(strategies: pd.DataFrame, period: str) -> list[list[str]]:
    part = strategies[strategies["period"] == period]
    order = ["cnn_all", "buy_and_hold", "constant_mix", "trend_200d", "cnn_market", "linear_all"]
    rows = []
    for name in order:
        match = part[part["strategy"] == name]
        if match.empty:
            continue
        r = match.iloc[0]
        rows.append([STRATEGY_HE.get(name, name), _pct(r["cagr"]), _pct(r["vol"]), _num(r["sharpe"]),
                     _pct(r["max_drawdown"]), _pct(r["avg_exposure"], 0), _num(r["trades_per_year"], 0),
                     _pct(r["timing_gain"], 1, True), _num(r["timing_pvalue"], 2)])
    return rows


def _verdict(strategies: pd.DataFrame) -> tuple[str, str]:
    """One-sentence reading of the holdout result, derived from the numbers."""
    hold = strategies[strategies["period"] == "holdout"].set_index("strategy")
    if "cnn_all" not in hold.index or "buy_and_hold" not in hold.index:
        return "neutral", "עדיין אין תוצאות לתקופת המבחן הנקייה."
    model, bench = hold.loc["cnn_all"], hold.loc["buy_and_hold"]
    better_sharpe = model["sharpe"] > bench["sharpe"]
    better_return = model["cagr"] > bench["cagr"]
    p = model["timing_pvalue"]
    numbers = (f"תשואה שנתית {_pct(model['cagr'])} מול {_pct(bench['cagr'])} לקנה-והחזק, "
               f"Sharpe {_num(model['sharpe'])} מול {_num(bench['sharpe'])}, "
               f"ירידה מקסימלית {_pct(model['max_drawdown'])} מול {_pct(bench['max_drawdown'])}.")
    if better_sharpe and better_return and p < 0.05:
        return "good", "בתקופת המבחן הנקייה המודל היכה את קנה-והחזק, ויכולת התזמון מובהקת סטטיסטית. " + numbers
    if better_sharpe and p < 0.10:
        return "good", "בתקופת המבחן הנקייה המודל שיפר את היחס בין תשואה לסיכון, עם עדות חלשה ליכולת תזמון. " + numbers
    if better_sharpe:
        return "neutral", ("בתקופת המבחן הנקייה היחס בין תשואה לסיכון מעט טוב יותר מקנה-והחזק, "
                           "אבל אי אפשר להבדיל את התזמון ממזל. ") + numbers
    return "bad", "בתקופת המבחן הנקייה המודל לא היכה את קנה-והחזק. " + numbers


def _quality_rows(quality: pd.DataFrame) -> list[list[str]]:
    rows = []
    for variant in ("cnn_all", "cnn_market", "linear_all"):
        for period in ("development", "holdout"):
            part = quality[(quality["variant"] == variant) & (quality["period"] == period)]
            for _, r in part.sort_values("horizon").iterrows():
                rows.append([STRATEGY_HE.get(variant, variant), PERIOD_HE[period], f"{int(r['horizon'])} ימים",
                             _pct(r["up_rate"]), _pct(r["accuracy"]), _num(r["auc"], 3),
                             f"{_num(r['ic'], 3, True)} [{_num(r['ic_lo'], 2, True)}, {_num(r['ic_hi'], 2, True)}]",
                             _num(r.get("score_ic", np.nan), 3, True),
                             f"{_pct(r['skill'], 1, True)}"])
    return rows


def _series(frame: pd.DataFrame, columns: dict[str, str]) -> dict:
    return {"x": [str(d.date()) for d in frame.index],
            "series": [{"key": key, "name": name, "values": [None if not np.isfinite(v) else round(float(v), 4)
                                                             for v in frame[key].to_numpy(dtype=float)]}
                       for key, name in columns.items() if key in frame.columns]}


def build_report(cfg: dict, paths: Paths) -> Path:
    run = paths.runs / "latest"
    research = paths.runs / "research"
    strategies = _read(run / "strategies.csv")
    quality = _read(run / "prediction_quality.csv")
    if strategies is None or quality is None:
        raise RuntimeError("no backtest results yet: run `python -m marketlab backtest` first")
    info = json.loads((run / "run_info.json").read_text(encoding="utf-8")) if (run / "run_info.json").exists() else {}
    data: dict = {}
    parts: dict[str, str] = {}

    # ------------------------------------------------------------------ headline
    tone, verdict = _verdict(strategies)
    parts["VERDICT"] = f'<p class="verdict {tone}">{html.escape(verdict)}</p>'
    hold = strategies[strategies["period"] == "holdout"].set_index("strategy")
    tiles = []
    if "cnn_all" in hold.index and "buy_and_hold" in hold.index:
        m, b = hold.loc["cnn_all"], hold.loc["buy_and_hold"]
        for label, value, note in (
            ("תשואה שנתית", _pct(m["cagr"]), f"קנה-והחזק: {_pct(b['cagr'])}"),
            ("Sharpe", _num(m["sharpe"]), f"קנה-והחזק: {_num(b['sharpe'])}"),
            ("ירידה מקסימלית", _pct(m["max_drawdown"]), f"קנה-והחזק: {_pct(b['max_drawdown'])}"),
            ("חשיפה ממוצעת", _pct(m["avg_exposure"], 0), f"{_num(m['trades_per_year'], 0)} עסקאות בשנה"),
            ("מבחן תזמון (p)", _num(m["timing_pvalue"], 2), f"תרומת תזמון {_pct(m['timing_gain'], 1, True)} לשנה"),
        ):
            tiles.append(f'<div class="tile"><div class="tile-label">{html.escape(label)}</div>'
                         f'<div class="tile-value">{html.escape(value)}</div>'
                         f'<div class="tile-note">{html.escape(note)}</div></div>')
    parts["TILES"] = "".join(tiles)

    # ------------------------------------------------------------------ equity curve
    equity = _read(run / "equity_full.csv", index_col=0, parse_dates=True)
    if equity is not None:
        indexed = equity[[c for c in ("cnn_all", "buy_and_hold", "trend_200d") if c in equity.columns]]
        indexed = indexed / indexed.iloc[0] * 100.0
        weekly = indexed.iloc[::5]
        data["equity"] = _series(weekly, {"cnn_all": "המודל", "buy_and_hold": "קנה והחזק",
                                          "trend_200d": "כלל מגמה 200 יום"})
        data["equity"]["split"] = f"{int(cfg['walkforward']['dev_end_year']) + 1}-01-01"
        if "cnn_all_weight" in equity.columns:
            data["exposure"] = _series(equity[["cnn_all_weight"]].iloc[::5], {"cnn_all_weight": "חשיפה למדד"})
        year_end = indexed.groupby(indexed.index.year).last().round(1)
        parts["EQUITY_TABLE"] = _table(["סוף שנה", "המודל", "קנה והחזק", "כלל מגמה"],
                                       [[str(y)] + [f"{v:,.1f}" for v in row] for y, row in
                                        zip(year_end.index, year_end.to_numpy())])

    # ------------------------------------------------------------------ strategies
    headers = ["אסטרטגיה", "תשואה שנתית", "תנודתיות", "Sharpe", "ירידה מקס׳", "חשיפה", "עסקאות/שנה",
               "תרומת תזמון", "p תזמון"]
    for period in ("holdout", "development"):
        parts[f"STRATEGIES_{period.upper()}"] = _table(headers, _strategy_rows(strategies, period), highlight=0)

    yearly = _read(run / "yearly_full.csv", index_col=0)
    if yearly is not None:
        data["yearly"] = {"x": [str(y) for y in yearly.index],
                          "series": [{"key": k, "name": n, "values": [round(float(v) * 100, 1) for v in yearly[k]]}
                                     for k, n in (("cnn_all", "המודל"), ("buy_and_hold", "קנה והחזק"))
                                     if k in yearly.columns]}
        parts["YEARLY_TABLE"] = _table(["שנה", "המודל", "קנה והחזק", "הפרש"],
                                       [[str(y), _pct(r["cnn_all"]), _pct(r["buy_and_hold"]),
                                         _pct(r["cnn_all"] - r["buy_and_hold"], 1, True)]
                                        for y, r in yearly.iterrows()])

    # ------------------------------------------------------------------ prediction quality
    parts["QUALITY"] = _table(["מודל", "תקופה", "אופק", "שיעור עליות בפועל", "דיוק כיוון", "AUC",
                               "IC [רווח סמך 90%]", "IC של הציון היחסי", "שיפור מול שיעור הבסיס"],
                              _quality_rows(quality))

    # ------------------------------------------------------------------ what each kind of information adds
    for period in ("holdout", "development"):
        ablation = _read(run / f"ablation_{period}.csv", index_col=0)
        if ablation is not None and len(ablation):
            ablation = ablation.sort_values("logloss_gain", ascending=False)
            data[f"ablation_{period}"] = {
                "labels": [GROUP_HE.get(g, g) for g in ablation.index],
                "values": [round(float(v) * 1e4, 2) for v in ablation["logloss_gain"]]}

    # ------------------------------------------------------------------ research
    events = _read(research / "event_study.csv", index_col=0)
    if events is not None and len(events):
        baseline = json.loads((research / "baseline.json").read_text()) if (research / "baseline.json").exists() else {}
        events = events.sort_values("car20")
        rows = [[EVENT_HE.get(name, name), str(int(r["n"])), _pct(r["day0"], 2, True), _pct(r["car5"], 2, True),
                 _pct(r["car20"], 2, True), _num(r["t20"], 1, True), _pct(r["hit20"], 0), _num(r["vol_ratio"], 2)]
                for name, r in events.iterrows()]
        parts["EVENTS"] = _table(["סוג אירוע", "מספר אירועים", "יום האירוע", "5 ימים אחרי", "20 ימים אחרי",
                                  "t (20 ימים)", "% עליות אחרי 20 יום", "תנודתיות אחרי/לפני"], rows)
        parts["EVENTS_BASELINE"] = (f"לשם השוואה: בלי שום אירוע, המדד עלה ב-{_pct(baseline.get('hit20', np.nan), 0)} "
                                    f"מחלונות 20 הימים וב-{_pct(baseline.get('hit5', np.nan), 0)} מחלונות 5 הימים.")
        paths_table = _read(research / "event_paths.csv", index_col=0)
        if paths_table is not None:
            wanted = [c for c in ("market: one-day drop over 3%", "market: VIX jump over 20%",
                                  "fed: scheduled FOMC decision", "news: fin_crisis") if c in paths_table.columns]
            data["event_paths"] = {"x": [int(d) for d in paths_table.index],
                                   "series": [{"key": c, "name": EVENT_HE.get(c, c),
                                               "values": [round(float(v) * 100, 3) for v in paths_table[c]]}
                                              for c in wanted]}

    ic = _read(research / "feature_ic.csv")
    if ic is not None and len(ic):
        top = ic[(ic["horizon"] == int(cfg["model"]["trade_horizon"])) & ic["same_sign"]].head(12)
        parts["FEATURES"] = _table(["פיצ'ר", "קבוצה", "IC מוקדם", "IC מאוחר", "t מוקדם", "t מאוחר"],
                                   [[r["feature"], GROUP_HE.get(r["group"], r["group"]), _num(r["ic_early"], 3, True),
                                     _num(r["ic_late"], 3, True), _num(r["t_early"], 1, True), _num(r["t_late"], 1, True)]
                                    for _, r in top.iterrows()])
        share = float(ic[ic["horizon"] == int(cfg["model"]["trade_horizon"])]["same_sign"].mean())
        parts["FEATURES_NOTE"] = (f"רק {_pct(share, 0)} מהפיצ'רים שומרים על אותו כיוון קשר בשתי התקופות; "
                                  "אם אין שום קשר אמיתי מצפים ל-50%.")

    # ------------------------------------------------------------------ paper account
    state = paper.load_state(paths.state)
    if state is not None:
        s = paper.summary(state)
        pending = s.get("pending")
        order = (f"הוראה ממתינה: משקל יעד {_pct(pending['target_weight'], 0)} במדד "
                 f"(הוחלט ב-{pending['decided_on']}, הסתברות לעלייה {_pct(pending['p_up'])}), "
                 "לביצוע בסגירת יום המסחר הבא." if pending else "אין הוראה ממתינה.")
        parts["PAPER"] = (
            '<div class="tiles">'
            f'<div class="tile"><div class="tile-label">שווי החשבון</div><div class="tile-value">${s["equity"]:,.0f}</div>'
            f'<div class="tile-note">נפתח ב-{s["opened_on"]} עם ${state["initial_cash"]:,.0f}</div></div>'
            f'<div class="tile"><div class="tile-label">תשואה מאז הפתיחה</div><div class="tile-value">{s["return_pct"]:+.2f}%</div>'
            f'<div class="tile-note">המדד באותה תקופה: {s["index_return_pct"]:+.2f}%</div></div>'
            f'<div class="tile"><div class="tile-label">חשיפה נוכחית</div><div class="tile-value">{_pct(s["weight"], 0)}</div>'
            f'<div class="tile-note">{s["trades"]} עסקאות, עלויות ${s["costs_paid"]:,.2f}</div></div>'
            f'</div><p>{html.escape(order)}</p>'
            f'<p class="muted">עודכן לאחרונה לפי סגירת {s["last_mark"]}. ימי מסחר שעברו מאז הפתיחה: {s["days"]}.</p>')
    else:
        parts["PAPER"] = '<p class="muted">חשבון המסחר הוירטואלי עוד לא נפתח: <code>python -m marketlab paper init</code>.</p>'

    # ------------------------------------------------------------------ facts line
    facts = []
    if info:
        facts = [f"נתונים עד {info.get('data_last_day', '?')}",
                 f"{info.get('panel_days', '?'):,} ימי מסחר × {info.get('panel_features', '?')} ערוצים",
                 f"{info.get('events', 0):,} אירועי חדשות",
                 f"מודל של {info.get('parameters', '?'):,} פרמטרים × {info.get('seeds', '?')} אתחולים",
                 f"שנות מבחן {info.get('first_test_year', '?')}–{info.get('last_test_year', '?')}"]
    parts["FACTS"] = " · ".join(facts)
    parts["TARGET"] = html.escape(str(cfg["project"]["target"]))
    parts["DEV_END"] = str(cfg["walkforward"]["dev_end_year"])
    parts["HOLD_START"] = str(int(cfg["walkforward"]["dev_end_year"]) + 1)
    parts["HORIZON"] = str(cfg["model"]["trade_horizon"])
    parts["COST"] = _num(float(cfg["trading"]["cost_bps"]), 0)

    page = TEMPLATE.read_text(encoding="utf-8")
    for key in ("VERDICT", "TILES", "EQUITY_TABLE", "STRATEGIES_HOLDOUT", "STRATEGIES_DEVELOPMENT", "YEARLY_TABLE",
                "QUALITY", "EVENTS", "EVENTS_BASELINE", "FEATURES", "FEATURES_NOTE", "PAPER", "FACTS", "TARGET",
                "DEV_END", "HOLD_START", "HORIZON", "COST"):
        page = page.replace("{{" + key + "}}", parts.get(key, '<p class="muted">אין נתונים עדיין.</p>'
                                                         if key.isupper() and key not in ("FACTS",) else ""))
    page = page.replace("{{DATA}}", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    paths.reports.mkdir(parents=True, exist_ok=True)
    out = paths.reports / "latest.html"
    out.write_text(page, encoding="utf-8")
    return out
