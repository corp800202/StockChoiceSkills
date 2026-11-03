# -*- coding: utf-8 -*-
"""
v1.3 校準：只用樣本內（< 2024-01-01）決定調整，樣本外（≥ 2024-01-01）確認。
=====================================================================

用法：
    python3 calibrate_tw.py --data data/tw_daily --out results_tw --volume-unit 1000

做法：
  - 每檔股票只掃描一次（事件與訊號不變），在同一份訊號上比較多種「交易規則」：
    最小停損距離、確認期長度、各項持有期出場規則、週線過濾、流動性門檻、訊號種類。
  - 每個版本分別統計樣本內、樣本外的筆數、勝率、平均淨報酬、獲利因子。
  - 對「基準」與「建議版本」另跑隨機進場基準（相同停損距離、相同出場規則與過濾條件）。
  - 版本很多時，樣本內最好的版本一定偏樂觀；以樣本外是否維持來判斷，不以樣本內排名決定。
"""
from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

from wyckoff_engine import Params, WyckoffScanner, backtest, load_ohlcv_csv, liquidity_mask, LONG_KINDS

warnings.filterwarnings("ignore")
SPLIT = pd.Timestamp("2024-01-01")
STRUCT = ("L_SPRING", "L_TEST", "L_LPS", "L_BU")

VARIANTS = {
    "基準（v1.2 規則）": {},
    "最小停損 0.5 ATR": dict(p=dict(min_stop_atr=0.5)),
    "最小停損 1.0 ATR": dict(p=dict(min_stop_atr=1.0)),
    "最小停損 1.5 ATR": dict(p=dict(min_stop_atr=1.5)),
    "最小停損 2.0 ATR": dict(p=dict(min_stop_atr=2.0)),
    "確認期 3 根": dict(p=dict(confirm_bars=3)),
    "確認期 7 根": dict(p=dict(confirm_bars=7)),
    "確認期 10 根": dict(p=dict(confirm_bars=10)),
    "關閉：放量滯漲後量縮出場": dict(p=dict(exit_on_stall=False)),
    "關閉：SOT 縮緊停損": dict(p=dict(tighten_on_sot=False)),
    "關閉：SOT＋回調>50% 出場": dict(p=dict(exit_on_sot_retr=False)),
    "關閉：跌破需求線出場": dict(p=dict(exit_on_demand_line_break=False)),
    "週線過濾": dict(weekly=True),
    "流動性 ≥ 5,000 萬元": dict(liq=5e7),
    "不含 L_PULLBACK": dict(kinds=("L_SPRING", "L_TEST", "L_LPS", "L_BU", "L_CORNER")),
    "只用區間結構訊號": dict(kinds=STRUCT),
    "最小停損 1.0＋流動性": dict(p=dict(min_stop_atr=1.0), liq=5e7),
    "最小停損 1.5＋流動性": dict(p=dict(min_stop_atr=1.5), liq=5e7),
    "最小停損 1.5＋流動性＋週線": dict(p=dict(min_stop_atr=1.5), liq=5e7, weekly=True),
    "最小停損 2.0＋流動性＋週線": dict(p=dict(min_stop_atr=2.0), liq=5e7, weekly=True),
    "最小停損 1.5＋流動性＋區間結構": dict(p=dict(min_stop_atr=1.5), liq=5e7, kinds=STRUCT),
}
RANDOM_FOR = ("基準（v1.2 規則）", "最小停損 1.5＋流動性", "最小停損 1.5＋流動性＋區間結構")
RAND_REPS = 30


def sums(t: pd.DataFrame) -> dict:
    out = {}
    for per in ("in", "out"):
        if t is None or len(t) == 0:
            g = pd.DataFrame(columns=["net_ret", "bars"])
        else:
            g = t[(pd.to_datetime(t["entry_date"]) < SPLIT) == (per == "in")]
        w = g["net_ret"] > 0
        out[per] = np.array([len(g), g["net_ret"].sum(), w.sum(), g.loc[w, "net_ret"].sum(),
                             -g.loc[~w, "net_ret"].sum(), g["bars"].sum()], dtype=float)
    return out


def random_sums(sc, base_trades, p, kinds_mask_weekly, entry_mask, reps, seed) -> dict:
    """隨機進場：每檔相同筆數（依樣本期分開）、相同停損距離（該檔中位數）、相同規則與過濾。"""
    d = sc.d
    res = {"in": np.zeros((reps, 3)), "out": np.zeros((reps, 3))}
    if base_trades is None or len(base_trades) == 0:
        return res
    stop_atr = float(np.nanmedian(base_trades["risk_atr"])) if base_trades["risk_atr"].notna().any() else 1.5
    stop_atr = 1.5 if (not np.isfinite(stop_atr) or stop_atr <= 0) else min(stop_atr, 5.0)
    dates = pd.to_datetime(d["date"]).to_numpy()
    ok = d["atr"].notna().to_numpy() & (np.arange(len(d)) > 60) & (np.arange(len(d)) < len(d) - 10)
    rng = np.random.default_rng(seed)
    orig = sc.signals
    keep = [s for s in orig if s["kind"] not in LONG_KINDS]
    for per in ("in", "out"):
        n_tr = int(((pd.to_datetime(base_trades["entry_date"]) < SPLIT) == (per == "in")).sum())
        valid = np.where(ok & ((dates < np.datetime64(SPLIT)) == (per == "in")))[0]
        if n_tr == 0 or len(valid) < n_tr:
            continue
        for r in range(reps):
            picks = np.sort(rng.choice(valid, size=n_tr, replace=False))
            sc.signals = keep + [dict(idx=int(i), date=d.at[i, "date"], kind="L_RANDOM", side="long",
                                      stop=float(d.at[i, "close"] - stop_atr * d.at[i, "atr"]), note="", page="",
                                      tr_id=-1) for i in picks]
            t = backtest(sc, long_kinds=("L_RANDOM",), p=p, weekly_filter=kinds_mask_weekly, entry_mask=entry_mask)
            if len(t):
                res[per][r] = [t["net_ret"].sum(), len(t), (t["net_ret"] > 0).sum()]
    sc.signals = orig
    return res


def run_stock(args):
    code, data_dir, unit, with_random = args
    df = list(load_ohlcv_csv(os.path.join(data_dir, f"{code}.csv")).values())[0]
    base = Params()
    sc = WyckoffScanner(df, base).run()
    liq = liquidity_mask(sc.d, unit, 5e7)
    out, rand = {}, {}
    for name, v in VARIANTS.items():
        p = replace(base, **v.get("p", {}))
        t = backtest(sc, long_kinds=v.get("kinds", LONG_KINDS), p=p, weekly_filter=v.get("weekly", False),
                     entry_mask=liq if v.get("liq") else None)
        out[name] = sums(t)
        if with_random and name in RANDOM_FOR:
            seed = (int(code) if str(code).isdigit() else 7) + 17 * list(VARIANTS).index(name)
            rand[name] = random_sums(sc, t, p, v.get("weekly", False), liq if v.get("liq") else None, RAND_REPS, seed)
    return code, out, rand


def stats(a: np.ndarray) -> dict:
    n, s, w, gp, gl, bars = a
    return dict(trades=int(n), win_rate=w / n if n else np.nan, avg_net=s / n if n else np.nan,
                pf=gp / gl if gl > 0 else np.nan, avg_bars=bars / n if n else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tw_daily")
    ap.add_argument("--out", default="results_tw")
    ap.add_argument("--volume-unit", type=float, default=1000.0, help="成交量單位：張＝1000、股＝1")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--no-random", action="store_true")
    a = ap.parse_args()
    man = pd.read_csv(os.path.join(a.data, "manifest.csv"), dtype={"code": str})
    codes = man.loc[man["group"] != "index", "code"].tolist()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        res = list(ex.map(run_stock, [(c, a.data, a.volume_unit, not a.no_random) for c in codes], chunksize=2))
    rows = []
    for name in VARIANTS:
        tot = {per: sum(r[1][name][per] for r in res) for per in ("in", "out")}
        si, so = stats(tot["in"]), stats(tot["out"])
        rows.append(dict(variant=name, **{f"in_{k}": v for k, v in si.items()}, **{f"out_{k}": v for k, v in so.items()}))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "calibration.csv"), index=False, encoding="utf-8-sig")
    rb = []
    if not a.no_random:
        for name in RANDOM_FOR:
            for per in ("in", "out"):
                M = sum(r[2][name][per] for r in res if name in r[2])
                avg = M[:, 0] / np.maximum(M[:, 1], 1)
                strat = R.loc[R["variant"] == name, f"{per}_avg_net"].iloc[0]
                rb.append(dict(variant=name, period=per, rand_avg=avg.mean(), rand_p05=np.quantile(avg, 0.05),
                               rand_p95=np.quantile(avg, 0.95), strat_avg=strat, beat_share=(strat > avg).mean()))
        pd.DataFrame(rb).to_csv(os.path.join(a.out, "calibration_random.csv"), index=False, encoding="utf-8-sig")

    f = lambda x, pct=True: "—" if not np.isfinite(x) else (f"{x * 100:+.2f}%" if pct else f"{x:.2f}")
    L = ["# v1.3 校準結果（樣本內決定、樣本外確認）", "",
         f"- 樣本內：2019-09～2023-12；樣本外：2024-01～2026-10；150 檔；價格未還原。", "",
         "| 版本 | 內：筆數 | 內：勝率 | 內：平均淨報酬 | 內：獲利因子 | 外：筆數 | 外：勝率 | 外：平均淨報酬 | 外：獲利因子 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in R.itertuples():
        L.append(f"| {r.variant} | {r.in_trades:,} | {f(r.in_win_rate)} | {f(r.in_avg_net)} | {f(r.in_pf, False)} | "
                 f"{r.out_trades:,} | {f(r.out_win_rate)} | {f(r.out_avg_net)} | {f(r.out_pf, False)} |")
    if rb:
        L += ["", "## 隨機進場基準（相同停損距離、出場規則與過濾條件）", "",
              "| 版本 | 樣本期 | 策略平均淨報酬 | 隨機平均 | 隨機 5%～95% | 策略勝過隨機的輪數 |", "|---|---|---|---|---|---|"]
        for r in rb:
            L.append(f"| {r['variant']} | {'樣本內' if r['period'] == 'in' else '樣本外'} | {f(r['strat_avg'])} | "
                     f"{f(r['rand_avg'])} | {f(r['rand_p05'])} ～ {f(r['rand_p95'])} | {r['beat_share'] * 100:.0f}% |")
    with open(os.path.join(a.out, "calibration.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
