# -*- coding: utf-8 -*-
"""
台股 CSV 驗證腳本（StockExpert 威科夫模組）
==========================================

用法：
    python3 run_validation.py data/*.csv --out results
    python3 run_validation.py data/2330.csv --index data/TAIEX.csv --out results
    python3 run_validation.py data/2330.csv --trades my_trades.csv --out results

輸入：
  - 日 K CSV：欄位可為英文（Date/Open/High/Low/Close/Volume[/Adj Close]）或中文
    （日期/開盤價/最高價/最低價/收盤價/成交股數…），支援民國年與千分位；
    建議使用「還原權值」價格，避免除權息缺口被誤判為 SOW/破冰。
  - --index：大盤（加權指數）CSV，用於原書的相對強弱選股〔p253-258〕。
  - --trades：使用者買賣紀錄（欄位 date, side, price[, symbol]），輸出決策檢討。

輸出（每檔一個資料夾）：
  events.csv / signals.csv / trs.csv / bars.csv / trades.csv / event_study.csv / card.json
  以及總表 summary.md：各訊號的交易統計、與「隨機進場＋相同出場規則」基準的比較。

方法學（避免自我欺騙）：
  - 所有事件以「確認K線」產生，回測於下一根開盤進場（無未來函數）。
  - 交易成本：買 0.1425%、賣 0.4425%（含 0.3% 證交稅）。
  - 基準：同一檔股票、相同停損距離（以 ATR 計）與相同出場規則的隨機進場，重複 200 次。
  - 參數應以前段樣本校準、後段樣本驗證（--split 2018-01-01 會分段輸出）。
  - 週線過濾：同時回報「不過濾／過濾逆週線背景的多方訊號」兩組結果〔p147、p225〕。
  - 參數穩健性（--sensitivity）：主要【參數】門檻 ±20% 單變數調整；原書規則（如 Spring 1~3 根）不調。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import replace

import numpy as np
import pandas as pd

from wyckoff_engine import (Params, WyckoffScanner, backtest, summarize_trades, event_study,
                            snapshot, audit_trades, load_ohlcv_csv, relative_strength, LONG_KINDS)

# 單變數敏感度（只調整【參數】類門檻）
SENS_GRID = {
    "vol_climax": [1.6, 2.4],
    "vol_high": [1.2, 1.8],
    "st_vol_ratio": [0.64, 0.96],
    "ar_min_atr": [1.6, 2.4],
    "shakeout_depth_atr": [0.8, 1.2],
    "min_decline": [0.12, 0.18],
    "min_rise": [0.16, 0.24],
    "confirm_bars": [3, 7],
}


def trade_sums(t: pd.DataFrame) -> dict:
    """可跨股票加總的交易統計（筆數、淨報酬和、勝場、R 和）。"""
    if t is None or len(t) == 0:
        return dict(trades=0, sum_net=0.0, wins=0, sum_R=0.0, n_R=0)
    return dict(trades=len(t), sum_net=float(t["net_ret"].sum()), wins=int((t["net_ret"] > 0).sum()),
                sum_R=float(t["R"].sum(skipna=True)), n_R=int(t["R"].notna().sum()))


def sums_to_stats(s: dict) -> dict:
    n = s["trades"]
    return dict(trades=n, win_rate=s["wins"] / n if n else math.nan, avg_net=s["sum_net"] / n if n else math.nan,
                avg_R=s["sum_R"] / s["n_R"] if s["n_R"] else math.nan)


def sensitivity(df: pd.DataFrame, p: Params, grid: dict = SENS_GRID) -> pd.DataFrame:
    """主要參數 ±20% 的單變數敏感度。結果平滑＝穩健；只有單一組合有效＝過度擬合的警訊。"""
    rows = [dict(param="(基準)", value="", **trade_sums(backtest(WyckoffScanner(df, p).run())))]
    for name, vals in grid.items():
        for v in vals:
            t = backtest(WyckoffScanner(df, replace(p, **{name: v})).run())
            rows.append(dict(param=name, value=v, **trade_sums(t)))
    return pd.DataFrame(rows)


def to_md(df: pd.DataFrame, index: bool = True) -> str:
    """不依賴 tabulate 的簡易 Markdown 表格。"""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    fmt = lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(fmt(x) for x in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(rows)


def random_baseline(sc: WyckoffScanner, n_trades: int, stop_atr: float, reps: int = 200, seed: int = 0) -> dict:
    """隨機進場基準：與策略相同的出場規則，只把進場點換成隨機。"""
    rng = np.random.default_rng(seed)
    d = sc.d
    valid = np.where(d["atr"].notna().to_numpy() & (np.arange(len(d)) > 60) & (np.arange(len(d)) < len(d) - 10))[0]
    if n_trades == 0 or len(valid) < n_trades:
        return {}
    exp = []
    orig = sc.signals
    for _ in range(reps):
        picks = np.sort(rng.choice(valid, size=n_trades, replace=False))
        fake = [s for s in orig if s["kind"] not in LONG_KINDS]
        for i in picks:
            fake.append(dict(idx=int(i), date=d.at[i, "date"], kind="L_RANDOM", side="long",
                             stop=float(d.at[i, "close"] - stop_atr * d.at[i, "atr"]), note="", page="", tr_id=-1))
        sc.signals = fake
        t = backtest(sc, long_kinds=("L_RANDOM",))
        if len(t):
            exp.append(t["net_ret"].mean())
    sc.signals = orig
    return dict(rand_avg_net_mean=float(np.mean(exp)) if exp else math.nan,
                rand_avg_net_p95=float(np.percentile(exp, 95)) if exp else math.nan)


def run_one(sym: str, df: pd.DataFrame, out: str, p: Params, index_df=None, trades_df=None, split=None,
            sens: bool = False):
    sc = WyckoffScanner(df, p).run()
    os.makedirs(out, exist_ok=True)
    ev, sig, trs = sc.events_df(), sc.signals_df(), sc.trs_df()
    ev.to_csv(f"{out}/events.csv", index=False, encoding="utf-8-sig")
    sig.to_csv(f"{out}/signals.csv", index=False, encoding="utf-8-sig")
    trs.to_csv(f"{out}/trs.csv", index=False, encoding="utf-8-sig")
    sc.bars_df().to_csv(f"{out}/bars.csv", index=False, encoding="utf-8-sig")
    t = backtest(sc)
    t.to_csv(f"{out}/trades.csv", index=False, encoding="utf-8-sig")
    event_study(sc).to_csv(f"{out}/event_study.csv", index=False, encoding="utf-8-sig")
    with open(f"{out}/card.json", "w", encoding="utf-8") as f:
        json.dump(snapshot(sc, symbol=sym, index_df=index_df), f, ensure_ascii=False, indent=2, default=str)
    res = dict(symbol=sym, bars=len(df), start=str(df["date"].iloc[0])[:10], end=str(df["date"].iloc[-1])[:10],
               n_tr=len(trs), n_events=len(ev), n_trades=len(t),
               buy_hold=float(df["close"].iloc[-1] / df["close"].iloc[0] - 1))
    if len(t):
        res.update(win_rate=float((t["net_ret"] > 0).mean()), avg_net=float(t["net_ret"].mean()),
                   avg_R=float(t["R"].mean()))
        stop_atr = float(np.nanmedian(t["risk_atr"])) if t["risk_atr"].notna().any() else 1.5
        stop_atr = 1.5 if (not np.isfinite(stop_atr) or stop_atr <= 0) else min(stop_atr, 5.0)
        res.update(random_baseline(sc, len(t), stop_atr))
        if split:
            t["period"] = np.where(pd.to_datetime(t["entry_date"]) < pd.Timestamp(split), "in_sample", "out_sample")
            for per, g in t.groupby("period"):
                res[f"{per}_trades"], res[f"{per}_avg_net"] = len(g), float(g["net_ret"].mean())
    # 週線過濾比較（同一次掃描，只改回測的進場條件）
    t_wf = backtest(sc, weekly_filter=True)
    t_wf.to_csv(f"{out}/trades_weekly_filter.csv", index=False, encoding="utf-8-sig")
    res.update({f"wf_{k}": v for k, v in sums_to_stats(trade_sums(t_wf)).items()})
    res["_sums_all"], res["_sums_wf"] = trade_sums(t), trade_sums(t_wf)
    if sens:
        sv = sensitivity(df, p)
        sv.to_csv(f"{out}/sensitivity.csv", index=False, encoding="utf-8-sig")
        res["_sens"] = sv
    if index_df is not None:
        rs = relative_strength(df, index_df)
        rs.to_csv(f"{out}/relative_strength.csv", index=False, encoding="utf-8-sig")
        res["rs_score_last"] = float(rs["rs_score"].iloc[-1]) if len(rs) else math.nan
        res["rs_strong_last"] = bool(rs["rs_strong"].iloc[-1]) if len(rs) else False
    if trades_df is not None:
        tt = trades_df if "symbol" not in trades_df else trades_df[trades_df["symbol"].astype(str) == str(sym)]
        if len(tt):
            audit_trades(sc, tt).to_csv(f"{out}/trade_review.csv", index=False, encoding="utf-8-sig")
    return res, t


def main():
    ap = argparse.ArgumentParser(description="威科夫訊號引擎：台股 CSV 驗證")
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default="results")
    ap.add_argument("--index", default=None, help="大盤 CSV（相對強弱）")
    ap.add_argument("--trades", default=None, help="使用者買賣紀錄 CSV")
    ap.add_argument("--split", default=None, help="樣本內/外分界日，如 2018-01-01")
    ap.add_argument("--sensitivity", action="store_true", help="執行主要參數 ±20% 的單變數穩健性測試（較慢）")
    a = ap.parse_args()
    p = Params()
    files = sorted({f for pat in a.csv for f in glob.glob(pat)})
    index_df = None
    if a.index:
        index_df = list(load_ohlcv_csv(a.index).values())[0]
    trades_df = pd.read_csv(a.trades) if a.trades else None
    rows, all_t = [], []
    for f in files:
        try:
            data = load_ohlcv_csv(f)
        except Exception as e:
            print(f"[略過] {f}: {e}")
            continue
        for sym, df in data.items():
            if len(df) < 150:
                print(f"[略過] {f}:{sym} 資料少於 150 根")
                continue
            name = sym if sym != "UNKNOWN" else os.path.splitext(os.path.basename(f))[0]
            r, t = run_one(name, df, os.path.join(a.out, name), p, index_df, trades_df, a.split, a.sensitivity)
            rows.append(r)
            if len(t):
                t.insert(0, "symbol", name)
                all_t.append(t)
            print(f"[完成] {name}: TR {r['n_tr']}、事件 {r['n_events']}、交易 {r['n_trades']}")
    if not rows:
        print("沒有可分析的資料。")
        return
    os.makedirs(a.out, exist_ok=True)
    priv = [k for k in ("_sums_all", "_sums_wf", "_sens")]
    summ = pd.DataFrame([{k: v for k, v in r.items() if k not in priv} for r in rows])
    summ.to_csv(os.path.join(a.out, "summary_by_symbol.csv"), index=False, encoding="utf-8-sig")
    lines = ["# 威科夫訊號驗證總表", "", f"- 檔案數：{len(files)}；股票數：{len(rows)}", ""]
    if all_t:
        T = pd.concat(all_t, ignore_index=True)
        T.to_csv(os.path.join(a.out, "all_trades.csv"), index=False, encoding="utf-8-sig")
        lines += ["## 各訊號交易統計（含交易成本）", "", to_md(summarize_trades(T)), ""]
    # 週線過濾比較
    agg = lambda key: {k: sum(r[key][k] for r in rows) for k in rows[0][key]}
    wf_tab = pd.DataFrame([dict(版本="不過濾", **sums_to_stats(agg("_sums_all"))),
                           dict(版本="過濾逆週線背景的多方訊號", **sums_to_stats(agg("_sums_wf")))])
    lines += ["## 週線背景過濾比較〔p147、p225〕", "", to_md(wf_tab, index=False), ""]
    # 參數穩健性
    if a.sensitivity and all("_sens" in r for r in rows):
        S = pd.concat([r["_sens"].assign(symbol=r["symbol"]) for r in rows], ignore_index=True)
        S.to_csv(os.path.join(a.out, "sensitivity_all.csv"), index=False, encoding="utf-8-sig")
        g = S.groupby(["param", "value"], sort=False)[["trades", "sum_net", "wins", "sum_R", "n_R"]].sum().reset_index()
        st = pd.DataFrame([dict(param=x.param, value=x.value, **sums_to_stats(x._asdict())) for x in g.itertuples(index=False)])
        lines += ["## 參數穩健性（單變數 ±20%）", "", to_md(st, index=False), "",
                  "> 判讀：各變體的平均淨報酬與勝率應與基準接近且方向一致；若只有基準有效、稍一調整就轉負，代表過度擬合。", ""]
    lines += ["## 各股摘要（含隨機進場基準）", "", to_md(summ, index=False), "",
              "> 判讀：策略平均淨報酬應明顯高於 rand_avg_net_mean，最好高於 rand_avg_net_p95，"
              "且樣本外（out_sample）不顯著劣化，才算有效。"]
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"總表：{os.path.join(a.out, 'summary.md')}")


if __name__ == "__main__":
    main()
