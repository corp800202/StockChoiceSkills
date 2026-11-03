# -*- coding: utf-8 -*-
"""
台股樣本驗證（v1.3）
====================

用法：
    python3 validate_tw.py --data data/tw_daily --out results_tw
    python3 validate_tw.py --data data/tw_daily --out results_tw --sensitivity      # 加跑參數穩健性（較慢）

輸入：data 資料夾內的 manifest.csv（code,name,market,industry,group,...）、<代號>.csv、
      index_TAIEX.csv（上市股比較基準）、index_TPEX.csv（上櫃股比較基準）。

產出（out 資料夾）：
  summary.md            給人看的總表與自動判讀
  study_summary.csv     每種訊號／事件的前瞻報酬統計
  trades.csv            所有回測交易（含組別、樣本期、相對強弱旗標）
  random_baseline.csv   隨機進場基準（每一輪的彙總）
  sensitivity.csv       （--sensitivity）參數 ±20% 的單變數穩健性
  study_detail.csv.gz   每一個訊號／事件的明細（體積較大，不納入版控）

方法（避免自我欺騙）：
  - 前瞻報酬：以訊號／事件「確認 K 線」的下一根開盤為起點，到第 h 根收盤（h = 5、10、20、60），無未來函數。
  - 相對個股：減去同一檔股票所有 K 線同一算法的平均前瞻報酬（排除個股長期漂移與存活者偏差的一部分）。
  - 相對大盤：減去同一段日期的大盤報酬（上市用加權指數、上櫃用櫃買指數）。
  - 信賴區間：以「股票」為單位的集群自助法（2,000 次），避免同一檔股票報酬重疊造成過度自信。
  - 預期方向【延伸】：多方事件／訊號預期 +，空方與警示預期 −（見 EXPECTED_DIR），用來判定「方向是否正確」。
  - 回測與隨機基準沿用 wyckoff_engine.backtest（一次一筆多單、下一根開盤進場、含交易成本）。
"""
from __future__ import annotations

import argparse
import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pandas as pd

from wyckoff_engine import (Params, WyckoffScanner, backtest, load_ohlcv_csv, relative_strength, LONG_KINDS)

warnings.filterwarnings("ignore")

H = (5, 10, 20, 60)
SPLIT = "2024-01-01"          # 樣本內（之前）／樣本外（之後）
RAND_REPS = 100

# 結構事件的預期方向【延伸：依原書對各事件後續走勢的描述】
EXPECTED_DIR = {
    "PS": 1, "SC": 1, "AR": -1, "ST": 1, "ST(量大)": -1, "SA": 1, "Spring": 1, "Shakeout": 1, "Test": 1,
    "Spring失敗": -1, "SOS": 1, "JOC": 1, "LPS": 1, "BU": 1, "SOS失效": -1, "MarkupE": 1,
    "PSY": -1, "BC": -1, "AR(自然回落)": 1, "ST(派發)": -1, "UT": -1, "UTAD": -1, "UT確認": -1,
    "SOW": -1, "ICE": -1, "LPSY": -1, "SOW失效": 1, "MarkdownE": -1,
    "OVERSOLD": 1, "OVERBOUGHT": -1, "SUPPLY_LINE_BREAK": 1, "DEMAND_LINE_BREAK": -1,
    "SOT_CH_UP": -1, "SOT_CH_DOWN": 1, "PB": 1, "CORNER": 1, "NoDemandRally": -1,
    "WAVE_STALL_UP": -1, "WAVE_STALL_DOWN": 1,
    "RETR_GT50": -1, "RALLY_LT50": -1, "SOT_up": -1, "SOT_down": 1, "突破(量不足)": -1,
}

SENS_GRID = {
    "vol_climax": [1.6, 2.4], "vol_high": [1.2, 1.8], "st_vol_ratio": [0.64, 0.96], "ar_min_atr": [1.6, 2.4],
    "shakeout_depth_atr": [0.8, 1.2], "min_decline": [0.12, 0.18], "min_rise": [0.16, 0.24], "confirm_bars": [3, 7],
    "pb_max_retr": [0.4, 0.6], "pb_ud_min": [0.88, 1.32],
}


def signal_dir(kind: str) -> int:
    return 1 if kind.startswith("L_") else -1


def event_dir(kind: str) -> int:
    return EXPECTED_DIR.get(kind, EXPECTED_DIR.get(kind.split("(")[0], 0))


# ---------------------------------------------------------------------------
# 資料
# ---------------------------------------------------------------------------
def load_df(path: str) -> pd.DataFrame:
    return list(load_ohlcv_csv(path).values())[0]


@lru_cache(maxsize=4)
def load_index(data_dir: str, market: str) -> pd.DataFrame:
    name = "index_TPEX.csv" if market == "上櫃" else "index_TAIEX.csv"
    return load_df(os.path.join(data_dir, name))


def forward_returns(d: pd.DataFrame, idx: pd.DataFrame) -> dict:
    """fwd[h][i] = close[i+h] / open[i+1] − 1；mkt[h][i] 為同一段日期的大盤報酬。"""
    o, c = d["open"].to_numpy(float), d["close"].to_numpy(float)
    dates = pd.to_datetime(d["date"]).to_numpy()
    io = pd.Series(idx["open"].to_numpy(float), index=pd.to_datetime(idx["date"]))
    ic = pd.Series(idx["close"].to_numpy(float), index=pd.to_datetime(idx["date"]))
    n = len(d)
    out = {}
    for h in H:
        f = np.full(n, np.nan)
        m = np.full(n, np.nan)
        if n > h:
            f[: n - h] = c[h:] / o[1: n - h + 1] - 1
            e_dates, x_dates = dates[1: n - h + 1], dates[h:]
            mo = io.reindex(e_dates).to_numpy()
            mc = ic.reindex(x_dates).to_numpy()
            m[: n - h] = mc / mo - 1
        out[h] = (f, m)
    return out


# ---------------------------------------------------------------------------
# 單檔股票
# ---------------------------------------------------------------------------
def random_reps(sc: WyckoffScanner, n_trades: int, stop_atr: float, reps: int, seed: int) -> np.ndarray:
    """隨機進場（相同停損距離與出場規則）；回傳每一輪的 [淨報酬和, 筆數, 勝場]。"""
    out = np.zeros((reps, 3))
    d = sc.d
    ok = d["atr"].notna().to_numpy() & (np.arange(len(d)) > 60) & (np.arange(len(d)) < len(d) - 10)
    valid = np.where(ok)[0]
    if n_trades == 0 or len(valid) < n_trades:
        return out
    rng = np.random.default_rng(seed)
    orig = sc.signals
    base = [s for s in orig if s["kind"] not in LONG_KINDS]
    for r in range(reps):
        picks = np.sort(rng.choice(valid, size=n_trades, replace=False))
        sc.signals = base + [dict(idx=int(i), date=d.at[i, "date"], kind="L_RANDOM", side="long",
                                  stop=float(d.at[i, "close"] - stop_atr * d.at[i, "atr"]), note="", page="",
                                  tr_id=-1) for i in picks]
        t = backtest(sc, long_kinds=("L_RANDOM",))
        if len(t):
            out[r] = [t["net_ret"].sum(), len(t), (t["net_ret"] > 0).sum()]
    sc.signals = orig
    return out


def run_stock(args) -> dict:
    code, group, market, data_dir, p, with_random = args
    df = load_df(os.path.join(data_dir, f"{code}.csv"))
    idx = load_index(data_dir, market)
    sc = WyckoffScanner(df, p).run()
    d = sc.d
    fr = forward_returns(d, idx)
    base_mean = {h: float(np.nanmean(fr[h][0])) for h in H}

    rows = []
    def add(kind, cat, i, direction):
        if i is None or i >= len(d):
            return
        r = dict(code=code, group=group, market=market, cat=cat, kind=kind, dir=direction,
                 date=d.at[i, "date"], period="in" if d.at[i, "date"] < pd.Timestamp(SPLIT) else "out")
        for h in H:
            f, m = fr[h][0][i], fr[h][1][i]
            r[f"fwd{h}"] = f
            r[f"exs{h}"] = f - base_mean[h] if np.isfinite(f) else np.nan
            r[f"exm{h}"] = f - m if (np.isfinite(f) and np.isfinite(m)) else np.nan
        rows.append(r)

    ev = sc.events_df()
    if len(ev):
        for k, c in ev[["kind", "confirm"]].drop_duplicates().itertuples(index=False):
            add(k, "event", int(c), event_dir(k))
    sig = sc.signals_df()
    if len(sig):
        for k, i in sig[["kind", "idx"]].drop_duplicates().itertuples(index=False):
            add(k, "signal", int(i), signal_dir(k))

    t = backtest(sc)
    t_wf = backtest(sc, weekly_filter=True)
    rs = relative_strength(df, idx)
    rs_map = dict(zip(pd.to_datetime(rs["date"]), rs["rs_strong"].astype(bool)))
    pos_map = {pd.Timestamp(x): k for k, x in enumerate(pd.to_datetime(d["date"]))}
    for frame, tag in ((t, "none"), (t_wf, "weekly")):
        if len(frame):
            frame.insert(0, "filter", tag)
            frame.insert(0, "market", market)
            frame.insert(0, "group", group)
            frame.insert(0, "code", code)
            # 訊號日＝進場日前一根
            sig_i = [max(pos_map[pd.Timestamp(x)] - 1, 0) for x in frame["entry_date"]]
            frame["signal_date"] = [d.at[k, "date"] for k in sig_i]
            frame["rs_strong"] = [bool(rs_map.get(pd.Timestamp(x), False)) for x in frame["signal_date"]]
            frame["period"] = np.where(pd.to_datetime(frame["entry_date"]) < pd.Timestamp(SPLIT), "in", "out")
    exposure = float(t["bars"].sum() / len(d)) if len(t) else 0.0

    rand = None
    if with_random and len(t):
        stop_atr = float(np.nanmedian(t["risk_atr"])) if t["risk_atr"].notna().any() else 1.5
        stop_atr = 1.5 if (not np.isfinite(stop_atr) or stop_atr <= 0) else min(stop_atr, 5.0)
        seed = int(code) if str(code).isdigit() else 7
        rand = random_reps(sc, len(t), stop_atr, RAND_REPS, seed)
    bh = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1)
    return dict(code=code, group=group, market=market, rows=rows, trades=t, trades_wf=t_wf, rand=rand,
                exposure=exposure, buy_hold=bh, bars=len(d), base_mean=base_mean, n_tr=len(sc.trs))


def run_variant(args) -> dict:
    """敏感度：只回傳交易彙總。"""
    code, group, market, data_dir, p, label, value = args
    df = load_df(os.path.join(data_dir, f"{code}.csv"))
    t = backtest(WyckoffScanner(df, p).run())
    return dict(param=label, value=value, trades=len(t), sum_net=float(t["net_ret"].sum()) if len(t) else 0.0,
                wins=int((t["net_ret"] > 0).sum()) if len(t) else 0)


# ---------------------------------------------------------------------------
# 統計
# ---------------------------------------------------------------------------
def cluster_boot(df: pd.DataFrame, col: str, B: int = 2000, seed: int = 0):
    """以股票為單位的自助法：回傳 (平均, 2.5%, 97.5%, 雙尾 p)。"""
    x = df[["code", col]].dropna()
    if x.empty:
        return (np.nan,) * 4
    g = x.groupby("code")[col].agg(["sum", "count"])
    mean = g["sum"].sum() / g["count"].sum()
    if len(g) < 5:
        return (mean, np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(g), size=(B, len(g)))
    m = g["sum"].to_numpy()[pick].sum(1) / g["count"].to_numpy()[pick].sum(1)
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return (mean, np.percentile(m, 2.5), np.percentile(m, 97.5), min(p, 1.0))


def verdict(direction: int, lo: float, hi: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "樣本太少"
    if direction == 0:
        return "—"
    if (direction > 0 and lo > 0) or (direction < 0 and hi < 0):
        return "✔ 符合預期（顯著）"
    if (direction > 0 and hi < 0) or (direction < 0 and lo > 0):
        return "✘ 與預期相反（顯著）"
    return "△ 不顯著"


def study_table(S: pd.DataFrame, cat: str) -> pd.DataFrame:
    rows = []
    for kind, g in S[S["cat"] == cat].groupby("kind"):
        direction = int(g["dir"].iloc[0])
        m20, lo, hi, p = cluster_boot(g, "exs20")
        mm20, mlo, mhi, _ = cluster_boot(g, "exm20")
        hit = (np.sign(g["exs20"].dropna()) == direction).mean() if direction else np.nan
        rows.append(dict(kind=kind, dir=direction, n=len(g), stocks=g["code"].nunique(),
                         exs5=g["exs5"].mean(), exs10=g["exs10"].mean(), exs20=m20, exs20_lo=lo, exs20_hi=hi,
                         exs20_p=p, exs60=g["exs60"].mean(), exm20=mm20, exm20_lo=mlo, exm20_hi=mhi,
                         dir_hit20=hit, fwd20=g["fwd20"].mean(),
                         exs20_in=g.loc[g["period"] == "in", "exs20"].mean(),
                         exs20_out=g.loc[g["period"] == "out", "exs20"].mean(),
                         verdict=verdict(direction, lo, hi), verdict_mkt=verdict(direction, mlo, mhi)))
    out = pd.DataFrame(rows)
    return out.sort_values(["dir", "n"], ascending=[False, False]) if len(out) else out


def trade_stats(t: pd.DataFrame) -> dict:
    if t is None or len(t) == 0:
        return dict(trades=0, win_rate=np.nan, avg_net=np.nan, med_net=np.nan, profit_factor=np.nan,
                    avg_bars=np.nan)
    w = t["net_ret"] > 0
    gl = -t.loc[~w, "net_ret"].sum()
    return dict(trades=len(t), win_rate=w.mean(), avg_net=t["net_ret"].mean(), med_net=t["net_ret"].median(),
                profit_factor=(t.loc[w, "net_ret"].sum() / gl) if gl > 0 else np.inf, avg_bars=t["bars"].mean())


def fmt(x, pct=False, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    if pct:
        return f"{x * 100:+.{nd}f}%"
    return f"{x:.{nd}f}"


def md_table(df: pd.DataFrame, cols: list, heads: list, pct_cols=(), nd=2) -> str:
    lines = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for r in df.itertuples(index=False):
        vals = []
        for c in cols:
            v = getattr(r, c)
            vals.append(fmt(v, pct=c in pct_cols, nd=nd) if not isinstance(v, str) else v)
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="威科夫訊號：台股樣本驗證")
    ap.add_argument("--data", default="data/tw_daily")
    ap.add_argument("--out", default="results_tw")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--no-random", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 檔（測試用）")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man = pd.read_csv(os.path.join(a.data, "manifest.csv"), dtype={"code": str})
    stocks = man[man["group"] != "index"].reset_index(drop=True)
    if a.limit:
        stocks = stocks.head(a.limit)
    p = Params()
    jobs = [(r.code, r.group, r.market, a.data, p, not a.no_random) for r in stocks.itertuples()]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        res = list(ex.map(run_stock, jobs, chunksize=2))

    S = pd.DataFrame([row for r in res for row in r["rows"]])
    S.to_csv(os.path.join(a.out, "study_detail.csv.gz"), index=False, encoding="utf-8-sig", compression="gzip")
    sig_tab, ev_tab = study_table(S, "signal"), study_table(S, "event")
    pd.concat([sig_tab.assign(cat="signal"), ev_tab.assign(cat="event")]).to_csv(
        os.path.join(a.out, "study_summary.csv"), index=False, encoding="utf-8-sig")

    T = pd.concat([r["trades"] for r in res if len(r["trades"])], ignore_index=True)
    TW = pd.concat([r["trades_wf"] for r in res if len(r["trades_wf"])], ignore_index=True)
    pd.concat([T, TW], ignore_index=True).to_csv(os.path.join(a.out, "trades.csv"), index=False, encoding="utf-8-sig")

    # 隨機基準：每一輪把所有股票加總後取平均
    rand = [r["rand"] for r in res if r["rand"] is not None]
    rb = None
    if rand:
        R = np.sum(rand, axis=0)                       # reps × [sum_net, n, wins]
        rb = pd.DataFrame(dict(rep=np.arange(len(R)), trades=R[:, 1], avg_net=R[:, 0] / np.maximum(R[:, 1], 1),
                               win_rate=R[:, 2] / np.maximum(R[:, 1], 1)))
        rb.to_csv(os.path.join(a.out, "random_baseline.csv"), index=False, encoding="utf-8-sig")

    # ---- 文字總表 ----
    L = ["# 威科夫訊號：台股樣本驗證結果（v1.3 第一輪）", ""]
    L += [f"- 樣本：{len(res)} 檔普通股（大型 {sum(r['group'] == 'large' for r in res)}、中型 "
          f"{sum(r['group'] == 'mid' for r in res)}、小型 {sum(r['group'] == 'small' for r in res)}），"
          f"K 線合計 {sum(r['bars'] for r in res):,} 根；樣本內＜{SPLIT}≦樣本外。",
          "- 價格未還原（除權息缺口仍在）、幾乎沒有下市股票（存活者偏差）。結果應視為初步。",
          f"- 交易成本：買 0.1425%、賣 0.4425%。隨機基準：每檔相同筆數、相同停損距離與出場規則，{RAND_REPS} 輪。",
          ""]
    heads = ["訊號", "預期", "次數", "檔數", "5日超額", "20日超額", "95% 區間", "60日超額", "相對大盤20日",
             "方向正確率", "樣本內20日", "樣本外20日", "判讀"]
    def tab(df):
        df = df.copy()
        df["ci"] = [f"{lo * 100:+.2f}% ~ {hi * 100:+.2f}%" if np.isfinite(lo) else "—"
                    for lo, hi in zip(df["exs20_lo"], df["exs20_hi"])]
        df["dir_s"] = df["dir"].map({1: "多", -1: "空", 0: "—"})
        return md_table(df, ["kind", "dir_s", "n", "stocks", "exs5", "exs20", "ci", "exs60", "exm20", "dir_hit20",
                             "exs20_in", "exs20_out", "verdict"], heads,
                        pct_cols=("exs5", "exs20", "exs60", "exm20", "dir_hit20", "exs20_in", "exs20_out"))
    L += ["## 1. 交易訊號的前瞻報酬（不含出場規則）", "",
          "超額＝訊號後 h 日報酬減去同一檔股票的平均 h 日報酬；方向正確率＝20 日超額與預期同向的比例。", "",
          tab(sig_tab), "", "## 2. 結構事件的前瞻報酬", "", tab(ev_tab), ""]

    # 回測
    rows = [dict(kind=k, **trade_stats(g)) for k, g in T.groupby("kind")] + [dict(kind="全部", **trade_stats(T))]
    bt = pd.DataFrame(rows)
    L += ["## 3. 回測（依原書進出場規則，一次一筆多單）", "",
          md_table(bt, ["kind", "trades", "win_rate", "avg_net", "med_net", "profit_factor", "avg_bars"],
                   ["訊號", "筆數", "勝率", "平均淨報酬", "中位數", "獲利因子", "平均持有K數"],
                   pct_cols=("win_rate", "avg_net", "med_net")), ""]
    if rb is not None:
        L += [f"**隨機進場基準**：平均淨報酬 {fmt(rb['avg_net'].mean(), True)}（5%～95%："
              f"{fmt(rb['avg_net'].quantile(0.05), True)} ～ {fmt(rb['avg_net'].quantile(0.95), True)}），"
              f"勝率 {fmt(rb['win_rate'].mean(), True)}；策略平均淨報酬 {fmt(T['net_ret'].mean(), True)}。"
              f"策略高於隨機的輪數比例：{(T['net_ret'].mean() > rb['avg_net']).mean() * 100:.0f}%。", ""]
    # 樣本內外、分組、週線、相對強弱
    def split_tab(col, labels):
        rr = []
        for k, g in T.groupby(col):
            rr.append(dict(key=labels.get(k, k), **trade_stats(g)))
        return md_table(pd.DataFrame(rr), ["key", "trades", "win_rate", "avg_net", "med_net", "profit_factor"],
                        ["分組", "筆數", "勝率", "平均淨報酬", "中位數", "獲利因子"],
                        pct_cols=("win_rate", "avg_net", "med_net"))
    L += ["## 4. 樣本內／樣本外", "", split_tab("period", {"in": f"樣本內（＜{SPLIT}）", "out": f"樣本外（≧{SPLIT}）"}), "",
          "## 5. 大型／中型／小型股", "", split_tab("group", {"large": "大型", "mid": "中型", "small": "小型"}), "",
          "## 6. 週線背景過濾〔p147、p225〕", "",
          md_table(pd.DataFrame([dict(key="不過濾", **trade_stats(T)), dict(key="過濾逆週線背景的多方訊號", **trade_stats(TW))]),
                   ["key", "trades", "win_rate", "avg_net", "med_net", "profit_factor"],
                   ["版本", "筆數", "勝率", "平均淨報酬", "中位數", "獲利因子"], pct_cols=("win_rate", "avg_net", "med_net")), "",
          "## 7. 相對強弱〔p253–258〕：訊號日是否為相對強勢股", "", split_tab("rs_strong", {True: "相對強勢", False: "非相對強勢"}), ""]
    # 出場原因
    er = T.groupby("reason")["net_ret"].agg(["count", "mean"]).reset_index().sort_values("count", ascending=False)
    er["share"] = er["count"] / er["count"].sum()
    L += ["## 8. 出場原因", "", md_table(er, ["reason", "count", "share", "mean"], ["原因", "筆數", "占比", "平均淨報酬"],
                                         pct_cols=("share", "mean")), ""]
    expo = np.mean([r["exposure"] for r in res])
    L += [f"- 平均持股時間占比（曝險）：{expo * 100:.1f}%；同期買進持有的中位數報酬："
          f"{np.median([r['buy_hold'] for r in res]) * 100:+.1f}%。", ""]

    # 敏感度
    if a.sensitivity:
        vjobs = [(r.code, r.group, r.market, a.data, p, "(基準)", "") for r in stocks.itertuples()]
        for name, vals in SENS_GRID.items():
            for v in vals:
                pv = replace(p, **{name: v})
                vjobs += [(r.code, r.group, r.market, a.data, pv, name, v) for r in stocks.itertuples()]
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            V = pd.DataFrame(list(ex.map(run_variant, vjobs, chunksize=4)))
        g = V.groupby(["param", "value"], sort=False)[["trades", "sum_net", "wins"]].sum().reset_index()
        g["avg_net"] = g["sum_net"] / g["trades"].clip(lower=1)
        g["win_rate"] = g["wins"] / g["trades"].clip(lower=1)
        g.to_csv(os.path.join(a.out, "sensitivity.csv"), index=False, encoding="utf-8-sig")
        g["value"] = g["value"].astype(str)
        L += ["## 9. 參數穩健性（單變數 ±20%）", "",
              md_table(g, ["param", "value", "trades", "win_rate", "avg_net"], ["參數", "數值", "筆數", "勝率", "平均淨報酬"],
                       pct_cols=("win_rate", "avg_net")), ""]
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
