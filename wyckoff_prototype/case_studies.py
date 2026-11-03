# -*- coding: utf-8 -*-
"""
報告第 11 章的台股案例：用真實日 K 重跑引擎，列出各案例期間的關鍵價位、結構事件與交易訊號。

用法：
    python3 case_studies.py --data data/tw_daily --out results_tw/cases.md

每個案例都用「整段歷史」掃描（狀態機逐根推進、無未來函數），再擷取案例期間的輸出，
因此表中的事件就是 StockExpert 在當時收盤後會看到的內容（事件 K 線與確認 K 線分開列出）。
"""
from __future__ import annotations

import argparse
import warnings

import pandas as pd

from wyckoff_engine import Params, WyckoffScanner, backtest, load_ohlcv_csv

warnings.filterwarnings("ignore")

CASES = [
    ("2330", "台積電", "2024-09-15", "2025-08-15", "派發→破冰→恐慌拋售→吸籌→上漲（完整週期）"),
    ("2330", "台積電", "2021-12-01", "2023-02-28", "2022 年空頭：兩次 SC 都沒有完成吸籌結構"),
    ("index_TAIEX", "加權指數", "2019-12-15", "2020-09-30", "2020 年 3 月崩跌與 V 型反轉"),
    ("index_TAIEX", "加權指數", "2024-05-15", "2024-11-30", "2024 年 7～8 月高檔急漲急跌"),
    ("index_TAIEX", "加權指數", "2025-02-15", "2025-08-31", "2025 年 4 月關稅衝擊"),
]

# 只列出與結構判讀有關的事件（通道類事件太多，另計次數）
KEEP = {"PS", "SC", "SA", "AR", "ST", "ST(量大)", "Spring", "Spring失敗", "Shakeout", "Test", "SOS", "SOS(波)", "JOC",
        "LPS", "BU", "SOS失效", "MarkupE", "PSY", "BC", "AR(自然回落)", "ST(派發)", "UT", "UTAD", "UT確認", "SOW",
        "SOW失效", "ICE", "LPSY", "MarkdownE", "NoDemandRally", "PB", "CORNER", "突破(量不足)"}


def load(data_dir: str, code: str) -> pd.DataFrame:
    return list(load_ohlcv_csv(f"{data_dir}/{code}.csv").values())[0]


def one_case(data_dir, code, name, a, b, title) -> list:
    df = load(data_dir, code)
    sc = WyckoffScanner(df, Params()).run()
    d = sc.d
    m = (d["date"] >= a) & (d["date"] <= b)
    w = d[m]
    hi, lo = w.loc[w["high"].idxmax()], w.loc[w["low"].idxmin()]
    L = [f"## {name}（{code}）{a[:7]}～{b[:7]}：{title}", "",
         f"- 期間最高 {hi['high']:,.2f}（{hi['date']:%Y-%m-%d}）、最低 {lo['low']:,.2f}（{lo['date']:%Y-%m-%d}）。", ""]
    chg = d["close"].pct_change()
    big = w.assign(chg=chg[m]).loc[lambda x: x["chg"].abs() >= 0.05]
    if len(big):
        L += ["| 大幅變動日 | 收盤 | 漲跌 | 相對量 | 相對振幅 | 收盤位置 |", "|---|---|---|---|---|---|"]
        L += [f"| {r.date:%Y-%m-%d} | {r.close:,.2f} | {r.chg * 100:+.1f}% | {r.rvol:.2f} | {r.rspread:.2f} | {r.close_pos:.2f} |"
              for r in big.itertuples()]
        L += [""]
    ev = sc.events_df()
    ev = ev[(ev["date"] >= a) & (ev["date"] <= b)]
    other = ev[~ev["kind"].isin(KEEP)]["kind"].value_counts()
    ev = ev[ev["kind"].isin(KEEP)]
    if len(ev):
        L += ["| 事件K線 | 確認K線 | 事件 | 價位 | 區間 | 引擎說明 |", "|---|---|---|---|---|---|"]
        for r in ev.itertuples():
            L.append(f"| {r.date:%Y-%m-%d} | {d.at[r.confirm, 'date']:%Y-%m-%d} | {r.kind} | {r.price:,.2f} | "
                     f"{r.tr_id if r.tr_id >= 0 else '—'} | {r.note} |")
    else:
        L += ["（期間內沒有偵測到區間結構事件）"]
    if len(other):
        L += ["", "其他（通道、波段、50% 原則）事件次數：" + "、".join(f"{k} {v}" for k, v in other.items())]
    sg = sc.signals_df()
    sg = sg[(sg["date"] >= a) & (sg["date"] <= b) & ~sg["kind"].str.startswith("W_")]
    if len(sg):
        L += ["", "| 訊號日 | 訊號 | 停損 |", "|---|---|---|"]
        L += [f"| {r.date:%Y-%m-%d} | {r.kind} | {r.stop:,.2f} |" if pd.notna(r.stop) else f"| {r.date:%Y-%m-%d} | {r.kind} | — |"
              for r in sg.itertuples()]
    t = backtest(sc)
    if len(t):
        t = t[(t["entry_date"] >= a) & (t["entry_date"] <= b)]
    if len(t):
        L += ["", "| 進場 | 出場 | 訊號 | 進場價 | 出場價 | 淨報酬 | 出場原因 |", "|---|---|---|---|---|---|---|"]
        L += [f"| {r.entry_date:%Y-%m-%d} | {r.exit_date:%Y-%m-%d} | {r.kind} | {r.entry:,.2f} | {r.exit:,.2f} | "
              f"{r.net_ret * 100:+.2f}% | {r.reason} |" for r in t.itertuples()]
    trs = sc.trs_df()
    if len(trs):
        trs = trs[(trs["start"] <= pd.Timestamp(b)) & ((trs["end"].astype(str) >= a) | (trs["end"].astype(str) == "-1"))]
        if len(trs):
            L += ["", "| 區間 | 前趨勢 | 起點 | 終點 | 支撐 | 阻力 | 最終分類 |", "|---|---|---|---|---|---|---|"]
            for r in trs.itertuples():
                sup = f"{r.support:,.2f}" if pd.notna(r.support) else "—"
                res = f"{r.resistance:,.2f}" if pd.notna(r.resistance) else "—"
                end = r.end.strftime("%Y-%m-%d") if hasattr(r.end, "strftime") else "進行中"
                L.append(f"| {r.tr_id} | {r.prior} | {r.start:%Y-%m-%d} | {end} | {sup} | {res} | {r.result} |")
    return L + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tw_daily")
    ap.add_argument("--out", default="results_tw/cases.md")
    a = ap.parse_args()
    L = ["# 第 11 章台股案例：引擎實際輸出", "",
         "> 價格未還原（除權息缺口仍在）。事件 K 線＝事件發生的那根；確認 K 線＝引擎能判定事件成立的那根（之後才可行動）。", ""]
    for c in CASES:
        L += one_case(a.data, *c)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
