# -*- coding: utf-8 -*-
"""
威科夫（Wyckoff）價量訊號引擎 —— StockExpert 參考原型
=====================================================

依據：孟洪濤《威科夫操盤法》（PDF 共 318 頁）。註解中的〔p.XX〕為 PDF 實體頁碼。

設計原則
--------
1. 原書規則優先：書中有明確可量化描述者，直接照書實作，例如
   - p57「降價中供應擴大」三條件（最高/最低價都低於昨日、收盤低於昨日、量 >= 昨日）
   - p194 Spring 須在 1~3 根 K 線內收回支撐；p154 UT 由 1~3 根 K 線完成
   - p95 SOS 可由「連續 2~3 根更高的最低價、收盤價、最高價 + 遞增成交量」構成
   - p143 50% 原則；p137/p140 SOT（突破幅度縮小）
2. 書中只有定性描述者（「天量」「超長陰線」「小蠟燭小成交量」），以相對量 rvol 與
   相對振幅 rspread 參數化。這些門檻屬於【延伸推論】，全部集中在 Params，
   應以實際台股資料重新校準。
3. 不偷看未來：每個事件都記錄 confirm（可被確認的那根 K 線）；回測一律在
   confirm 的下一根開盤成交。

主要介面
--------
    load_ohlcv_csv(path)        讀取台股/通用 CSV（支援民國年、千分位、中文欄名、還原權值）
    add_features(df)            基礎特徵：rvol、rspread、close_pos、影線比例、ATR、背景
    bar_signals(df)             原書單根/數根 K 線型態旗標
    zigzag(df) / waves(...)     波段轉折與波段累積成交量（p60-62 的「波段成交量」）
    WyckoffScanner(df).run()    交易區間（TR）、Phase A–E、事件、交易訊號
    backtest(...)               依原書進出場/停損規則的簡化回測（含台股交易成本）
    event_study(...)            事件後 N 日報酬統計
    snapshot(...)               最新 K 線的威科夫診斷卡（StockExpert 個股頁用）
    audit_trades(...)           以威科夫脈絡檢討使用者的買賣紀錄
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict, replace
from typing import Optional

import numpy as np
import pandas as pd

NaN = float("nan")


# ---------------------------------------------------------------------------
# 參數（所有門檻集中於此；標【原書】者直接來自書中，其餘為【延伸】可校準）
# ---------------------------------------------------------------------------
@dataclass
class Params:
    # 基礎統計窗
    n: int = 20                      # 均量、均幅回看根數【延伸】
    trend_n: int = 50                # 背景均線【延伸】
    slope_n: int = 10
    # 成交量分級（相對前 n 根均量）【延伸：把書中定性詞參數化】
    vol_climax: float = 2.0          # 天量／超高成交量（SC、BC）〔p65、p66〕
    vol_high: float = 1.5            # 明顯放量（SOS、SOW、垂直柱）
    vol_up: float = 1.2              # 量增
    vol_low: float = 0.8             # 縮量（測試、回測）
    vol_dry: float = 0.6             # 量極小（無供應／無需求）
    # 振幅分級（相對前 n 根平均高低價差；高低價差定義見〔p282〕）
    spread_wide: float = 1.5         # 超長／寬幅
    spread_big: float = 1.25
    spread_narrow: float = 0.8
    # 背景與高潮
    climax_lookback: int = 60        # 「新低／新高」判斷窗
    trend_lookback: int = 120
    min_decline: float = 0.15        # SC 前已下跌幅度（書：下跌趨勢持續一段時間之後〔p65〕）
    min_rise: float = 0.20           # BC 前已上漲幅度
    accel_atr: float = 4.0           # BC 前 10 根上漲 >= 4 ATR ≈「角度變陡、走向垂直」〔p97、p151〕
    allow_stopping_action_start: bool = True   # 無 SC/BC 時以停止行為起算 TR〔p127〕
    # 區間建立
    ar_max_bars: int = 25
    ar_min_atr: float = 2.0          # AR 幅度至少 2 ATR
    ar_confirm_atr: float = 1.0      # 自 AR 極值回撤 1 ATR 視為 AR 已確認
    st_zone: float = 0.40            # ST 須回到區間下（上）方 40% 範圍
    st_vol_ratio: float = 0.80       # ST 量 <= 高潮量 80% → 供應減弱〔p74：與 SC 相比都非常小〕
    st_max_bars: int = 80
    # Phase C
    pen_min_atr: float = 0.05        # 跌破/突破至少 0.05 ATR 才算穿越
    spring_recover_bars: int = 3     # 【原書】Spring/UT 須在 1~3 根內收回〔p66、p154、p194〕
    shakeout_depth_atr: float = 1.0  # 深度跌破 >= 1 ATR 視為震倉〔p90、p131、p137〕
    shakeout_recover_bars: int = 10  # 震倉允許 10 根內收回區間【延伸】
    test_window: int = 20            # 等待 Spring/震倉二次測試的窗口
    # Phase D
    lps_max_bars: int = 15           # SOS/JOC 後等待 LPS/BU 的窗口
    lpsy_max_bars: int = 15
    exit_range_atr: float = 1.0      # 收盤離開區間 1 ATR 且連續 3 根 → Phase E
    exit_range_bars: int = 3
    max_tr_bars: int = 300
    # 交易管理
    stop_buffer_atr: float = 0.1     # 停損多留 0.1 ATR（書：低一個點/兩個點〔p207、p215、p289〕）
    confirm_bars: int = 5            # 【原書】進場後 1~3 或 3~5 根內甄別〔p316〕
    max_hold: int = 120
    buy_cost: float = 0.001425       # 台股手續費（未折扣）
    sell_cost: float = 0.004425      # 手續費 0.1425% + 證交稅 0.3%
    # 通道線〔p38、p65 註8、p132、p142–147、p263–270〕
    channel_min_sep: int = 5         # 兩個波段低點至少相隔 5 根才畫通道【延伸】
    channel_min_width_atr: float = 1.0
    exit_on_demand_line_break: bool = True   # 【原書 p142、p146】帶量跌破上升趨勢線（需求線）→ 持股離場
    # 多時間框架：由日 K 合成週 K 作為大背景〔p31、p127、p146、p268–272〕
    use_weekly: bool = True
    weekly_filter: bool = False      # True：週線仍在下跌且尚未形成區間時，不採用日線多方訊號〔p147、p225〕
    # v1.2：趨勢回調進場（書中後記推薦的入門方法〔p316〕：牛市＋需要的回調＋停止行為扣扳機）
    pb_lookback: int = 40            # 近 40 根最高點視為這一波的高點【延伸】
    pb_min_leg_atr: float = 3.0      # 上漲腿至少 3 ATR 才算一波【延伸】
    pb_max_retr: float = 0.5         # 【原書 p143–146、p187】回調不超過上漲腿 50%
    pb_max_age: int = 25             # 高點之後最多 25 根仍視為回調【延伸】
    pb_zone_atr: float = 0.75        # 支撐區寬度【延伸】
    pb_ud_min: float = 1.1           # 近 20 日上漲日量／下跌日量【原書 p316 概念，門檻為延伸】
    pb_max_risk_atr: float = 3.5     # 進場到停損超過 3.5 ATR 則放棄【延伸】
    pb_cooldown: int = 8
    # v1.2：死角交易法〔p114、p233、p290〕
    corner_demand_min: float = 55.0
    corner_range_pos_max: float = 0.45
    corner_rvol_max: float = 0.8
    # v1.2：持有期間的出場規則（回測用）
    exit_on_stall: bool = True       # 【原書 p59–60】放量滯漲後次日量縮 → 清倉最後時機
    tighten_on_sot: bool = True      # 【原書 p141】SOT／通道式 SOT／波段放量滯漲／回調>50% → 縮緊停損
    exit_on_sot_retr: bool = True    # 【原書 p145–146】SOT 與回調>50% 先後出現 → 離場


# ---------------------------------------------------------------------------
# 資料讀取（台股 CSV 常見格式）
# ---------------------------------------------------------------------------
_COLS = {
    "date": ["date", "日期", "交易日期", "資料日期", "datetime", "time", "年月日"],
    "open": ["open", "開盤價", "开盘价", "開盤", "开盘", "開"],
    "high": ["high", "最高價", "最高价", "最高", "高"],
    "low": ["low", "最低價", "最低价", "最低", "低"],
    "close": ["close", "收盤價", "收盘价", "收盤", "收盘", "收"],
    "adj_close": ["adj close", "adj_close", "adjclose", "還原收盤價", "調整收盤價"],
    "volume": ["volume", "成交股數", "成交量", "成交張數", "成交量(張)", "成交量(股)", "vol"],
    "symbol": ["symbol", "代號", "股票代號", "證券代號", "stock_id", "ticker", "code"],
}


def _find_col(cols, names):
    low = {str(c).strip().lower(): c for c in cols}
    for nm in names:
        if nm.lower() in low:
            return low[nm.lower()]
    return None


def _to_num(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(",", "", regex=False).str.strip()
    s = s.replace({"--": np.nan, "---": np.nan, "X": np.nan, "": np.nan, "nan": np.nan})
    return pd.to_numeric(s, errors="coerce")


def _parse_date(s: pd.Series) -> pd.Series:
    def one(x):
        x = str(x).strip()
        m = re.match(r"^(\d{2,4})[/\-.](\d{1,2})[/\-.](\d{1,2})", x)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if y < 1911:          # 民國年
                y += 1911
            return pd.Timestamp(y, mo, d)
        m = re.match(r"^(\d{4})(\d{2})(\d{2})$", x)
        if m:
            return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        m = re.match(r"^(\d{3})(\d{2})(\d{2})$", x)      # 1130102 這種民國格式
        if m:
            return pd.Timestamp(int(m.group(1)) + 1911, int(m.group(2)), int(m.group(3)))
        return pd.to_datetime(x, errors="coerce")
    return s.map(one)


def load_ohlcv_csv(path: str, encoding: Optional[str] = None) -> dict:
    """讀取 CSV，回傳 {symbol: DataFrame[date, open, high, low, close, volume]}。

    - 支援中文欄名、民國年、千分位逗號、'--' 無交易。
    - 若有 Adj Close（還原收盤價），以 adj/close 比例還原 OHLC，避免除權息缺口被誤判為 SOW。
    - 若同一檔案含多檔股票（有代號欄），依代號分組。
    """
    raw = None
    for enc in ([encoding] if encoding else ["utf-8-sig", "utf-8", "cp950", "big5"]):
        try:
            raw = pd.read_csv(path, encoding=enc, dtype=str)
            break
        except Exception:
            continue
    if raw is None:
        raise ValueError(f"無法讀取 {path}")
    m = {k: _find_col(raw.columns, v) for k, v in _COLS.items()}
    need = ["date", "open", "high", "low", "close", "volume"]
    miss = [k for k in need if m[k] is None]
    if miss:
        raise ValueError(f"{path} 缺少欄位 {miss}；欄位為 {list(raw.columns)}")
    df = pd.DataFrame({
        "date": _parse_date(raw[m["date"]]),
        "open": _to_num(raw[m["open"]]),
        "high": _to_num(raw[m["high"]]),
        "low": _to_num(raw[m["low"]]),
        "close": _to_num(raw[m["close"]]),
        "volume": _to_num(raw[m["volume"]]),
    })
    if m["adj_close"] is not None:
        adj = _to_num(raw[m["adj_close"]])
        f = (adj / df["close"]).where(df["close"] > 0)
        for k in ("open", "high", "low", "close"):
            df[k] = df[k] * f
    df["symbol"] = raw[m["symbol"]].astype(str).str.strip() if m["symbol"] is not None else "UNKNOWN"
    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df[(df["high"] >= df["low"]) & (df["volume"].fillna(0) >= 0)]
    out = {}
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        out[sym] = g[["date", "open", "high", "low", "close", "volume"]]
    return out


# ---------------------------------------------------------------------------
# 特徵層
# ---------------------------------------------------------------------------
def add_features(df: pd.DataFrame, p: Params = Params()) -> pd.DataFrame:
    """基礎特徵。聰明錢三要素：價格、成交量、走勢速度〔p16〕。"""
    d = df.reset_index(drop=True).copy()
    o, h, l, c, v = (d[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    d["spread"] = h - l                                            # 高低價差〔p282〕
    d["atr"] = tr.rolling(p.n, min_periods=5).mean()
    d["vol_ma"] = v.shift(1).rolling(p.n, min_periods=5).mean()
    d["rvol"] = v / d["vol_ma"]                                    # 相對成交量（努力）
    sp_ma = d["spread"].shift(1).rolling(p.n, min_periods=5).mean()
    d["rspread"] = d["spread"] / sp_ma                             # 相對振幅（結果）
    rng = d["spread"].where(d["spread"] > 0)
    d["close_pos"] = ((c - l) / rng).fillna(0.5)                   # 收盤位置：0=最低 1=最高
    d["upper_wick"] = ((h - np.maximum(o, c)) / rng).fillna(0.0)
    d["lower_wick"] = ((np.minimum(o, c) - l) / rng).fillna(0.0)
    d["ret"] = c.pct_change()
    d["sma"] = c.rolling(p.trend_n, min_periods=p.trend_n // 2).mean()
    d["sma_slope"] = d["sma"] / d["sma"].shift(p.slope_n) - 1
    d["regime"] = np.select(
        [(c > d["sma"]) & (d["sma_slope"] > 0), (c < d["sma"]) & (d["sma_slope"] < 0)],
        ["up", "down"], default="range")                          # 背景〔p17：看盤第一步〕
    d["drawdown"] = c / c.rolling(p.trend_lookback, min_periods=20).max() - 1
    d["runup"] = c / c.rolling(p.trend_lookback, min_periods=20).min() - 1
    d["accel"] = (c - c.shift(10)) / d["atr"]                      # 走勢速度（ATR 單位）
    d["limit_move"] = d["ret"].abs() >= 0.095                      # 台股漲跌停（±10%）
    d["evr"] = d["rvol"] / d["rspread"].clip(lower=0.2)            # 努力/結果比〔p24、p58〕
    return d


def bar_signals(d: pd.DataFrame, p: Params = Params()) -> pd.DataFrame:
    """原書單根/數根 K 線型態。每欄都是布林值。"""
    o, h, l, c, v = (d[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    s = pd.DataFrame(index=d.index)
    # 【原書 p57】降價中供應擴大：最低、最高皆低於昨日；收盤低於昨日；量 >= 昨日
    s["supply_exp_p57"] = (l < l.shift(1)) & (h < h.shift(1)) & (c < c.shift(1)) & (v >= v.shift(1))
    # 【延伸：p57 的鏡像】價升量增的需求擴大
    s["demand_exp"] = (h > h.shift(1)) & (l > l.shift(1)) & (c > c.shift(1)) & (v >= v.shift(1))
    # 【原書 p95、p115】連續三根「三個更高」+ 遞增量 = 以上升波完成的 SOS
    de = s["demand_exp"].fillna(False)
    s["demand_wave3"] = de & de.shift(1, fill_value=False) & de.shift(2, fill_value=False)
    falling, rising = c < c.shift(3), c > c.shift(3)
    # 【原書 p78】第一種停止行為：下跌中長下影 + 放量；上升中長上影 + 放量
    s["stop1_down"] = falling & (d["lower_wick"] >= 0.4) & (d["rvol"] >= p.vol_up)
    s["stop1_up"] = rising & (d["upper_wick"] >= 0.4) & (d["rvol"] >= p.vol_up)
    # 【原書 p80、p239】第二種停止行為：蠟燭大幅縮小但成交量增加（努力沒有結果）
    enr = (v >= v.shift(1)) & (d["spread"] <= 0.6 * d["spread"].shift(1)) & (d["rvol"] >= 1.0)
    s["stop2_down"] = enr & falling      # 放量滯跌（偏多）〔p46、p66 註13〕
    s["stop2_up"] = enr & rising         # 放量滯漲（偏空）〔p58〕
    # 【原書 p31、p37、p66】垂直需求柱 / 垂直供應柱（VSB）
    s["vdb"] = (c > o) & (d["rspread"] >= p.spread_wide) & (d["rvol"] >= p.vol_high) & (d["close_pos"] >= 0.6)
    s["vsb"] = (c < o) & (d["rspread"] >= p.spread_wide) & (d["rvol"] >= p.vol_high) & (d["close_pos"] <= 0.4)
    # 【延伸：p229-235「無需求反彈」、p36「低量回調」的量化】
    s["no_demand"] = (c > c.shift(1)) & (d["rspread"] <= p.spread_narrow) & (d["rvol"] <= p.vol_low) & (v < v.shift(1))
    s["no_supply"] = (c < c.shift(1)) & (d["rspread"] <= p.spread_narrow) & (d["rvol"] <= p.vol_low) & (v < v.shift(1))
    # 【原書 p298-299】OKR：陰線包住前一根（外包反轉）
    s["okr_bear"] = (h >= h.shift(1)) & (l <= l.shift(1)) & (c < o) & (c < c.shift(1)) & (d["rvol"] >= p.vol_up)
    s["okr_bull"] = (h >= h.shift(1)) & (l <= l.shift(1)) & (c > o) & (c > c.shift(1)) & (d["rvol"] >= p.vol_up)
    # 放量陰線 / 放量陽線（進場後出場判斷用〔p83、p304〕）
    s["supply_bar"] = (c < o) & (d["rvol"] >= 1.3) & (d["rspread"] >= 1.2) & (d["close_pos"] <= 0.35)
    s["demand_bar"] = (c > o) & (d["rvol"] >= 1.3) & (d["rspread"] >= 1.2) & (d["close_pos"] >= 0.65)
    # 台股：漲跌停鎖死時量縮不代表無需求/無供應
    s.loc[d["limit_move"].fillna(False), ["no_demand", "no_supply"]] = False
    return s.fillna(False).astype(bool)


def wyckoff_indicators(d: pd.DataFrame, s: Optional[pd.DataFrame] = None, n: int = 20, box: int = 60) -> pd.DataFrame:
    """連續型指標（選股器/排序用）。除 ud_vol_ratio、evr 有原書依據外，demand_score 權重屬【延伸】。

    ud_vol_ratio  近 n 根上漲日總量 / 下跌日總量（>1＝上坡量大於下坡量＝需求占上風〔p81、p316〕）
    evr           努力/結果比：相對量 ÷ 相對振幅（高＝放量滯漲/滯跌＝停止或吸收〔p24、p58〕）
    absorption    高量窄幅（rvol>=1.2 且 rspread<=0.8）〔p24、p80〕
    dry_up        近 5 根平均相對量（<0.7＝量能枯竭，測試/回測的理想狀態〔p82、p100〕）
    range_pos     收盤在近 box 根高低區間的位置（0=底、1=頂；判斷左手邊/右手邊與支撐阻力）
    contraction   近 10 根平均振幅 / 近 box 根平均振幅（<0.6＝窄幅、可能形成死角〔p65 註4、p114〕）
    dead_corner   高點遞降＋低點遞升＋振幅收斂＝死角〔p65、p114、p272〕
    demand_score  0~100 綜合需求分數（50 為中性）
    """
    s = s if s is not None else bar_signals(d)
    c, v = d["close"], d["volume"]
    up, dn = c > c.shift(1), c < c.shift(1)
    out = pd.DataFrame(index=d.index)
    upv = v.where(up, 0.0).rolling(n, min_periods=5).sum()
    dnv = v.where(dn, 0.0).rolling(n, min_periods=5).sum()
    out["ud_vol_ratio"] = upv / dnv.replace(0, np.nan)
    out["evr"] = d["evr"]
    out["absorption"] = (d["rvol"] >= 1.2) & (d["rspread"] <= 0.8)
    out["dry_up"] = d["rvol"].rolling(5, min_periods=3).mean()
    lo, hi = d["low"].rolling(box, min_periods=20).min(), d["high"].rolling(box, min_periods=20).max()
    out["range_pos"] = (c - lo) / (hi - lo).replace(0, np.nan)
    out["contraction"] = d["spread"].rolling(10).mean() / d["spread"].rolling(box, min_periods=20).mean()
    h5, l5 = d["high"].rolling(5).max(), d["low"].rolling(5).min()
    out["dead_corner"] = (out["contraction"] < 0.6) & (h5 < h5.shift(5)) & (l5 > l5.shift(5))
    sup10 = (s["vsb"] | s["supply_bar"]).rolling(10, min_periods=1).sum()
    dem10 = (s["vdb"] | s["demand_bar"]).rolling(10, min_periods=1).sum()
    nos10 = s["no_supply"].rolling(10, min_periods=1).sum()
    nod10 = s["no_demand"].rolling(10, min_periods=1).sum()
    score = (50 + 15 * np.clip(np.log2(out["ud_vol_ratio"].fillna(1.0).clip(lower=0.01)), -2, 2)
             + 20 * (d["close_pos"].rolling(10, min_periods=3).mean() - 0.5)
             + 4 * (dem10 - sup10) + 2 * (nos10 - nod10))
    out["demand_score"] = np.clip(score, 0, 100)
    return out


# ---------------------------------------------------------------------------
# 波段（ZigZag）與波段成交量
# ---------------------------------------------------------------------------
def zigzag(d: pd.DataFrame, mult: float = 2.0) -> list:
    """ATR 門檻的轉折點。每個轉折點含 confirm（被確認的 K 線索引），避免未來函數。"""
    h, l = d["high"].to_numpy(float), d["low"].to_numpy(float)
    atr = d["atr"].bfill().ffill().to_numpy(float)
    n = len(h)
    piv, up, ext = [], None, 0
    lo_i = hi_i = 0
    for i in range(1, n):
        thr = mult * atr[i]
        if up is None:
            if l[i] < l[lo_i]:
                lo_i = i
            if h[i] > h[hi_i]:
                hi_i = i
            if lo_i < i and h[i] - l[lo_i] >= thr:
                piv.append(dict(idx=lo_i, price=l[lo_i], kind="L", confirm=i)); up, ext = True, i
            elif hi_i < i and h[hi_i] - l[i] >= thr:
                piv.append(dict(idx=hi_i, price=h[hi_i], kind="H", confirm=i)); up, ext = False, i
            continue
        if up:
            if h[i] >= h[ext]:
                ext = i
            elif h[ext] - l[i] >= thr:
                piv.append(dict(idx=ext, price=h[ext], kind="H", confirm=i)); up, ext = False, i
        else:
            if l[i] <= l[ext]:
                ext = i
            elif h[i] - l[ext] >= thr:
                piv.append(dict(idx=ext, price=l[ext], kind="L", confirm=i)); up, ext = True, i
    return piv


def waves(d: pd.DataFrame, piv: list) -> pd.DataFrame:
    """波段累積成交量〔p60-62、p187、p222〕：比較上漲波與下跌波的「努力」。"""
    v = d["volume"].to_numpy(float)
    rows = []
    for a, b in zip(piv[:-1], piv[1:]):
        s, e = a["idx"], b["idx"]
        rows.append(dict(start=s, end=e, dir="up" if b["kind"] == "H" else "down",
                         p0=a["price"], p1=b["price"], change=b["price"] / a["price"] - 1,
                         bars=e - s, wave_vol=v[s + 1:e + 1].sum(),
                         vol_per_bar=v[s + 1:e + 1].mean() if e > s else NaN, confirm=b["confirm"]))
    return pd.DataFrame(rows)


def pivot_warnings(d: pd.DataFrame, piv: list) -> list:
    """SOT（突破幅度遞減）〔p137 註2、p140-142〕與 50% 原則〔p143-147〕。
    只在對應的背景下記錄：上升背景看 SOT_up 與回調 > 50%；下降背景看 SOT_down 與反彈 < 50%。"""
    out = []
    reg = d["regime"].to_numpy()
    highs = [x for x in piv if x["kind"] == "H"]
    lows = [x for x in piv if x["kind"] == "L"]
    for a, b, c in zip(highs, highs[1:], highs[2:]):
        i1, i2 = b["price"] - a["price"], c["price"] - b["price"]
        if i1 > 0 and i2 > 0 and i2 < i1 and reg[c["confirm"]] == "up":
            out.append(dict(idx=c["idx"], confirm=c["confirm"], kind="SOT_up",
                            note=f"上升波突破幅度遞減 {i1:.2f}→{i2:.2f}（需求減弱，縮緊停損）", page="p140-142"))
    for a, b, c in zip(lows, lows[1:], lows[2:]):
        i1, i2 = a["price"] - b["price"], b["price"] - c["price"]
        if i1 > 0 and i2 > 0 and i2 < i1 and reg[c["confirm"]] == "down":
            out.append(dict(idx=c["idx"], confirm=c["confirm"], kind="SOT_down",
                            note=f"下跌波推進幅度遞減 {i1:.2f}→{i2:.2f}（供應減弱，待吸籌驗證）", page="p142"))
    for a, b, c in zip(piv, piv[1:], piv[2:]):
        # 上升背景：上升波之後的回調超過 50% → 警訊〔p143-146〕
        if a["kind"] == "L" and b["kind"] == "H" and c["kind"] == "L" and reg[c["confirm"]] == "up":
            up = b["price"] - a["price"]
            r = (b["price"] - c["price"]) / up if up > 0 else 0
            if r > 0.5:
                out.append(dict(idx=c["idx"], confirm=c["confirm"], kind="RETR_GT50",
                                note=f"回調 {r:.0%} 超過前一上升波 50%（供應擴大警訊）", page="p143-147"))
        # 下降背景：反彈未超過前一下跌波 50% → 仍是弱勢〔p147、p231、p299〕
        if a["kind"] == "H" and b["kind"] == "L" and c["kind"] == "H" and reg[c["confirm"]] == "down":
            dn = a["price"] - b["price"]
            r = (c["price"] - b["price"]) / dn if dn > 0 else 1
            if r < 0.5:
                out.append(dict(idx=c["idx"], confirm=c["confirm"], kind="RALLY_LT50",
                                note=f"反彈僅 {r:.0%}，未過前一下跌波 50%（熊市背景未變）", page="p147,p299"))
    return out


def wave_events(d: pd.DataFrame, piv: list) -> list:
    """波段層級的努力與結果〔p60–62、p187、p222〕（只用已確認的轉折點）。

    WAVE_STALL_UP   上漲波累積量 >= 前 2~3 個上漲波的平均，高點卻沒有超過前一個上漲波高點
                    ＝放量滯漲，原書稱「這是個警告」（CM 出貨）〔p60〕；只在上升背景記錄。
    WAVE_STALL_DOWN 下跌波累積量 >= 前 2~3 個下跌波的平均，低點卻沒有跌破前一個下跌波低點
                    ＝放量滯跌（需求在吸收供應）〔p60、p222〕；只在下降或區間背景記錄。
    """
    w = waves(d, piv)
    out = []
    if w.empty:
        return out
    reg = d["regime"].to_numpy()
    ups, dns = [], []
    for r in w.itertuples():
        same = ups if r.dir == "up" else dns
        if len(same) >= 2:
            avg_v = float(np.mean([x.wave_vol for x in same[-3:]]))
            if avg_v > 0 and r.wave_vol >= avg_v:
                ratio = r.wave_vol / avg_v
                if r.dir == "up" and r.p1 <= same[-1].p1 and reg[r.confirm] == "up":
                    out.append(dict(idx=int(r.end), confirm=int(r.confirm), kind="WAVE_STALL_UP", price=float(r.p1),
                                    note=f"上漲波累積量為前幾波的 {ratio:.1f} 倍，卻沒有創新高：放量滯漲（CM 出貨的警告）",
                                    page="p60-62,p187"))
                elif r.dir == "down" and r.p1 >= same[-1].p1 and reg[r.confirm] in ("down", "range"):
                    out.append(dict(idx=int(r.end), confirm=int(r.confirm), kind="WAVE_STALL_DOWN", price=float(r.p1),
                                    note=f"下跌波累積量為前幾波的 {ratio:.1f} 倍，卻沒有創新低：放量滯跌（需求在吸收供應）",
                                    page="p60-62,p222"))
        same.append(r)
    return out


# ---------------------------------------------------------------------------
# 通道線（超買線／超賣線／供應線／需求線）
# ---------------------------------------------------------------------------
def channels(d: pd.DataFrame, piv: list, p: Params = Params()) -> dict:
    """以「已確認」的波段低點畫通道（無未來函數）。

    上升通道：需求線（支撐線、上升趨勢線）＝連接兩個墊高的波段低點；
              超買線＝通過兩低點之間最高點的平行線（通道上軌）〔p38、p142、p269–270〕。
    下降通道：超賣線＝連接兩個連續、下降的波段低點〔p263〕；
              供應線＝通過兩低點之間最高點的平行線（通道上軌）〔p65 註8、p132〕。
    中線＝上下軌平均〔p132、p143〕。

    事件（每個都有 idx 與 confirm）：
      OVERSOLD            價格回到或跌破超賣線〔p65 註2、p263–264〕
      SUPPLY_LINE_BREAK   收盤突破供應線；放量（rvol>=1.3、rspread>=1.0）才算強〔p65 註8、p112、p132〕
      OVERBOUGHT          觸及超買線，觀察供應〔p38、p98、p235〕
      DEMAND_LINE_BREAK   收盤跌破需求線；放量（rvol>=1.2）才算強＝持股離場〔p142、p146、p270〕
      SOT_CH_UP           回調到需求線後，反彈到不了超買線（只到中線以下更弱）〔p142–143、p186〕
      SOT_CH_DOWN         下跌到不了超賣線（供應減弱）〔p132、p142〕
    """
    n = len(d)
    h, l, c = (d[k].to_numpy(float) for k in ("high", "low", "close"))
    atr = d["atr"].bfill().ffill().to_numpy(float)
    rvol = d["rvol"].fillna(1.0).to_numpy(float)
    rsp = d["rspread"].fillna(1.0).to_numpy(float)
    ch_dir = np.array([""] * n, dtype=object)
    lower, upper = np.full(n, NaN), np.full(n, NaN)
    events, fired = [], {}
    pv = sorted(piv, key=lambda x: (x["confirm"], x["idx"]))
    lows = []
    cur, cid, k = None, 0, 0

    def line(ch, t):
        return ch["y1"] + ch["slope"] * (t - ch["t1"])

    def build():
        if len(lows) < 2:
            return None
        a, b = lows[-2], lows[-1]
        if b["idx"] - a["idx"] < p.channel_min_sep:
            return None
        slope = (b["price"] - a["price"]) / (b["idx"] - a["idx"])
        off = max(h[t] - (a["price"] + slope * (t - a["idx"])) for t in range(a["idx"], b["idx"] + 1))
        ab = atr[b["idx"]]
        if not np.isfinite(ab) or off < p.channel_min_width_atr * ab:
            return None
        if b["price"] > a["price"] + 0.25 * ab:
            dirn = "up"
        elif b["price"] < a["price"] - 0.25 * ab:
            dirn = "down"
        else:
            return None
        return dict(dir=dirn, t1=a["idx"], y1=a["price"], slope=slope, off=off, start=b["confirm"])

    def emit(t, idx, kind, price, note, page, strong=False):
        events.append(dict(idx=int(idx), confirm=int(t), kind=kind, price=float(price),
                           note=note, page=page, strong=bool(strong), cid=cid))

    def once(t, kind, gap=None):
        last = fired.get((cid, kind))
        if last is not None and (gap is None or t - last <= gap):
            return False
        fired[(cid, kind)] = t
        return True

    for t in range(n):
        while k < len(pv) and pv[k]["confirm"] <= t:
            q = pv[k]
            k += 1
            if cur is not None and q["idx"] >= cur["t1"]:          # 用「舊通道」檢查通道式 SOT
                lo_q = line(cur, q["idx"])
                up_q, mid_q = lo_q + cur["off"], lo_q + cur["off"] / 2
                if cur["dir"] == "up" and q["kind"] == "H" and lows:
                    pl = lows[-1]
                    deep = pl["price"] <= line(cur, pl["idx"]) + 0.25 * cur["off"]
                    if deep and q["price"] < up_q - 0.25 * cur["off"]:
                        lvl = "只到中線以下" if q["price"] < mid_q else "未達超買線"
                        emit(t, q["idx"], "SOT_CH_UP", q["price"],
                             f"回調到需求線後的反彈{lvl}：突破幅度縮小（需求減弱，縮緊停損）", "p142-143,p186")
                if cur["dir"] == "down" and q["kind"] == "L" and q["price"] > lo_q + 0.25 * cur["off"]:
                    emit(t, q["idx"], "SOT_CH_DOWN", q["price"], "下跌未能到達超賣線：供應減弱，待吸籌過程驗證", "p132,p142")
            if q["kind"] == "L":
                lows.append(q)
                nb = build()
                if nb is not None:
                    cid += 1
                    cur = nb
        if cur is None:
            continue
        lo = line(cur, t)
        up = lo + cur["off"]
        ch_dir[t], lower[t], upper[t] = cur["dir"], lo, up
        if t < cur["start"]:
            continue
        if cur["dir"] == "down":
            if l[t] <= lo and once(t, "OVERSOLD", 5):
                emit(t, t, "OVERSOLD", l[t], "價格回到或跌破超賣線：超賣，之後至少會有反彈", "p65,p263-264")
            if c[t] > up + 0.1 * atr[t] and once(t, "SUPPLY_LINE_BREAK"):
                strong = rvol[t] >= 1.3 and rsp[t] >= 1.0
                emit(t, t, "SUPPLY_LINE_BREAK", c[t],
                     "放量突破供應線：需求吸收了供應線上的賣單，可能反轉（仍須看供應是否枯竭）" if strong
                     else "突破供應線但量能不足，需觀察後續", "p65,p112,p132", strong)
        else:
            if h[t] >= up and once(t, "OVERBOUGHT", 5):
                emit(t, t, "OVERBOUGHT", h[t], "觸及超買線：觀察是否出現供應（放量上影或放量回落）", "p38,p98,p235")
            if c[t] < lo - 0.1 * atr[t] and once(t, "DEMAND_LINE_BREAK"):
                strong = rvol[t] >= p.vol_up
                emit(t, t, "DEMAND_LINE_BREAK", c[t],
                     "放量跌破需求線（上升趨勢線）：供應超過需求，持股離場" if strong
                     else "跌破需求線但未放量，要看跌破過程與後續跟隨", "p142,p146,p270", strong)
        # 收盤明確穿出通道 1 ATR：通道失效，等新的波段低點重畫【延伸】
        if (cur["dir"] == "up" and c[t] < lo - atr[t]) or (cur["dir"] == "down" and c[t] > up + atr[t]):
            cur = None
    return dict(dir=ch_dir, lower=lower, upper=upper, mid=(lower + upper) / 2, events=events)


# ---------------------------------------------------------------------------
# 多時間框架：週線背景
# ---------------------------------------------------------------------------
def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """日 K → 週 K（週五為週別）。date 取該週最後一個交易日＝這根週 K 資料完整的時點。"""
    d = df[["date", "open", "high", "low", "close", "volume"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    g = d.groupby(d["date"].dt.to_period("W-FRI"), sort=True)
    w = pd.DataFrame({"date": g["date"].max(), "open": g["open"].first(), "high": g["high"].max(),
                      "low": g["low"].min(), "close": g["close"].last(), "volume": g["volume"].sum()})
    return w.reset_index(drop=True)


def weekly_params(base: Params = Params()) -> Params:
    """週線用參數【參數】：均線約 30 週、高潮判斷窗半年、AR/ST 等待窗依週數縮短；
    週量為 5 日合計、較日量平滑，高潮與放量門檻略降。"""
    return replace(base, n=20, trend_n=30, slope_n=4, climax_lookback=26, trend_lookback=52,
                   vol_climax=1.6, vol_high=1.3, spread_big=1.2,
                   min_decline=0.20, min_rise=0.25, ar_max_bars=12, st_max_bars=40, test_window=10,
                   lps_max_bars=8, lpsy_max_bars=8, max_tr_bars=150, confirm_bars=3, max_hold=52,
                   use_weekly=False, weekly_filter=False)


def weekly_context(df: pd.DataFrame, p: Params = Params()):
    """在週線上跑同一套狀態機，並以「已完成的週」對應回每根日 K（merge_asof 向後對齊，無未來函數）。

    wk_ok_long＝False 代表：週線仍是下降背景，且尚未出現由高潮/二次測試構成的區間（Phase 空白、A 或 E）。
    依原書「熊市中不能隨意抄底」〔p225〕、「進場前先看市場背景」〔p147〕，此時日線多方訊號應降級或略過。
    """
    w = to_weekly(df)
    n = len(df)
    empty = pd.DataFrame({"wk_regime": [""] * n, "wk_phase": [""] * n, "wk_bias": [0.0] * n, "wk_ok_long": [True] * n})
    if len(w) < 10:
        return empty, None
    wsc = WyckoffScanner(w, weekly_params(p)).run()
    wb = wsc.bars_df()[["date", "regime", "phase", "bias"]].rename(
        columns={"regime": "wk_regime", "phase": "wk_phase", "bias": "wk_bias"})
    wb["date"] = pd.to_datetime(wb["date"])
    wb["wk_n"] = np.arange(1, len(wb) + 1)                      # 截至該週已累積的週 K 根數
    left = pd.DataFrame({"date": pd.to_datetime(df["date"]).to_numpy(), "_i": np.arange(n)})
    m = pd.merge_asof(left.sort_values("date"), wb.sort_values("date"), on="date", direction="backward")
    m = m.sort_values("_i").reset_index(drop=True)
    # 當時累積不足 40 週：週線背景視為「未知」（逐日判斷，不受整份資料長度影響）
    unknown = m["wk_n"].isna() | (m["wk_n"] < 40)
    m["wk_regime"] = m["wk_regime"].where(~unknown, "").fillna("")
    m["wk_phase"] = m["wk_phase"].where(~unknown, "").fillna("")
    m["wk_bias"] = m["wk_bias"].where(~unknown, 0.0).fillna(0.0)
    m["wk_ok_long"] = ~((m["wk_regime"] == "down") & (m["wk_phase"].isin(["", "A", "E"])))
    return m[["wk_regime", "wk_phase", "wk_bias", "wk_ok_long"]], wsc


# ---------------------------------------------------------------------------
# 點數圖（Point & Figure）與因果目標【延伸：原書僅在 p11 提到點數圖，未教授數格】
# ---------------------------------------------------------------------------
def pnf_columns(prices, box: float, reversal: int = 3) -> list:
    """收盤價點數圖。回傳欄位清單：dir（X 上漲欄／O 下跌欄）、top、bot、start、end。"""
    prices = np.asarray(prices, float)
    cols, cur = [], None
    if len(prices) == 0 or not np.isfinite(box) or box <= 0:
        return cols
    p0 = prices[0]
    for i, x in enumerate(prices):
        if cur is None:
            if x >= p0 + box:
                cur = dict(dir="X", top=p0 + box * np.floor((x - p0) / box), bot=p0, start=0, end=i)
            elif x <= p0 - box:
                cur = dict(dir="O", top=p0, bot=p0 - box * np.floor((p0 - x) / box), start=0, end=i)
            continue
        if cur["dir"] == "X":
            if x >= cur["top"] + box:
                cur["top"] += box * np.floor((x - cur["top"]) / box)
                cur["end"] = i
            elif x <= cur["top"] - reversal * box:
                cols.append(cur)
                cur = dict(dir="O", top=cur["top"] - box, bot=cur["top"] - box * np.floor((cur["top"] - x) / box), start=i, end=i)
        else:
            if x <= cur["bot"] - box:
                cur["bot"] -= box * np.floor((cur["bot"] - x) / box)
                cur["end"] = i
            elif x >= cur["bot"] + reversal * box:
                cols.append(cur)
                cur = dict(dir="X", top=cur["bot"] + box * np.floor((x - cur["bot"]) / box), bot=cur["bot"] + box, start=i, end=i)
    if cur is not None:
        cols.append(cur)
    return cols


def cause_target(sc, tr=None, reversal: int = 3) -> Optional[dict]:
    """【延伸】點數圖橫向數格：交易區間內的欄數 × 格值 × 反轉格數 ≈「因」可推動的「果」。

    格值取區間內 ATR 中位數的一半；多方目標自區間下緣（保守）或中線起算，空方目標自區間上緣或中線起算。
    原書沒有教授這個方法，結果僅供參考，必須以台股資料回測後才可使用。
    """
    tr = tr or sc.tr or (sc.trs[-1] if sc.trs else None)
    if tr is None or not np.isfinite(tr.support) or not np.isfinite(tr.resistance):
        return None
    d = sc.d
    s0 = tr.climax_idx
    e0 = tr.end if tr.end >= 0 else len(d) - 1
    seg = d["close"].to_numpy(float)[s0:e0 + 1]
    if len(seg) < 10:
        return None
    box = float(np.nanmedian(d["atr"].to_numpy(float)[s0:e0 + 1])) / 2
    cols = pnf_columns(seg, box, reversal)
    move = len(cols) * box * reversal
    mid = (tr.support + tr.resistance) / 2
    return dict(tr_id=tr.id, columns=len(cols), box=round(box, 4), reversal=reversal,
                bull_from_support=round(tr.support + move, 2), bull_from_mid=round(mid + move, 2),
                bear_from_resistance=round(tr.resistance - move, 2), bear_from_mid=round(mid - move, 2),
                note="【延伸】點數圖橫向數格估算（欄數×格值×反轉格數）；原書未教授，僅供參考，需回測驗證")


# ---------------------------------------------------------------------------
# 交易區間（TR）掃描器：事件、Phase、訊號
# ---------------------------------------------------------------------------
@dataclass
class TR:
    id: int
    prior: str                 # 'down'（之後為吸籌或再派發）或 'up'（之後為派發或再吸籌）
    climax_kind: str           # 'SC' / 'BC' / 'SA'（停止行為代替高潮〔p127〕）
    climax_idx: int
    climax_price: float        # SC 最低價或 BC 最高價
    climax_vol: float
    climax_spread: float
    start: int
    state: str = "await_ar"    # await_ar → await_st → B（之後 C/D 以事件標記）→ done
    phase: str = "A"
    ar_idx: int = -1
    ar_price: float = NaN
    st_idx: int = -1
    support: float = NaN
    resistance: float = NaN
    st_ep: Optional[dict] = None
    pen_dn: Optional[dict] = None
    pen_up: Optional[dict] = None
    last_test_low: float = NaN
    bias: float = 0.0
    events: list = field(default_factory=list)
    result: str = ""
    end: int = -1
    exit_cnt: int = 0
    last_sos: int = -99
    last_sow: int = -99


EVENT_PAGES = {
    "PS": "p72-73", "SC": "p65,p73-74", "AR": "p72,p113", "ST": "p74-75", "ST(量大)": "p74,p77",
    "SA": "p78-80,p127", "PSY": "p149", "BC": "p66-67,p150-151", "AR(自然回落)": "p152,p160",
    "ST(派發)": "p152-153,p161", "Spring": "p66,p192-197", "Shakeout": "p65,p129-131",
    "Test": "p90,p136-137", "Spring失敗": "p197,p203", "SOS": "p66,p93-98", "JOC": "p66,p98-111",
    "LPS": "p115,p133", "BU": "p106,p111", "SOS失效": "p93,p98", "UT": "p66,p154",
    "UTAD": "p182-184", "UT確認": "p154", "SOW": "p66,p155-156", "ICE": "p65,p156,p160",
    "LPSY": "p152,p156", "SOW失效": "p161", "MarkupE": "p92,p98", "MarkdownE": "p148,p156",
    "OVERSOLD": "p65,p263-264", "SUPPLY_LINE_BREAK": "p65,p112,p132", "OVERBOUGHT": "p38,p98,p235",
    "DEMAND_LINE_BREAK": "p142,p146,p270", "SOT_CH_UP": "p142-143,p186", "SOT_CH_DOWN": "p132,p142",
    "PB": "p316,p31,p39-40,p239-244,p248-251", "CORNER": "p65,p114,p233,p290",
    "NoDemandRally": "p229-231,p253,p272-273,p284", "WAVE_STALL_UP": "p60-62,p187", "WAVE_STALL_DOWN": "p60-62,p222",
}


class WyckoffScanner:
    """逐根 K 線推進的狀態機（無未來函數）。

    產出：
      events   事件表（每列：idx 事件K線、confirm 確認K線、kind、price、tr_id、note、page）
      signals  交易訊號（L_* 多方進場；S_* 空方/多單出場；X_* 出場；W_* 警示）
      trs      交易區間摘要（含最終分類：吸籌/再吸籌/派發/再派發）
      bars     每根 K 線的 phase、tr_id、bias、support、resistance
    """

    def __init__(self, df: pd.DataFrame, p: Params = Params()):
        self.p = p
        self.d = add_features(df, p)
        self.s = bar_signals(self.d, p)
        self.piv = zigzag(self.d, 2.0)
        self.events, self.signals, self.trs = [], [], []
        self.setups = []          # 跨 TR 存活的待觸發設定（Spring 測試、LPS/BU、LPSY、UT 確認）
        self.tr: Optional[TR] = None
        self._tid = 0
        n = len(self.d)
        self.bar_phase = [""] * n
        self.bar_tr = [-1] * n
        self.bar_bias = [0.0] * n
        self.bar_sup = [NaN] * n
        self.bar_res = [NaN] * n
        self._last_resolved: Optional[TR] = None
        # 通道線（只用已確認的轉折點）
        self.ch = channels(self.d, self.piv, p)
        self._ch_by_conf = {}
        for e in self.ch["events"]:
            self._ch_by_conf.setdefault(e["confirm"], []).append(e)
        # 週線背景（以已完成的週對應回日 K）
        self.wk, self.wsc = None, None
        if p.use_weekly and "date" in self.d:
            self.wk, self.wsc = weekly_context(self.d, p)

    # ---------------- helpers ----------------
    def _ev(self, idx, confirm, kind, price, note="", tr=None, w=0.0):
        tr = tr if tr is not None else self.tr
        base = kind.split("(")[0] if kind not in EVENT_PAGES else kind
        e = dict(idx=int(idx), confirm=int(confirm), date=self.d.at[int(idx), "date"] if "date" in self.d else idx,
                 kind=kind, price=float(price), tr_id=(tr.id if tr else -1), note=note,
                 page=EVENT_PAGES.get(kind, EVENT_PAGES.get(base, "")))
        self.events.append(e)
        if tr is not None:
            tr.events.append(e)
            tr.bias += w
        return e

    def _sig(self, i, kind, side, stop=NaN, note="", page="", tr=None, tr_id=None):
        # tr_id 明確指定時（例如趨勢回調訊號，不隸屬任何區間），不回退到「目前區間」
        tr = tr if tr is not None else (None if tr_id is not None else self.tr)
        if side == "long" and self.wk is not None and not bool(self.wk.at[int(i), "wk_ok_long"]):
            note += "；⚠ 週線仍在下跌且尚未形成區間，逆大背景〔p147、p225〕"
        self.signals.append(dict(idx=int(i), date=self.d.at[int(i), "date"] if "date" in self.d else i,
                                 kind=kind, side=side, stop=float(stop), note=note, page=page,
                                 tr_id=(tr_id if tr_id is not None else (tr.id if tr else -1))))

    def _row(self, i):
        d = self.d
        return (d.at[i, "open"], d.at[i, "high"], d.at[i, "low"], d.at[i, "close"], d.at[i, "volume"],
                d.at[i, "atr"], d.at[i, "rvol"], d.at[i, "rspread"], d.at[i, "close_pos"])

    def _supply_dominant(self, i, k=10):
        """賣盤控制背景〔p212-224〕：近 k 根有 >=2 根放量陰線，且下跌量 > 上漲量。"""
        lo = max(0, i - k)
        s, d = self.s, self.d
        n_sup = int((s["vsb"].iloc[lo:i] | s["supply_bar"].iloc[lo:i]).sum())
        seg = d.iloc[max(0, i - 20):i]
        upv = seg.loc[seg["close"] > seg["open"], "volume"].sum()
        dnv = seg.loc[seg["close"] < seg["open"], "volume"].sum()
        return n_sup >= 2 and dnv > upv

    def _new_tr(self, i, prior, kind, price):
        self._tid += 1
        d = self.d
        tr = TR(id=self._tid, prior=prior, climax_kind=kind, climax_idx=i, climax_price=price,
                climax_vol=float(d.at[i, "volume"]), climax_spread=float(d.at[i, "spread"]), start=i)
        if prior == "down":
            tr.ar_price, tr.ar_idx = d.at[i, "high"], i
        else:
            tr.ar_price, tr.ar_idx = d.at[i, "low"], i
        self.tr = tr
        return tr

    # ---------------- climax / TR start ----------------
    def _climax_at(self, i) -> Optional[str]:
        """判斷第 i 根是否為高潮/停止行為起點（無副作用）。回傳 'SC'、'BC'、'SA_down'、'SA_up' 或 None。"""
        p, d, s = self.p, self.d, self.s
        if i < max(p.n, 30):
            return None
        o, h, l, c, v, atr, rvol, rsp, cpos = self._row(i)
        if any(isinstance(x, float) and math.isnan(x) for x in (atr, rvol, rsp)):
            return None
        lo_win = d["low"].iloc[max(0, i - p.climax_lookback + 1):i + 1].min()
        hi_win = d["high"].iloc[max(0, i - p.climax_lookback + 1):i + 1].max()
        down_ctx = d.at[i, "drawdown"] <= -p.min_decline and c < d.at[i, "sma"]
        up_ctx = d.at[i, "runup"] >= p.min_rise and c > d.at[i, "sma"]
        # SC：下跌趨勢後、創新低、天量、寬幅〔p65 註3、p74〕
        if down_ctx and l <= lo_win + 1e-12 and rvol >= p.vol_climax and rsp >= p.spread_big:
            return "SC"
        # BC：上漲趨勢後、創新高、天量、寬幅、急漲〔p66 註14、p151〕
        if up_ctx and h >= hi_win - 1e-12 and rvol >= p.vol_climax and rsp >= p.spread_big and d.at[i, "accel"] >= p.accel_atr:
            return "BC"
        # 無高潮時以停止行為起算〔p127：台股/陸股常無恐慌拋售〕
        if p.allow_stopping_action_start:
            if down_ctx and l <= lo_win + 1e-12 and rvol >= p.vol_high and (s.at[i, "stop1_down"] or s.at[i, "stop2_down"]):
                return "SA_down"
            if up_ctx and h >= hi_win - 1e-12 and rvol >= p.vol_high and (s.at[i, "stop1_up"] or s.at[i, "stop2_up"]):
                return "SA_up"
        return None

    def _start_tr(self, i, kind):
        d = self.d
        o, h, l, c, v, atr, rvol, rsp, cpos = self._row(i)
        if kind == "SC":
            tr = self._new_tr(i, "down", "SC", l)
            self._find_preliminary(i, tr, "PS")
            self._ev(i, i, "SC", l, f"恐慌拋售：量 {rvol:.1f}×、振幅 {rsp:.1f}×，下跌後創新低", tr, 0.5)
        elif kind == "BC":
            tr = self._new_tr(i, "up", "BC", h)
            self._find_preliminary(i, tr, "PSY")
            self._ev(i, i, "BC", h, f"搶購高潮：量 {rvol:.1f}×、振幅 {rsp:.1f}×、10 日漲 {d.at[i, 'accel']:.1f} ATR", tr, -0.5)
            self._sig(i, "W_BC", "warn", NaN, "搶購高潮：持股預警、縮緊停損至 BC 低點", "p116-117,p141", tr)
        elif kind == "SA_down":
            tr = self._new_tr(i, "down", "SA", l)
            self._ev(i, i, "SA", l, f"下跌中的停止行為（代替 SC）：量 {rvol:.1f}×", tr, 0.3)
        elif kind == "SA_up":
            tr = self._new_tr(i, "up", "SA", h)
            self._ev(i, i, "SA", h, f"上漲中的停止行為（代替 BC）：量 {rvol:.1f}×", tr, -0.3)

    def _find_preliminary(self, i, tr, kind):
        """在高潮前 3~40 根尋找 PS（初次支撐）/ PSY（初始供應）〔p72-73、p149〕。"""
        s, d = self.s, self.d
        lo = max(0, i - 40)
        for j in range(i - 3, lo - 1, -1):
            if kind == "PS" and (s.at[j, "stop1_down"] or s.at[j, "stop2_down"]) and d.at[j, "rvol"] >= self.p.vol_high:
                self._ev(j, i, "PS", d.at[j, "low"], "初次支撐：放量停止行為，SC 出現後才可回頭確認", tr, 0.2)
                return
            if kind == "PSY" and (s.at[j, "stop1_up"] or s.at[j, "stop2_up"] or s.at[j, "supply_bar"]) and d.at[j, "rvol"] >= self.p.vol_high:
                self._ev(j, i, "PSY", d.at[j, "high"], "初始供應：放量上影/放量下跌，BC 出現後回頭確認", tr, -0.2)
                return

    # ---------------- Phase A ----------------
    def _phase_a(self, i, tr):
        p, d = self.p, self.d
        o, h, l, c, v, atr, rvol, rsp, cpos = self._row(i)
        if tr.state == "await_ar":
            if tr.prior == "down":
                if l < tr.climax_price:                       # 高潮延續數日〔p65：一天或幾天內〕
                    if rvol >= p.vol_high:
                        tr.climax_idx, tr.climax_price = i, l
                        tr.climax_vol, tr.climax_spread = max(tr.climax_vol, v), max(tr.climax_spread, h - l)
                        tr.ar_price, tr.ar_idx = h, i
                    elif l < tr.climax_price - atr:
                        self._drop(i, tr, "高潮後續創新低且未放量：停止行為失敗")
                        return
                if h > tr.ar_price:
                    tr.ar_price, tr.ar_idx = h, i
                rally = tr.ar_price - tr.climax_price
                # AR 確認：極值出現在先前的 K 線，且收盤自極值回撤 >= 1 ATR（避免單根寬幅K線自我確認）
                if (rally >= p.ar_min_atr * atr and tr.ar_idx < i and tr.ar_price - c >= p.ar_confirm_atr * atr
                        and tr.ar_idx > tr.climax_idx):
                    tr.resistance = tr.ar_price
                    tr.state = "await_st"
                    self._ev(tr.ar_idx, i, "AR", tr.ar_price, "自動反彈高點＝區間上緣（多為空頭回補，勿追）", tr, 0.0)
            else:
                if h > tr.climax_price:
                    if rvol >= p.vol_high:
                        tr.climax_idx, tr.climax_price = i, h
                        tr.climax_vol, tr.climax_spread = max(tr.climax_vol, v), max(tr.climax_spread, h - l)
                        tr.ar_price, tr.ar_idx = l, i
                    elif h > tr.climax_price + atr:
                        self._drop(i, tr, "高潮後續創新高且未放量：上漲恢復")
                        return
                if l < tr.ar_price:
                    tr.ar_price, tr.ar_idx = l, i
                drop = tr.climax_price - tr.ar_price
                if (drop >= p.ar_min_atr * atr and tr.ar_idx < i and c - tr.ar_price >= p.ar_confirm_atr * atr
                        and tr.ar_idx > tr.climax_idx):
                    tr.support = tr.ar_price
                    tr.state = "await_st"
                    self._ev(tr.ar_idx, i, "AR(自然回落)", tr.ar_price, "自然回落低點＝派發支撐／冰線〔p160〕", tr, 0.0)
            if tr.state == "await_ar" and i - tr.climax_idx > p.ar_max_bars:
                self._drop(i, tr, "高潮後未出現有效的 AR")
            return

        # await_st
        if tr.prior == "down":
            height = tr.resistance - tr.climax_price
            zone_top = tr.climax_price + p.st_zone * height
            in_zone = l <= zone_top
            ep = tr.st_ep
            if in_zone:
                if ep is None:
                    ep = tr.st_ep = dict(start=i, low=l, low_i=i, vmax=v, smax=h - l)
                else:
                    if l < ep["low"]:
                        ep["low"], ep["low_i"] = l, i
                    ep["vmax"], ep["smax"] = max(ep["vmax"], v), max(ep["smax"], h - l)
                if l < tr.climax_price - atr and rvol >= p.vol_high and c < tr.climax_price:
                    self._drop(i, tr, "二次測試放量跌破 SC 低點：熊市恢復〔p74〕")
                    return
            elif ep is not None and c > zone_top:
                ratio = ep["vmax"] / max(tr.climax_vol, 1e-9)
                if ratio <= p.st_vol_ratio and ep["low"] >= tr.climax_price - atr:
                    tr.support = min(tr.climax_price, ep["low"]) if ep["low"] >= tr.climax_price - 0.5 * atr else tr.climax_price
                    tr.st_idx, tr.state, tr.phase = ep["low_i"], "B", "B"
                    tr.last_test_low = ep["low"]
                    self._ev(ep["low_i"], i, "ST", ep["low"], f"二次測試成功：測試量為高潮量的 {ratio:.0%}（供應減弱）", tr, 1.0)
                else:
                    self._ev(ep["low_i"], i, "ST(量大)", ep["low"], f"二次測試量仍大（{ratio:.0%}），需再測試〔p74〕", tr, -0.3)
                tr.st_ep = None
        else:
            height = tr.climax_price - tr.support
            zone_bot = tr.climax_price - p.st_zone * height
            in_zone = h >= zone_bot
            ep = tr.st_ep
            if in_zone:
                if ep is None:
                    ep = tr.st_ep = dict(start=i, high=h, high_i=i, vmax=v, smax=h - l)
                else:
                    if h > ep["high"]:
                        ep["high"], ep["high_i"] = h, i
                    ep["vmax"], ep["smax"] = max(ep["vmax"], v), max(ep["smax"], h - l)
                if h > tr.climax_price + atr and rvol >= p.vol_high and c > tr.climax_price:
                    self._drop(i, tr, "二次測試放量突破 BC：上漲恢復")
                    return
            elif ep is not None and c < zone_bot:
                ratio = ep["vmax"] / max(tr.climax_vol, 1e-9)
                if ratio <= p.st_vol_ratio and ep["high"] <= tr.climax_price + atr:
                    tr.resistance = max(tr.climax_price, ep["high"]) if ep["high"] <= tr.climax_price + 0.5 * atr else tr.climax_price
                    tr.st_idx, tr.state, tr.phase = ep["high_i"], "B", "B"
                    self._ev(ep["high_i"], i, "ST(派發)", ep["high"], f"二次測試：量為高潮量的 {ratio:.0%}，購買力耗盡〔p152〕", tr, -1.0)
                else:
                    self._ev(ep["high_i"], i, "ST(量大)", ep["high"], f"派發二次測試量仍大（{ratio:.0%}），需再測試〔p183〕", tr, 0.3)
                tr.st_ep = None
        if tr.state == "await_st" and i - tr.start > p.st_max_bars:
            self._drop(i, tr, "長時間未出現二次測試")

    def _drop(self, i, tr, why):
        tr.result, tr.end, tr.state = "未成形：" + why, i, "done"
        self.trs.append(tr)
        self.tr = None

    # ---------------- Phase B/C/D ----------------
    def _in_range(self, i, tr):
        p, d, s = self.p, self.d, self.s
        o, h, l, c, v, atr, rvol, rsp, cpos = self._row(i)
        sup, res = tr.support, tr.resistance
        height = res - sup
        mid = sup + height / 2
        # ---- 下方穿越：Spring / 震倉 / 破冰 ----
        if l < sup - p.pen_min_atr * atr:
            pe = tr.pen_dn
            if pe is None:
                pe = tr.pen_dn = dict(start=i, low=l, low_i=i, vmax=v, rvol=rvol, rsp=rsp, cpos=cpos, ice=False)
            elif l < pe["low"]:
                pe.update(low=l, low_i=i, rvol=max(pe["rvol"], rvol), rsp=max(pe["rsp"], rsp), cpos=cpos)
                pe["vmax"] = max(pe["vmax"], v)
        pe = tr.pen_dn
        if pe is not None:
            k = i - pe["start"]
            depth = (sup - pe["low"]) / atr
            if c > sup:                                         # 收回支撐上方
                if k < p.spring_recover_bars and not pe["ice"]:
                    self._spring(i, tr, pe, depth)
                elif (pe["ice"] or depth >= p.shakeout_depth_atr) and k < p.shakeout_recover_bars:
                    self._ev(pe["low_i"], i, "Shakeout", pe["low"],
                             f"跌破 {depth:.1f} ATR 後 {k} 根內收回區間：終極震倉，待二次測試〔p122、p131〕", tr, 2.0)
                    self._add_test_setup(i, tr, pe, "Shakeout")
                    tr.phase = "C" if tr.phase == "B" else tr.phase
                elif k < p.shakeout_recover_bars:
                    self._ev(pe["low_i"], i, "Spring(延遲收回)", pe["low"],
                             f"小幅跌破 {depth:.2f} ATR，{k} 根後才收回（超過原書 1~3 根），需二次測試〔p194〕", tr, 1.0)
                    self._add_test_setup(i, tr, pe, "延遲收回的 Spring")
                tr.pen_dn = None
            elif not pe["ice"] and k >= p.spring_recover_bars - 1 and c < sup - 0.25 * atr:
                pe["ice"] = True
                self._ev(i, i, "ICE", c, "收盤有效跌破支撐（破冰）：若 10 根內未收回即確認下跌〔p156〕", tr, -3.0)
                self._sig(i, "X_ICE", "exit", NaN, "破冰：多單出場；反彈無量為 LPSY 做空點", "p156,p171", tr)
                self.setups.append(dict(kind="LPSY", start=i, ref=sup, ref_hi=sup, tr=tr, rally=[], exp=i + p.lpsy_max_bars))
            elif pe["ice"] and k >= p.shakeout_recover_bars:
                self._resolve(i, tr, "down")
                return
        # ---- 上方穿越：UT / JOC ----
        if h > res + p.pen_min_atr * atr:
            pu = tr.pen_up
            if pu is None:
                pu = tr.pen_up = dict(start=i, high=h, high_i=i, vmax=v, rvol=rvol, rsp=rsp, joc=False)
            elif h > pu["high"]:
                pu.update(high=h, high_i=i, vmax=max(pu["vmax"], v), rvol=max(pu["rvol"], rvol))
        pu = tr.pen_up
        if pu is not None:
            k = i - pu["start"]
            if c < res and k < p.spring_recover_bars and not pu["joc"]:
                self._ut(i, tr, pu)
                tr.pen_up = None
            elif not pu["joc"] and c > res + 0.1 * atr:
                if rvol >= 1.3 and rsp >= 1.0 and cpos >= 0.6:
                    pu["joc"] = True
                    self._ev(i, i, "JOC", c, f"跳離震盪區：放量長陽收在區間上緣之上（量 {rvol:.1f}×）", tr, 3.0)
                    if tr.phase in ("B", "C"):
                        tr.phase = "D"
                    # JOC 取代先前的 LPS 設定：之後的縮量回測以「回測突破點」為準〔p98、p111、p133〕
                    self.setups = [x for x in self.setups if not (x["kind"] == "LPS" and x["tr"] is tr)]
                    self.setups.append(dict(kind="BU", start=i, ref=res, sos_low=l, tr=tr, pull=[], exp=i + p.lps_max_bars))
                elif k == 0:
                    self._ev(i, i, "突破(量不足)", c, "突破但量或陽線長度不足，耐心等測試〔p125〕", tr, 0.5)
                    pu["joc"] = True
            elif c < res and pu["joc"]:
                tr.pen_up = None                              # 突破後回到區間內
        # ---- 區間內的強弱柱 ----
        if tr.pen_dn is None and tr.pen_up is None and sup <= c <= res:
            fresh_sos, fresh_sow = i - tr.last_sos > 3, i - tr.last_sow > 3
            if c > o and rsp >= p.spread_big and rvol >= p.vol_high and cpos >= 0.6 and c > mid:
                if fresh_sos:
                    self._ev(i, i, "SOS", c, f"區間內強勢出現：長陽放量（量 {rvol:.1f}×），需縮量回調確認〔p66〕", tr, 2.0)
                    self.setups.append(dict(kind="LPS", start=i, ref=max(sup, tr.last_test_low), sos_low=l, tr=tr, pull=[], exp=i + p.lps_max_bars))
                tr.last_sos = i
                if tr.phase in ("B", "C"):
                    tr.phase = "D"
            elif s.at[i, "demand_wave3"] and c > mid and d["rvol"].iloc[i - 2:i + 1].mean() >= 1.0:
                if fresh_sos:
                    self._ev(i, i, "SOS(波)", c, "連續三根三個更高＋遞增量：以上升波完成的 SOS〔p95〕", tr, 1.5)
                    self.setups.append(dict(kind="LPS", start=i, ref=max(sup, tr.last_test_low), sos_low=d["low"].iloc[i - 2], tr=tr, pull=[], exp=i + p.lps_max_bars))
                tr.last_sos = i
                if tr.phase in ("B", "C"):
                    tr.phase = "D"
            elif c < o and rsp >= p.spread_big and rvol >= p.vol_high and cpos <= 0.4 and c < mid:
                if fresh_sow:
                    self._ev(i, i, "SOW", c, f"弱勢出現：放量長陰（量 {rvol:.1f}×），需無量反彈確認〔p155-156〕", tr, -2.0)
                    self.setups.append(dict(kind="LPSY", start=i, ref=h, ref_hi=h, tr=tr, rally=[], exp=i + p.lpsy_max_bars))
                tr.last_sow = i
                if tr.phase in ("B", "C"):
                    tr.phase = "D"
        # ---- Phase E：離開區間 ----
        if c > res + p.exit_range_atr * atr:
            tr.exit_cnt = tr.exit_cnt + 1 if tr.exit_cnt >= 0 else 1
        elif c < sup - p.exit_range_atr * atr:
            tr.exit_cnt = tr.exit_cnt - 1 if tr.exit_cnt <= 0 else -1
        else:
            tr.exit_cnt = 0
        if tr.exit_cnt >= p.exit_range_bars:
            self._resolve(i, tr, "up")
        elif tr.exit_cnt <= -p.shakeout_recover_bars:
            self._resolve(i, tr, "down")
        elif i - tr.start > p.max_tr_bars:
            tr.result, tr.end, tr.state = "逾時未決", i, "done"
            self.trs.append(tr)
            self.tr = None

    def _spring(self, i, tr, pe, depth):
        p = self.p
        rvol, rsp, cpos = pe["rvol"], pe["rsp"], self.d.at[i, "close_pos"]
        if depth >= p.shakeout_depth_atr and (rvol >= p.vol_high or rsp >= p.spread_wide):
            self._ev(pe["low_i"], i, "Shakeout", pe["low"], f"深跌 {depth:.1f} ATR、放量後快速收回：震倉，待二次測試〔p131、p196〕", tr, 2.0)
            self._add_test_setup(i, tr, pe, "Shakeout")
            tr.phase = "C"
            return
        if rvol < 1.0 and rsp < 1.0:
            typ, direct = "A 縮量縮幅（最佳，可直接進場）", True                      # p196 中間型
        elif rvol >= 1.0 and cpos >= 0.5 and rsp < p.spread_wide:
            typ, direct = "B+ 量增但收在 50% 之上（衍生成功型）", True                # p197 第 3/4 項
        elif rvol >= p.vol_high and rsp >= p.spread_wide:
            typ, direct = "C 量大幅寬（類震倉，等二次測試）", False                   # p196 左側型
        else:
            typ, direct = "B 量略增（等二次測試）", False                            # p196 右側型
        bias_ok = not self._supply_dominant(i)
        self._ev(pe["low_i"], i, "Spring", pe["low"], f"彈簧效應 {typ}；跌破 {depth:.2f} ATR 後 {i - pe['start']} 根收回", tr, 2.0 if bias_ok else 0.5)
        tr.phase = "C"
        tr.last_test_low = pe["low"]
        stop = pe["low"] - p.stop_buffer_atr * self.d.at[i, "atr"]
        if direct and bias_ok:
            self._sig(i, "L_SPRING", "long", stop, f"Spring 直接進場（{typ[:2]}），停損放在跌破最低點下方", "p195-197,p224", tr)
            self.setups.append(dict(kind="SPRING_FAIL", start=i, low=pe["low"], tr=tr, exp=i + 5))
        else:
            why = "賣盤控制背景，Spring 易失敗" if not bias_ok else "量能偏大"
            self._add_test_setup(i, tr, pe, f"Spring（{why}）")

    def _add_test_setup(self, i, tr, pe, what):
        # 同一區間只保留最新一個待測試設定
        self.setups = [x for x in self.setups if not (x["kind"] == "TEST" and x["tr"] is tr)]
        self.setups.append(dict(kind="TEST", start=i, low=pe["low"], vmax=pe["vmax"], tr=tr, what=what,
                                exp=i + self.p.test_window))

    def _ut(self, i, tr, pu):
        p, d = self.p, self.d
        late = tr.st_idx >= 0 and (i - tr.st_idx) >= 10
        vol_ok = pu["rvol"] >= p.vol_up
        kind = "UTAD" if (late and tr.prior == "up") else "UT"
        w = -2.0 if kind == "UTAD" else (-1.0 if tr.prior == "up" else -0.5)
        note = ("區間右手邊上衝回落：放量假突破後 1~3 根回到區間（派發測試）" if kind == "UTAD"
                else "上衝回落：突破區間上緣後收回" + ("（放量）" if vol_ok else "（未放量＝試探，不宜做空〔p38〕）"))
        self._ev(pu["high_i"], i, kind, pu["high"], note, tr, w if vol_ok else w / 2)
        if tr.phase == "B" and kind == "UTAD":
            tr.phase = "C"
        self.setups.append(dict(kind="UT_CONFIRM", start=i, high=pu["high"], tr=tr, ut_kind=kind, exp=i + 3))

    def _resolve(self, i, tr, direction):
        if direction == "up":
            tr.result = "吸籌（Accumulation）" if tr.prior == "down" else "再吸籌（Re-accumulation）"
            self._ev(i, i, "MarkupE", self.d.at[i, "close"], "價格離開區間向上：Phase E（上漲）", tr, 0.0)
        else:
            tr.result = "派發（Distribution）" if tr.prior == "up" else "再派發（Re-distribution）"
            self._ev(i, i, "MarkdownE", self.d.at[i, "close"], "價格離開區間向下：Phase E（下跌）", tr, 0.0)
        tr.end, tr.state, tr.phase = i, "done", "E"
        self.trs.append(tr)
        self._last_resolved = tr
        self.tr = None

    # ---------------- 跨 TR 的待觸發設定 ----------------
    def _process_setups(self, i):
        p, d, s = self.p, self.d, self.s
        o, h, l, c, v, atr, rvol, rsp, cpos = self._row(i)
        group = lambda st: (id(st["tr"]), {"LPS": "entry", "BU": "entry", "LPSY": "lpsy"}.get(st["kind"], st["kind"] + str(st["start"])))
        fired = set()
        keep = []
        for st in self.setups:
            if group(st) in fired:
                continue
            if i <= st["start"]:
                keep.append(st)
                continue
            tr = st["tr"]
            done = False
            if st["kind"] == "TEST":                                  # Spring/震倉的二次測試〔p90、p137、p197〕
                if c < st["low"] - 0.25 * atr:
                    self._ev(i, i, "Spring失敗", c, f"{st['what']}後收盤跌破低點：需求不足〔p197〕", tr, -2.0)
                    done = True
                elif (l <= st["low"] + 1.0 * atr and l >= st["low"] - 0.25 * atr
                      and rvol <= p.vol_low and v <= 0.7 * st["vmax"] and rsp <= 1.1 and cpos >= 0.4):
                    self._ev(i, i, "Test", l, f"{st['what']}的二次測試：量縮（{rvol:.1f}×）、守住低點＝風險最低的進場點〔p90〕", tr, 1.0)
                    self._sig(i, "L_TEST", "long", min(st["low"], l) - p.stop_buffer_atr * atr,
                              "Spring/震倉二次測試進場，停損放在測試/跌破低點下方", "p83,p90,p137", tr)
                    if tr is not None:
                        tr.last_test_low = l
                    done = True
            elif st["kind"] == "SPRING_FAIL":
                if c < st["low"]:
                    self._ev(i, i, "Spring失敗", c, "Spring 後跌破低點：進場依據被否定，立刻離場〔p192、p203〕", tr, -2.0)
                    done = True
            elif st["kind"] in ("LPS", "BU"):                          # SOS/JOC 後的縮量回調〔p82、p98、p115〕
                pull = st["pull"]
                if s.at[i, "supply_bar"]:
                    self._ev(i, i, "SOS失效", c, "回測出現放量長陰：SOS/JOC 失效〔p93、p98、p111〕", tr, -2.0)
                    done = True
                elif c < d.at[i - 1, "close"] or (pull and c <= d.at[i - 1, "high"]):
                    pull.append(i)
                elif pull:
                    rv = d["rvol"].iloc[pull].mean()
                    rs = d["rspread"].iloc[pull].max()
                    plow = d["low"].iloc[pull].min()
                    floor = (st["ref"] - 0.5 * atr) if st["kind"] == "BU" else (st["ref"] + 0.25 * atr)
                    if c > d.at[i - 1, "high"] and rv <= 0.9 and rs <= 1.25 and plow >= floor:
                        name = "BU" if st["kind"] == "BU" else "LPS"
                        self._ev(pull[int(np.argmin(d["low"].iloc[pull].to_numpy()))], i, name, plow,
                                 f"縮量回調（均量 {rv:.1f}×）守住{'突破點' if name == 'BU' else '更高的低點'}後需求回來", tr, 2.0)
                        stop = (min(st["sos_low"], plow) if name == "BU" else plow) - p.stop_buffer_atr * atr
                        self._sig(i, "L_BU" if name == "BU" else "L_LPS", "long", stop,
                                  "JOC 回測進場（停損：突破K線低點下方）" if name == "BU" else "SOS 後 LPS 進場（停損：LPS 低點下方）",
                                  "p36,p98,p106,p215,p289" if name == "BU" else "p95,p115,p133", tr)
                        fired.add(group(st))
                        done = True
                    elif plow < floor:
                        done = True                                      # 回測太深，放棄
                    else:
                        pull.clear()
            elif st["kind"] == "LPSY":     # SOW/破冰後的無量反彈〔p152、p156〕；跌破需求線後的無需求反彈〔p229–231〕
                rally = st["rally"]
                nd = st.get("sig_kind") == "S_NODEMAND"
                if s.at[i, "demand_bar"] and c > st["ref_hi"]:
                    if not nd:
                        self._ev(i, i, "SOW失效", c, "SOW 後出現放量長陽收復：新需求進場，SOW 失效〔p161〕", tr, 2.0)
                    done = True
                elif c > d.at[i - 1, "close"]:
                    rally.append(i)
                elif rally and c < d.at[i - 1, "low"]:
                    rv = d["rvol"].iloc[rally].mean()
                    rhi = d["high"].iloc[rally].max()
                    if rv <= 0.9 and rhi <= st["ref_hi"] + 0.25 * atr:
                        top = rally[int(np.argmax(d["high"].iloc[rally].to_numpy()))]
                        if nd:
                            self._ev(top, i, "NoDemandRally", rhi,
                                     f"跌破需求線後的無需求反彈（均量 {rv:.1f}×）無法收復破位：供應背景下的反彈", tr, -1.0)
                            self._sig(i, "S_NODEMAND", "short", rhi + p.stop_buffer_atr * atr,
                                      "無需求反彈：多單出場、不搶反彈；放空者停損放在反彈高點上方", "p229-231,p253,p273", tr)
                        else:
                            self._ev(top, i, "LPSY", rhi,
                                     f"弱勢反彈（均量 {rv:.1f}×）無法收復：最後派發點＝最佳做空點／最後逃命點", tr, -2.0)
                            self._sig(i, "S_LPSY", "short", rhi + p.stop_buffer_atr * atr,
                                      "LPSY：多單最後出場點；可放空者停損放在反彈高點上方", "p152,p156", tr)
                        fired.add(group(st))
                        done = True
                    else:
                        rally.clear()
            elif st["kind"] == "UT_CONFIRM":                           # UT 須有放量下挫跟隨〔p154〕
                if c < o and v > d.at[i - 1, "volume"] and rvol >= 1.0:
                    self._ev(i, i, "UT確認", c, "上衝回落後放量下挫：供應持續控制", tr, -1.0)
                    if tr is None or tr.prior == "up" or (tr.bias < 0):
                        self._sig(i, "S_UT", "short", st["high"] + p.stop_buffer_atr * atr,
                                  f"{st['ut_kind']} 確認：多單出場；放空停損設在 UT 高點上方一檔", "p154,p207,p212", tr)
                    done = True
            if not done and i < st["exp"]:
                keep.append(st)
        self.setups = [st for st in keep if group(st) not in fired]

    # ---------------- 通道事件併入 ----------------
    def _channel_events(self, i):
        """把當根確認的通道事件併入事件表，並依原書意義調整進行中區間的偏多/偏空分數。"""
        for e in self._ch_by_conf.get(i, []):
            tr, k, strong = self.tr, e["kind"], e["strong"]
            w = 0.0
            if k == "SUPPLY_LINE_BREAK" and strong:
                w = 1.0 if (tr is not None and tr.prior == "down") else 0.5   # 驗證 SC 形成的臨時支撐〔p112〕
            elif k == "DEMAND_LINE_BREAK":
                w = -1.0 if strong else 0.0     # 未放量的跌破只是資訊（區間內的小通道常被震盪穿越）
            elif k == "SOT_CH_UP":
                w = -0.5
            elif k == "SOT_CH_DOWN":
                w = 0.5
            self._ev(e["idx"], e["confirm"], k, e["price"], e["note"], tr, w)
            if k == "DEMAND_LINE_BREAK" and strong:
                if self.p.exit_on_demand_line_break:
                    self._sig(i, "X_TRENDLINE", "exit", NaN, "放量跌破需求線（上升趨勢線）：無論盈虧離場", "p142,p146", tr)
                # 之後若出現無需求反彈（量縮、收不回破位），發出 S_NODEMAND〔p229–231、p272–273〕
                hi3 = float(self.d["high"].iloc[max(0, i - 2):i + 1].max())
                self.setups.append(dict(kind="LPSY", start=i, ref=hi3, ref_hi=hi3, tr=tr, rally=[],
                                        exp=i + self.p.lpsy_max_bars, sig_kind="S_NODEMAND"))
            elif k == "SOT_CH_UP":
                self._sig(i, "W_SOT_CH", "warn", NaN, e["note"], e["page"], tr)

    # ---------------- v1.2：趨勢回調進場（書中後記推薦的入門方法〔p316〕）----------------
    def _pullback_pass(self):
        """牛市回調進場，不需要先有交易區間。全部只用當根以前的資料。

        ① 背景：50 日均線上升且收盤不低於均線 0.5 ATR，近 20 日上漲日量／下跌日量 >= pb_ud_min（需求占上風）〔p316〕
        ② 需要的回調：自近 40 根高點回落不超過上漲腿 50%〔p143–146〕；回調中沒有放量長陰或垂直供應柱
           （回調沒有引起供應擴大〔p28〕）；回調 K 線平均量與振幅不高於均值（小蠟燭、小成交量〔p316〕）；
           20 根內沒有搶購高潮〔p116–117〕
        ③ 支撐區：垂直需求柱底部〔p31、p39–40、p248–251〕、突破點（前高變支撐）〔p98、p111、p241〕、
           需求線〔p269–270〕、上漲腿 50% 位置〔p239–240〕
        ④ 扣扳機＝停止行為〔p316〕：跌破支撐區後收回（Spring）、放量滯跌、長下影放量、縮量小蠟燭（無供應）、
           下跌推進幅度縮小（SOT）
        停損：觸發 K 線低點與支撐結構點兩者較低者之下〔p36、p224、p289〕。
        """
        p, d, s = self.p, self.d, self.s
        n = len(d)
        if n < 70:
            return
        h, l, c = (d[k].to_numpy(float) for k in ("high", "low", "close"))
        atr = d["atr"].to_numpy(float)
        rvol = d["rvol"].to_numpy(float)
        rsp = d["rspread"].to_numpy(float)
        cpos = d["close_pos"].to_numpy(float)
        sma = d["sma"].to_numpy(float)
        slope = d["sma_slope"].to_numpy(float)
        lim = d["limit_move"].fillna(False).to_numpy(bool)
        udr = wyckoff_indicators(d, s)["ud_vol_ratio"].to_numpy(float)
        f = {k: s[k].to_numpy(bool) for k in ("vdb", "vsb", "supply_bar", "stop1_down", "stop2_down", "no_supply")}
        bc_conf = np.array(sorted(e["confirm"] for e in self.events if e["kind"] == "BC"), dtype=int)
        W, last, last_stop = p.pb_lookback, -10 ** 9, None
        for i in range(max(W, 60), n):
            if last_stop is not None and l[i] < last_stop:
                last, last_stop = -10 ** 9, None      # 前一個訊號已被否定（跌破停損）→ 等下一次測試〔p223〕
            a = atr[i]
            if lim[i] or not np.isfinite(a) or a <= 0 or i - last < p.pb_cooldown:
                continue
            if not (slope[i] > 0 and c[i] >= sma[i] - 0.5 * a):                    # 牛市背景
                continue
            seg = h[i - W + 1:i + 1]
            hh_i = i - W + 1 + (len(seg) - 1 - int(np.argmax(seg[::-1])))         # 最近一次的最高點
            if not (2 <= i - hh_i <= p.pb_max_age):
                continue
            b0 = max(0, hh_i - W)
            base_i = b0 + int(np.argmin(l[b0:hh_i + 1]))
            hh, base = h[hh_i], l[base_i]
            leg = hh - base
            if base_i >= hh_i or leg < p.pb_min_leg_atr * a:
                continue
            retr = (hh - float(np.min(l[hh_i + 1:i + 1]))) / leg
            if retr > p.pb_max_retr:
                continue
            if f["vsb"][hh_i + 1:i + 1].any() or f["supply_bar"][hh_i + 1:i + 1].any():
                continue
            if not (udr[i] >= p.pb_ud_min):
                continue
            q = slice(hh_i + 1, i)                                                 # 回調品質（不含觸發 K）
            if np.nanmean(rvol[q]) > 1.0 or np.nanmean(rsp[q]) > 1.0:
                continue
            if len(bc_conf) and ((bc_conf >= i - 20) & (bc_conf <= i)).any():
                continue
            z = p.pb_zone_atr * a
            zones = []
            js = np.where(f["vdb"][base_i:hh_i + 1])[0]
            if len(js):
                j = base_i + int(js[-1])
                zones.append(("垂直需求柱底部", l[j] - z, l[j] + z, l[j], "p31,p39-40,p248-251"))
            prev_hi = float(np.max(h[max(0, base_i - W):base_i + 1]))
            if hh > prev_hi + 0.5 * a:
                zones.append(("突破點（前高變支撐）", prev_hi - z, prev_hi + z, prev_hi - 0.5 * a, "p98,p111,p241"))
            if self.ch["dir"][i] == "up" and np.isfinite(self.ch["lower"][i]):
                lv = float(self.ch["lower"][i])
                zones.append(("需求線（上升趨勢線）", lv - z, lv + z, lv - 0.5 * a, "p269-270"))
            mid = hh - 0.5 * leg
            zones.append(("上漲腿 50% 位置", mid - z, mid + z, None, "p143-146,p239-240"))
            # 原書後記的牛市回調方法〔p316〕不要求特定支撐；回調在 25%～50% 之間即屬健康回調〔p143–146〕
            zones.append(("健康回調區（上漲腿 25%～50%）", hh - 0.5 * leg, hh - 0.25 * leg, None, "p143-146,p316"))
            sot = l[i] < l[i - 1] < l[i - 2] and (l[i - 1] - l[i]) <= 0.5 * (l[i - 2] - l[i - 1]) and rvol[i] <= 1.0
            hit = None
            for name, zlo, zhi, anchor, page in zones:
                inside = zlo <= l[i] <= zhi
                pierced = l[i] < zlo and l[i] >= zlo - a and c[i] >= zlo and cpos[i] >= 0.5 and rsp[i] <= 1.5
                if pierced:
                    trig = ("跌破支撐區後收回（Spring）", "p66,p194-197")
                elif not inside:
                    continue
                elif f["stop2_down"][i]:
                    trig = ("放量滯跌（努力沒有結果）", "p46,p80,p239")
                elif f["stop1_down"][i]:
                    trig = ("長下影線＋放量的停止行為", "p78")
                elif f["no_supply"][i] and rvol[i] <= p.vol_low:
                    trig = ("縮量小蠟燭回調（無供應）", "p36,p100,p111")
                elif sot:
                    trig = ("下跌推進幅度縮小（SOT）", "p137,p242")
                else:
                    continue
                hit = (name, anchor, page, trig)
                break
            if hit is None:
                continue
            name, anchor, page, (tname, tpage) = hit
            stop = min(l[i], anchor if anchor is not None else l[i]) - p.stop_buffer_atr * a
            if c[i] <= stop or c[i] - stop > p.pb_max_risk_atr * a:
                continue
            note = f"多頭回調到{name}出現{tname}（回調 {retr:.0%}、回調均量 {np.nanmean(rvol[q]):.1f}×）"
            pages = f"p316,{page},{tpage}"
            self.events.append(dict(idx=i, confirm=i, date=d.at[i, "date"] if "date" in d else i, kind="PB",
                                    price=float(l[i]), tr_id=-1, note=note, page=pages))
            self._sig(i, "L_PULLBACK", "long", stop, note + "；停損放在觸發K線與支撐結構的低點之下", pages, tr_id=-1)
            last, last_stop = i, stop

    # ---------------- v1.2：死角交易法〔p114、p233、p290〕----------------
    def _corner_pass(self):
        """需求背景下，價格在近期區間的下半部「進入死角」（高點遞降、低點遞升、振幅收斂、量能萎縮）。
        原書：死角是進場的最好時機，扣扳機的依據是市場進入死角這個現象本身〔p114〕；
        死角的突破方向由背景決定，需求大於供應時向上〔p233、p290〕。停損放在死角最低點之下【延伸】。"""
        p, d, s = self.p, self.d, self.s
        n = len(d)
        if n < 70:
            return
        ind = wyckoff_indicators(d, s)
        dc = ind["dead_corner"].fillna(False).to_numpy(bool)
        ds = ind["demand_score"].to_numpy(float)
        rp = ind["range_pos"].to_numpy(float)
        dry = ind["dry_up"].to_numpy(float)
        l, c = d["low"].to_numpy(float), d["close"].to_numpy(float)
        v, sp = d["volume"].to_numpy(float), d["spread"].to_numpy(float)
        atr = d["atr"].to_numpy(float)
        reg = d["regime"].to_numpy()
        sup = (s["vsb"] | s["supply_bar"]).to_numpy(bool)
        last = -10 ** 9
        for i in range(60, n):
            if not dc[i] or dc[i - 1] or i - last < 10 or not np.isfinite(atr[i]):
                continue
            bg = reg[i] == "up" or (self.bar_phase[i] in ("B", "C", "D") and self.bar_bias[i] >= 1.0)
            if not (bg and rp[i] <= p.corner_range_pos_max and dry[i] <= p.corner_rvol_max):
                continue
            # 死角內的需求證據：上漲 K 的量與長度都大於下跌 K〔p290〕，或綜合需求分數夠高
            w = slice(i - 9, i + 1)
            upk = c[w] > np.r_[c[i - 10], c[i - 9:i]]
            dnk = ~upk
            corner_demand = (upk.any() and dnk.any() and v[w][upk].sum() > v[w][dnk].sum()
                             and sp[w][upk].mean() >= sp[w][dnk].mean())
            if not (corner_demand or ds[i] >= p.corner_demand_min):
                continue
            if sup[max(0, i - 10):i + 1].any():
                continue
            stop = float(np.min(l[i - 9:i + 1])) - p.stop_buffer_atr * atr[i]
            if c[i] <= stop or c[i] - stop > 3.0 * atr[i]:
                continue
            note = (f"需求背景下進入死角（需求分數 {ds[i]:.0f}、近 5 日均量 {dry[i]:.1f}×、區間位置 {rp[i]:.0%}）："
                    "供求都很弱，需求稍作努力即可向上突破")
            tid = int(self.bar_tr[i])
            self.events.append(dict(idx=i, confirm=i, date=d.at[i, "date"] if "date" in d else i, kind="CORNER",
                                    price=float(c[i]), tr_id=tid, note=note, page="p65,p114,p233,p290"))
            self._sig(i, "L_CORNER", "long", stop, note + "；停損放在死角最低點之下", "p114,p233,p290", tr_id=tid)
            last = i

    # ---------------- 主迴圈 ----------------
    def run(self):
        n = len(self.d)
        for i in range(1, n):
            self._process_setups(i)
            self._channel_events(i)
            tr = self.tr
            if tr is None:
                k = self._climax_at(i)
                if k:
                    self._start_tr(i, k)
            else:
                # 較弱的 SA 起點（仍在 Phase A）遇到真正高潮時，以高潮重新起算
                if tr.state in ("await_ar", "await_st") and tr.climax_kind == "SA":
                    k = self._climax_at(i)
                    if k in ("SC", "BC"):
                        tr.result, tr.end, tr.state = "被高潮取代", i, "done"
                        self.trs.append(tr)
                        self.tr = None
                        self._start_tr(i, k)
                        self._record(i)
                        continue
                if tr.state in ("await_ar", "await_st"):
                    self._phase_a(i, tr)
                elif tr.state == "B":
                    self._in_range(i, tr)
            self._record(i)
        if self.tr is not None:
            self.tr.result = self.tr.result or "進行中"
            self.trs.append(self.tr)
        dt = lambda j: self.d.at[j, "date"] if "date" in self.d else j
        # 波段層級警示（SOT、50%）
        for w in pivot_warnings(self.d, self.piv):
            self.events.append(dict(idx=w["idx"], confirm=w["confirm"], date=dt(w["idx"]),
                                    kind=w["kind"], price=float(self.d.at[w["idx"], "close"]), tr_id=-1, note=w["note"], page=w["page"]))
            if w["kind"] in ("SOT_up", "RETR_GT50") and self.d.at[w["confirm"], "regime"] == "up":
                self._sig(w["confirm"], "W_" + w["kind"], "warn", NaN, w["note"], w["page"], tr_id=-1)
        # v1.2：波段層級的努力與結果（放量滯漲／放量滯跌）
        for w in wave_events(self.d, self.piv):
            self.events.append(dict(idx=w["idx"], confirm=w["confirm"], date=dt(w["idx"]), kind=w["kind"],
                                    price=w["price"], tr_id=-1, note=w["note"], page=w["page"]))
            if w["kind"] == "WAVE_STALL_UP":
                self._sig(w["confirm"], "W_WAVE_STALL_UP", "warn", NaN, w["note"], w["page"], tr_id=-1)
        # v1.2：不依賴交易區間的進場（書中後記推薦的牛市回調、死角交易法）
        self._pullback_pass()
        self._corner_pass()
        return self

    def _record(self, i):
        tr = self.tr
        if tr is not None:
            self.bar_phase[i], self.bar_tr[i], self.bar_bias[i] = tr.phase, tr.id, tr.bias
            self.bar_sup[i], self.bar_res[i] = tr.support, tr.resistance
        elif self._last_resolved is not None:
            lr = self._last_resolved
            c = self.d.at[i, "close"]
            if (lr.result.startswith(("吸籌", "再吸籌")) and c > lr.support) or \
               (lr.result.startswith(("派發", "再派發")) and c < lr.resistance):
                self.bar_phase[i], self.bar_tr[i] = "E", lr.id
            else:
                self._last_resolved = None

    # ---------------- 輸出 ----------------
    EVENT_COLS = ["idx", "confirm", "date", "kind", "price", "tr_id", "note", "page"]
    SIGNAL_COLS = ["idx", "date", "kind", "side", "stop", "note", "page", "tr_id"]

    def events_df(self):
        e = pd.DataFrame(self.events, columns=self.EVENT_COLS if not self.events else None)
        return e.sort_values(["confirm", "idx"]).reset_index(drop=True) if len(e) else e

    def signals_df(self):
        s = pd.DataFrame(self.signals, columns=self.SIGNAL_COLS if not self.signals else None)
        return s.sort_values("idx").reset_index(drop=True) if len(s) else s

    def trs_df(self):
        rows = []
        for t in self.trs:
            rows.append(dict(tr_id=t.id, prior=t.prior, climax=t.climax_kind,
                             start=self.d.at[t.start, "date"] if "date" in self.d else t.start,
                             end=(self.d.at[t.end, "date"] if (t.end >= 0 and "date" in self.d) else t.end),
                             support=t.support, resistance=t.resistance, bias=round(t.bias, 2),
                             n_events=len(t.events), result=t.result))
        return pd.DataFrame(rows)

    def bars_df(self):
        out = self.d[["date", "open", "high", "low", "close", "volume", "rvol", "rspread", "close_pos", "regime"]].copy() \
            if "date" in self.d else self.d.copy()
        out["phase"], out["tr_id"], out["bias"] = self.bar_phase, self.bar_tr, self.bar_bias
        out["support"], out["resistance"] = self.bar_sup, self.bar_res
        ind = wyckoff_indicators(self.d, self.s)
        for k in ("ud_vol_ratio", "evr", "dry_up", "range_pos", "contraction", "dead_corner", "demand_score"):
            out[k] = ind[k]
        out["ch_dir"], out["ch_lower"], out["ch_upper"], out["ch_mid"] = \
            self.ch["dir"], self.ch["lower"], self.ch["upper"], self.ch["mid"]
        if self.wk is not None:
            for k in ("wk_regime", "wk_phase", "wk_bias", "wk_ok_long"):
                out[k] = self.wk[k].to_numpy()
        return out


# ---------------------------------------------------------------------------
# 簡化回測（依原書進出場與危機管理規則）
# ---------------------------------------------------------------------------
LONG_KINDS = ("L_SPRING", "L_TEST", "L_LPS", "L_BU", "L_PULLBACK", "L_CORNER")
EXIT_KINDS = ("X_ICE", "S_UT", "S_LPSY", "S_NODEMAND")
TIGHTEN_KINDS = ("W_SOT_up", "W_SOT_CH", "W_WAVE_STALL_UP")


def backtest(sc: WyckoffScanner, long_kinds=LONG_KINDS, exit_kinds=EXIT_KINDS, p: Optional[Params] = None,
             weekly_filter: Optional[bool] = None) -> pd.DataFrame:
    """一次只持有一筆多單。規則：
    - 訊號 K 線收盤確認，下一根開盤進場；開盤已跌破停損則放棄。
    - 停損：訊號提供的結構停損（Spring/測試低點、LPS 低點、JOC K 線低點）。
    - 進場後 confirm_bars 根內須出現「收盤突破訊號K高點且量 >= 均量」的跟隨，否則收盤出場〔p83、p316〕。
    - 確認期內出現放量陰線立即出場〔p83、p304〕。
    - 確認後停損上移至最近已確認的波段低點（收緊停損〔p17、p141〕）。
    - 出現破冰／UT 確認／LPSY 訊號，下一根開盤出場；搶購高潮則停損上移到 BC 低點。
    - 確認後若放量跌破需求線（X_TRENDLINE），下一根開盤出場〔p142、p146〕。
    - SOT／通道式 SOT／波段放量滯漲／回調超過 50% 的警示：停損縮緊到近 5 根最低點下〔p141〕；
      SOT 類警示與回調超過 50% 在 10 根內先後出現：下一根開盤出場〔p145–146〕。
    - 確認後出現放量滯漲（量增、振幅縮小）且次日量縮到 0.8 倍均量以下：當日收盤出場〔p59–60〕。
    - weekly_filter=True 時，週線仍在下跌且尚未形成區間的日子不接受多方訊號〔p147、p225〕。
    - 交易成本：買 0.1425%、賣 0.4425%（含證交稅）。
    """
    p = p or sc.p
    d, s = sc.d, sc.s
    wf = p.weekly_filter if weekly_filter is None else weekly_filter
    wk_ok = sc.wk["wk_ok_long"].to_numpy(bool) if (wf and sc.wk is not None) else None
    trend_exit = p.exit_on_demand_line_break
    sig = sc.signals_df()
    if sig.empty:
        return pd.DataFrame()
    by_idx = {}
    for r in sig.itertuples():
        by_idx.setdefault(r.idx, []).append(r)
    piv_lows = [x for x in sc.piv if x["kind"] == "L"]
    trades, pos, pending = [], None, None
    n = len(d)
    for i in range(1, n):
        o, h, l, c = d.at[i, "open"], d.at[i, "high"], d.at[i, "low"], d.at[i, "close"]
        if pos is None and pending is not None:
            if o > pending["stop"]:
                pos = dict(entry_i=i, entry=o, stop=pending["stop"], init_stop=pending["stop"], kind=pending["kind"],
                           sig_high=d.at[pending["idx"], "high"], confirmed=False, exit_next=False)
            pending = None
        if pos is not None:
            reason, px = None, None
            if pos["exit_next"]:
                reason, px = pos["exit_reason"], o
            elif l <= pos["stop"]:
                reason, px = "停損", min(o, pos["stop"])
            else:
                held = i - pos["entry_i"] + 1
                if not pos["confirmed"]:
                    if c > pos["sig_high"] and d.at[i, "rvol"] >= 1.0:
                        pos["confirmed"] = True
                    elif s.at[i, "supply_bar"]:
                        reason, px = "確認期放量陰線", c
                    elif held >= p.confirm_bars:
                        reason, px = "無跟隨（進場過早/判斷失誤）", c
                if pos["confirmed"] and reason is None:
                    lows = [x for x in piv_lows if x["confirm"] <= i and x["idx"] > pos["entry_i"]]
                    if lows:
                        pos["stop"] = max(pos["stop"], lows[-1]["price"] - p.stop_buffer_atr * d.at[i, "atr"])
                    # 放量滯漲後次日量縮：清倉的最後時機〔p59–60〕
                    if p.exit_on_stall and i - 1 > pos["entry_i"] and s.at[i - 1, "stop2_up"] and d.at[i, "rvol"] <= 0.8:
                        reason, px = "放量滯漲後量縮（清倉最後時機）", c
                if reason is None and held >= p.max_hold:
                    reason, px = "持有期滿", c
            if reason is None:
                for r in by_idx.get(i, []):
                    if r.kind in exit_kinds or (trend_exit and r.kind == "X_TRENDLINE" and pos["confirmed"]):
                        pos["exit_next"], pos["exit_reason"] = True, f"出場訊號 {r.kind}"
                    elif r.kind == "W_BC":
                        pos["stop"] = max(pos["stop"], d.at[i, "low"])
                    elif r.kind in TIGHTEN_KINDS or r.kind == "W_RETR_GT50":
                        if r.kind == "W_RETR_GT50":
                            pos["last_retr"] = i
                        else:
                            pos["last_sot"] = i
                        if p.tighten_on_sot:              # 縮緊停損〔p141〕
                            lo5 = float(d["low"].iloc[max(pos["entry_i"], i - 4):i + 1].min())
                            pos["stop"] = max(pos["stop"], lo5 - p.stop_buffer_atr * d.at[i, "atr"])
                if (p.exit_on_sot_retr and not pos["exit_next"] and "last_sot" in pos and "last_retr" in pos
                        and abs(pos["last_sot"] - pos["last_retr"]) <= 10):
                    pos["exit_next"], pos["exit_reason"] = True, "SOT 與回調超過 50% 先後出現（立刻離場）"
            if reason is not None:
                gross = px / pos["entry"] - 1
                net = (px * (1 - p.sell_cost)) / (pos["entry"] * (1 + p.buy_cost)) - 1
                risk = pos["entry"] - pos["init_stop"]
                atr_e = d.at[pos["entry_i"], "atr"]
                trades.append(dict(kind=pos["kind"], entry_date=d.at[pos["entry_i"], "date"], entry=pos["entry"],
                                   init_stop=pos["init_stop"], risk_atr=risk / atr_e if atr_e > 0 else NaN,
                                   exit_date=d.at[i, "date"], exit=px, bars=i - pos["entry_i"] + 1,
                                   ret=gross, net_ret=net, R=(px - pos["entry"]) / risk if risk > 0 else NaN,
                                   reason=reason))
                pos = None
        if pos is None and pending is None and (wk_ok is None or wk_ok[i]):
            for r in by_idx.get(i, []):
                if r.kind in long_kinds and not math.isnan(r.stop):
                    pending = dict(idx=i, stop=r.stop, kind=r.kind)
                    break
    if pos is not None:                                   # 資料結束時仍持有：以最後收盤價結算（標示為未平倉）
        i = n - 1
        px = d.at[i, "close"]
        risk = pos["entry"] - pos["init_stop"]
        atr_e = d.at[pos["entry_i"], "atr"]
        trades.append(dict(kind=pos["kind"], entry_date=d.at[pos["entry_i"], "date"], entry=pos["entry"],
                           init_stop=pos["init_stop"], risk_atr=risk / atr_e if atr_e > 0 else NaN,
                           exit_date=d.at[i, "date"], exit=px, bars=i - pos["entry_i"] + 1,
                           ret=px / pos["entry"] - 1,
                           net_ret=(px * (1 - p.sell_cost)) / (pos["entry"] * (1 + p.buy_cost)) - 1,
                           R=(px - pos["entry"]) / risk if risk > 0 else NaN, reason="資料結束（未平倉）"))
    return pd.DataFrame(trades)


def summarize_trades(t: pd.DataFrame) -> pd.DataFrame:
    if t.empty:
        return pd.DataFrame()
    def agg(g):
        wins = g["net_ret"] > 0
        gp, gl = g.loc[wins, "net_ret"].sum(), -g.loc[~wins, "net_ret"].sum()
        return pd.Series(dict(trades=len(g), win_rate=wins.mean(), avg_net=g["net_ret"].mean(),
                              avg_R=g["R"].mean(), profit_factor=(gp / gl) if gl > 0 else np.inf,
                              avg_bars=g["bars"].mean()))
    out = t.groupby("kind").apply(agg, include_groups=False)
    out.loc["ALL"] = agg(t)
    return out


def event_study(sc: WyckoffScanner, horizons=(5, 10, 20, 60)) -> pd.DataFrame:
    """事件確認後 N 根的報酬（以確認K收盤為基準），並附無條件基準。"""
    d = sc.d
    c = d["close"].to_numpy(float)
    e = sc.events_df()
    rows = []
    for kind, g in e.groupby("kind"):
        row = dict(kind=kind, n=len(g))
        for hz in horizons:
            r = [c[j + hz] / c[j] - 1 for j in g["confirm"] if j + hz < len(c)]
            row[f"fwd{hz}_mean"] = np.mean(r) if r else NaN
            row[f"fwd{hz}_win"] = np.mean(np.array(r) > 0) if r else NaN
        rows.append(row)
    base = dict(kind="(全部K線基準)", n=len(c))
    for hz in horizons:
        r = c[hz:] / c[:-hz] - 1 if len(c) > hz else np.array([])
        base[f"fwd{hz}_mean"] = r.mean() if len(r) else NaN
        base[f"fwd{hz}_win"] = (r > 0).mean() if len(r) else NaN
    rows.append(base)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 選股：相對強弱〔p253–258〕；部位：固定風險比例
# ---------------------------------------------------------------------------
def relative_strength(stock: pd.DataFrame, index: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """原書的相對強弱選股〔p253–258〕：
    ① 大盤創新低時，個股沒有創新低，且堅守在支撐上、回調未過 50%（rs_divergence）〔p253、p256 的 4/4a〕
    ② 個股回調幅度明顯淺於大盤（例：大盤回落近 50%、個股僅 30%）（rs_shallower）〔p256 的 2/2a〕
    ③ 大盤回升時個股漲幅更大：自大盤近 lookback 日最低點起算，個股報酬減大盤報酬（rs_out）〔p253〕
    rs_strong＝近 lookback 日內出現過 ① 或 ②，且 ③ > 0、個股回調仍未過 50%。只用當日以前的資料。"""
    m = stock[["date", "high", "low", "close"]].merge(
        index[["date", "high", "low", "close"]], on="date", suffixes=("", "_idx")).reset_index(drop=True)
    mp = max(5, lookback // 2)
    m["rs_line"] = m["close"] / m["close_idx"]
    m["idx_new_low"] = m["low_idx"] <= m["low_idx"].rolling(lookback, min_periods=mp).min()
    m["stk_new_low"] = m["low"] <= m["low"].rolling(lookback, min_periods=mp).min()
    for pre in ("", "_idx"):
        hh = m["high" + pre].rolling(lookback, min_periods=mp).max()
        ll = m["low" + pre].rolling(lookback, min_periods=mp).min()
        m["retr" + pre] = (hh - m["close" + pre]) / (hh - ll).replace(0, np.nan)
    m["rs_divergence"] = m["idx_new_low"] & ~m["stk_new_low"] & (m["retr"] < 0.5)
    m["rs_shallower"] = (m["retr_idx"] - m["retr"]) >= 0.15
    lo_idx = m["low_idx"].to_numpy(float)
    cs, ci = m["close"].to_numpy(float), m["close_idx"].to_numpy(float)
    out = np.full(len(m), NaN)
    for t in range(len(m)):
        a = max(0, t - lookback + 1)
        j = a + int(np.argmin(lo_idx[a:t + 1]))
        out[t] = (cs[t] / cs[j] - 1) - (ci[t] / ci[j] - 1)
    m["rs_out"] = out
    m["rs_score"] = (m["rs_divergence"] | m["rs_shallower"]).rolling(lookback, min_periods=1).sum()
    m["rs_strong"] = (m["rs_score"] >= 1) & (m["rs_out"] > 0) & (m["retr"] < 0.5)
    return m


def position_size(equity: float, entry: float, stop: float, risk_pct: float = 0.01, lot_size: int = 1000,
                  allow_odd_lot: bool = True, max_position_pct: float = 0.25, p: Params = Params()) -> dict:
    """依結構停損計算股數【延伸：固定風險比例法】。
    原書精神：低風險進場、停損時虧損最小〔p6、p27〕；先輕倉、需求跟隨再加碼〔p32、p288〕。
    每股風險含來回交易成本；部位市值不超過淨值的 max_position_pct；台股一張 1,000 股，可選擇零股。"""
    if not (equity > 0 and entry > stop > 0 and 0 < risk_pct < 1):
        return dict(shares=0, lots=0, odd_shares=0, risk_per_share=NaN, risk_amount=0.0, position_value=0.0,
                    note="參數不合理：需 淨值>0、進場價>停損價>0、0<風險比例<1")
    rps = (entry - stop) + entry * p.buy_cost + stop * p.sell_cost
    raw = int((equity * risk_pct) // rps)
    raw = min(raw, int((equity * max_position_pct) // entry))
    lots = raw // lot_size
    odd = raw - lots * lot_size if allow_odd_lot else 0
    shares = lots * lot_size + odd
    return dict(shares=shares, lots=lots, odd_shares=odd, risk_per_share=round(rps, 4),
                risk_amount=round(shares * rps, 2), position_value=round(shares * entry, 2),
                risk_pct_of_equity=round(shares * rps / equity, 4),
                note=f"單筆風險 {risk_pct:.1%}、部位上限 {max_position_pct:.0%}；分批時先買 1/3，需求跟隨再加碼〔p32、p288〕")


# ---------------------------------------------------------------------------
# StockExpert：診斷卡與交易紀錄檢討
# ---------------------------------------------------------------------------
def snapshot(sc: WyckoffScanner, lookback: int = 20, symbol: Optional[str] = None,
             index_df: Optional[pd.DataFrame] = None) -> dict:
    """最新一根 K 線的威科夫診斷卡（可直接轉 JSON 給前端）。"""
    d = sc.d
    i = len(d) - 1
    tr = sc.tr
    ev = sc.events_df()
    recent = ev[ev["confirm"] >= i - lookback].tail(8) if len(ev) else ev
    sig = sc.signals_df()
    rsig = sig[sig["idx"] >= i - lookback] if len(sig) else sig
    ind = wyckoff_indicators(d, sc.s).iloc[i]
    num = lambda x: None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 3)
    card = dict(
        symbol=symbol,
        date=str(d.at[i, "date"])[:10] if "date" in d else i,
        close=float(d.at[i, "close"]),
        background=dict(up="上升背景", down="下降背景", range="區間/不明")[d.at[i, "regime"]],
        phase=sc.bar_phase[i] or "無有效區間",
        tr=None if tr is None else dict(id=tr.id, prior=tr.prior, climax=tr.climax_kind, support=tr.support,
                                       resistance=tr.resistance, bias=round(tr.bias, 2),
                                       lean=("偏吸籌/再吸籌" if tr.bias >= 3 else "偏派發/再派發" if tr.bias <= -3 else "未定")),
        recent_events=[dict(date=str(r.date)[:10], kind=r.kind, note=r.note, page=r.page) for r in recent.itertuples()],
        recent_signals=[dict(date=str(r.date)[:10], kind=r.kind, stop=None if math.isnan(r.stop) else round(r.stop, 2),
                             note=r.note, page=r.page) for r in rsig.itertuples()],
        pending=[dict(kind=st["kind"], since=str(d.at[st["start"], "date"])[:10] if "date" in d else st["start"])
                 for st in sc.setups],
        indicators=dict(ud_vol_ratio=num(ind["ud_vol_ratio"]), demand_score=num(ind["demand_score"]),
                        dry_up=num(ind["dry_up"]), range_pos=num(ind["range_pos"]),
                        dead_corner=bool(ind["dead_corner"])),
        channel=None,
        weekly=None,
        cause_target=cause_target(sc),
        rule_reminders=[
            "震盪區內部交易風險最大，耐心等右手邊出現 SOS 或 SOW〔p93、p153、p191〕",
            "進場後 1~5 根內沒有價量齊揚的跟隨，就無條件出場〔p316〕",
        ],
    )
    cd = sc.ch["dir"][i]
    if cd:
        lo, up = float(sc.ch["lower"][i]), float(sc.ch["upper"][i])
        pos = (float(d.at[i, "close"]) - lo) / (up - lo) if up > lo else NaN
        card["channel"] = dict(direction="上升通道" if cd == "up" else "下降通道",
                               lower=round(lo, 2), upper=round(up, 2), mid=round((lo + up) / 2, 2), position=num(pos),
                               lower_name="需求線" if cd == "up" else "超賣線",
                               upper_name="超買線" if cd == "up" else "供應線")
    if sc.wk is not None:
        w = sc.wk.iloc[i]
        card["weekly"] = dict(regime=dict(up="上升", down="下降", range="區間/不明").get(w["wk_regime"], "資料不足"),
                              phase=w["wk_phase"] or "無有效區間", bias=num(w["wk_bias"]), ok_long=bool(w["wk_ok_long"]))
        if not bool(w["wk_ok_long"]):
            card["rule_reminders"].append("週線仍在下跌且尚未形成區間：熊市中不能隨意抄底〔p225〕")
    if index_df is not None and "date" in d:
        rs = relative_strength(d[["date", "high", "low", "close"]], index_df)
        if len(rs):
            r = rs.iloc[-1]
            card["relative_strength"] = dict(rs_strong=bool(r["rs_strong"]), rs_score=num(r["rs_score"]),
                                             stock_retr=num(r["retr"]), index_retr=num(r["retr_idx"]),
                                             outperformance=num(r["rs_out"]), page="p253-258")
    return card


def audit_trades(sc: WyckoffScanner, trades: pd.DataFrame) -> pd.DataFrame:
    """以威科夫脈絡檢討使用者買賣紀錄。trades 需含欄位：date、side（buy/sell 或 買/賣）、price。"""
    d = sc.d
    ev = sc.events_df()
    dates = pd.to_datetime(d["date"]).to_numpy()
    out = []
    for r in trades.itertuples():
        t = np.datetime64(pd.Timestamp(r.date))
        i = int(np.searchsorted(dates, t, side="right") - 1)
        if i < 0:
            continue
        side = "buy" if str(r.side).lower() in ("buy", "b", "買", "買進", "買入") else "sell"
        near = ev[(ev["confirm"] <= i) & (ev["confirm"] >= i - 5)] if len(ev) else ev
        kinds = set(near["kind"]) if len(near) else set()
        just = set(ev.loc[(ev["confirm"] <= i) & (ev["confirm"] >= i - 2), "kind"]) if len(ev) else set()
        joc_today = bool(len(ev)) and bool(((ev["kind"] == "JOC") & (ev["confirm"] == i)).any())
        ph = sc.bar_phase[i]
        flags = []
        if side == "buy":
            if (just & {"Spring", "Test", "LPS", "BU"}) and not joc_today:
                flags.append("✔ 買在原書進場點（Spring/測試/LPS/回測後 0~2 根）")
            if just & {"PB", "CORNER"}:
                flags.append("✔ 買在牛市回調的停止行為或死角（原書後記推薦的牛市回調〔p316〕、死角交易法〔p114〕）")
            if ph == "B":
                flags.append("✘ 在震盪區內部買進（原書：風險最大〔p93〕）")
            if joc_today:
                flags.append("✘ 在 JOC 當天追價（原書：屬追高，應等縮量回測〔p202〕）")
            if kinds & {"BC", "PSY"}:
                flags.append("✘ 搶購高潮附近買進（原書：危險、應等派發確認〔p116-117〕）")
            if kinds & {"SOW", "ICE", "LPSY", "UT確認", "NoDemandRally"}:
                flags.append("✘ 供應控制背景下買進（原書：抢反弹是逆勢〔p147〕）")
            if d.at[i, "regime"] == "down" and ph not in ("B", "C", "D", "E"):
                flags.append("△ 下降背景且尚未出現 SC/ST 構成的區間（原書：熊市不隨意抄底〔p225〕）")
            if sc.wk is not None and not bool(sc.wk.at[i, "wk_ok_long"]):
                flags.append("△ 週線仍在下跌且尚未形成區間（逆大背景〔p147〕）")
            if "DEMAND_LINE_BREAK" in kinds:
                flags.append("✘ 剛放量跌破需求線後買進（原書：供應超過需求〔p142、p146〕）")
        else:
            if kinds & {"SC", "Spring", "Shakeout", "Test"}:
                flags.append("✘ 賣在恐慌拋售/震倉附近（原書：公眾在底部割肉〔p72、p131〕）")
            if kinds & {"UT確認", "UTAD", "SOW", "ICE", "LPSY", "BC"}:
                flags.append("✔ 在派發/弱勢訊號出場")
        out.append(dict(date=str(d.at[i, "date"])[:10], side=side, price=r.price, phase=ph or "-",
                        regime=d.at[i, "regime"], nearby_events=",".join(sorted(kinds)), review="；".join(flags) or "無明確對應"))
    return pd.DataFrame(out)
