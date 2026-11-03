# -*- coding: utf-8 -*-
"""
程式邏輯自我測試（合成資料）
============================
注意：本檔的價格序列是「依原書描述刻意構造」的合成資料，只用來檢查程式是否把
書中定義的價量型態辨識正確（單元測試），**不代表任何市場績效或策略有效性**。
真實驗證請用 run_validation.py 搭配台股 CSV。

執行：python3 -m unittest -v test_engine.py
"""
import unittest

import numpy as np
import pandas as pd

from wyckoff_engine import (Params, WyckoffScanner, add_features, bar_signals, backtest,
                            snapshot, audit_trades, load_ohlcv_csv, summarize_trades, event_study,
                            zigzag, channels, to_weekly, weekly_context, pnf_columns, cause_target)


class Builder:
    """逐根構造 K 線：給定收盤價、振幅(%)、收盤位置、成交量。"""

    def __init__(self, start=100.0):
        self.rows, self.last = [], start
        self.date = pd.Timestamp("2020-01-02")
        self.mark = {}

    def bar(self, close, spread_pct, close_pos, vol, low=None, high=None):
        sp = close * spread_pct
        lo = close - close_pos * sp if low is None else low
        hi = lo + sp if high is None else high
        o = min(max(self.last, lo), hi)
        self.rows.append(dict(date=self.date, open=o, high=max(hi, o, close), low=min(lo, o, close), close=close, volume=vol))
        self.last = close
        self.date += pd.offsets.BDay(1)
        return len(self.rows) - 1

    def path(self, n, end, spread_pct, close_pos, vols, wiggle=0.0, period=10):
        start = self.last
        for k in range(1, n + 1):
            base = start * (end / start) ** (k / n)
            c = base * (1 + wiggle * np.sin(2 * np.pi * k / period))
            v = vols[k - 1] if hasattr(vols, "__len__") else vols
            self.bar(c, spread_pct, close_pos, v)

    def df(self):
        return pd.DataFrame(self.rows)


def make_cycle():
    """暖身 → 下跌 → PS → SC → AR → ST → Phase B → Spring → SOS/JOC → BU → 上漲
       → 急漲 → BC → AR → ST → Phase B → UTAD → 確認 → SOW → 破冰 → LPSY → 下跌"""
    b = Builder(100.0)
    rng = np.random.default_rng(7)
    b.path(80, 110, 0.02, 0.55, list(1e6 * (1 + 0.05 * rng.standard_normal(80))), wiggle=0.01)
    b.path(54, 80, 0.02, 0.45, 1e6)
    b.mark["PS"] = b.bar(b.last * 0.99, 0.03, 0.75, 1.9e6)
    b.path(10, b.last * 0.92, 0.02, 0.4, 1e6)
    b.path(5, b.last * 0.91, 0.025, 0.35, [1.2e6, 1.3e6, 1.35e6, 1.4e6, 1.45e6])
    b.mark["SC"] = b.bar(b.last * 0.96, 0.065, 0.45, 3.6e6)
    sc_low = b.rows[-1]["low"]
    b.path(7, b.last * 1.13, 0.025, 0.7, [1.6e6, 1.4e6, 1.2e6, 1.1e6, 1.0e6, 0.9e6, 0.9e6])
    ar_high = max(r["high"] for r in b.rows[b.mark["SC"]:])
    b.path(8, sc_low * 1.02, 0.013, 0.45, [0.6e6, 0.58e6, 0.56e6, 0.55e6, 0.54e6, 0.53e6, 0.52e6, 0.5e6])
    b.mark["ST"] = len(b.rows) - 1
    sup, res = sc_low, ar_high
    height = res - sup
    # Phase B：區間內震盪，量能逐漸萎縮（嚴格遞減，避免誤判）
    for k in range(1, 46):
        c = sup + height * (0.5 + 0.32 * np.sin(2 * np.pi * k / 15 - np.pi / 2))
        b.bar(c, 0.016, 0.5, 0.9e6 - k * 6000)
    # 逐步接近支撐
    for f in (0.30, 0.18, 0.10, 0.05):
        b.bar(sup + height * f, 0.014, 0.45, 0.62e6 - f * 1e5)
    # Spring A：小幅跌破支撐、縮量縮幅、當根收回
    b.mark["Spring"] = b.bar(sup * 1.004, 0.0, 0.5, 0.5e6, low=sup * 0.993, high=sup * 1.006)
    b.bar(sup * 1.016, 0.014, 0.65, 0.7e6)
    b.bar(sup * 1.031, 0.016, 0.7, 0.85e6)
    b.mark["SOS"] = b.bar(sup + height * 0.62, 0.035, 0.8, 1.7e6)
    b.bar(res * 0.985, 0.035, 0.95, 1.9e6)
    b.mark["JOC"] = b.bar(res * 1.02, 0.04, 0.85, 2.3e6)
    for f, v in ((1.014, 0.6e6), (1.009, 0.55e6), (1.006, 0.52e6), (1.004, 0.5e6)):
        b.bar(res * f, 0.012, 0.45, v)
    b.mark["BU_trigger"] = b.bar(res * 1.03, 0.02, 0.8, 1.0e6)
    b.path(60, b.last * 1.5, 0.02, 0.6, 1.05e6, wiggle=0.004, period=12)
    b.path(8, b.last * 1.22, 0.03, 0.75, [1.25e6, 1.3e6, 1.35e6, 1.4e6, 1.45e6, 1.5e6, 1.55e6, 1.6e6])
    b.mark["BC"] = b.bar(b.last * 1.04, 0.065, 0.6, 3.8e6)
    bc_high = b.rows[-1]["high"]
    b.path(7, b.last * 0.89, 0.025, 0.3, [1.7e6, 1.5e6, 1.3e6, 1.2e6, 1.1e6, 1.0e6, 1.0e6])
    ar_low = min(r["low"] for r in b.rows[b.mark["BC"]:])
    b.path(8, bc_high * 0.983, 0.015, 0.55, [0.65e6] * 8)
    sup2, res2 = ar_low, bc_high
    h2 = res2 - sup2
    for k in range(1, 36):
        c = sup2 + h2 * (0.5 + 0.30 * np.sin(2 * np.pi * k / 12 + np.pi / 2))
        b.bar(c, 0.016, 0.5, 1.0e6 - k * 5000)
    for f in (0.70, 0.82, 0.90):
        b.bar(sup2 + h2 * f, 0.015, 0.6, 0.8e6)
    b.mark["UTAD"] = b.bar(res2 * 0.992, 0.0, 0.3, 1.7e6, low=res2 * 0.975, high=res2 * 1.012)
    b.mark["UT_follow"] = b.bar(sup2 + h2 * 0.56, 0.03, 0.2, 1.9e6)
    b.mark["SOW"] = b.bar(sup2 + h2 * 0.18, 0.035, 0.15, 2.2e6)
    b.bar(sup2 * 1.01, 0.03, 0.2, 2.3e6)
    b.mark["ICE_bar"] = b.bar(sup2 * 0.972, 0.03, 0.2, 2.4e6)
    for f, v in ((0.978, 0.7e6), (0.983, 0.62e6), (0.987, 0.58e6), (0.99, 0.55e6)):
        b.bar(sup2 * f, 0.012, 0.6, v)
    b.mark["LPSY_trigger"] = b.bar(sup2 * 0.965, 0.025, 0.2, 1.3e6)
    b.path(40, b.last * 0.75, 0.02, 0.4, 1.2e6)
    return b.df(), b.mark


class TestBookRules(unittest.TestCase):
    def test_p57_supply_expansion(self):
        df = pd.DataFrame(dict(date=pd.date_range("2024-01-01", periods=3),
                               open=[10, 9.8, 9.6], high=[10.2, 10.0, 9.7], low=[9.7, 9.5, 9.3],
                               close=[9.9, 9.6, 9.4], volume=[100, 100, 90]))
        s = bar_signals(add_features(df))
        # 第 2 根：高低皆低於昨日、收低於昨日、量 >= 昨日 → 成立（p57）；第 3 根量縮 → 不成立
        self.assertEqual(list(s["supply_exp_p57"]), [False, True, False])

    def test_effort_no_result(self):
        df = pd.DataFrame(dict(date=pd.date_range("2024-01-01", periods=25),
                               open=np.linspace(20, 16, 25), high=np.linspace(20.3, 16.3, 25),
                               low=np.linspace(19.6, 15.6, 25), close=np.linspace(19.8, 15.8, 25),
                               volume=[100.0] * 25))
        df.loc[24, ["open", "high", "low", "close", "volume"]] = [15.8, 15.85, 15.7, 15.8, 180.0]
        s = bar_signals(add_features(df))
        self.assertTrue(s.at[24, "stop2_down"])        # 放量滯跌＝第二種停止行為（p80）


class TestCycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.mark = make_cycle()
        cls.sc = WyckoffScanner(cls.df, Params()).run()
        cls.ev = cls.sc.events_df()
        cls.sig = cls.sc.signals_df()

    def _near(self, kind, idx, tol=2, col="idx"):
        e = self.ev[self.ev["kind"] == kind]
        return bool(((e[col] - idx).abs() <= tol).any())

    def test_accumulation_events(self):
        m = self.mark
        self.assertTrue(self._near("SC", m["SC"]), "SC 未偵測")
        self.assertTrue((self.ev["kind"] == "AR").any(), "AR 未偵測")
        self.assertTrue((self.ev["kind"] == "ST").any(), "ST 未偵測")
        self.assertTrue(self._near("Spring", m["Spring"]), "Spring 未偵測")
        self.assertTrue(self._near("SOS", m["SOS"], 1), "SOS 未偵測")
        self.assertTrue(self._near("JOC", m["JOC"], 0), "JOC 未偵測")
        self.assertTrue(self._near("BU", m["BU_trigger"], 0, col="confirm"), "BU 未偵測")

    def test_distribution_events(self):
        m = self.mark
        self.assertTrue(self._near("BC", m["BC"]), "BC 未偵測")
        self.assertTrue((self.ev["kind"] == "AR(自然回落)").any())
        self.assertTrue((self.ev["kind"] == "ST(派發)").any())
        self.assertTrue(self._near("UTAD", m["UTAD"], 0), "UTAD 未偵測")
        self.assertTrue(self._near("UT確認", m["UT_follow"], 0), "UT 確認未偵測")
        self.assertTrue(self._near("SOW", m["SOW"], 0), "SOW 未偵測")
        self.assertTrue((self.ev["kind"] == "ICE").any(), "破冰未偵測")
        self.assertTrue(self._near("LPSY", m["LPSY_trigger"], 0, col="confirm"), "LPSY 未偵測")

    def test_event_order(self):
        order = ["SC", "AR", "ST", "Spring", "SOS", "JOC", "BU", "BC", "UTAD", "SOW", "ICE", "LPSY"]
        first = [int(self.ev.loc[self.ev["kind"] == k, "confirm"].min()) for k in order]
        self.assertEqual(first, sorted(first), f"事件順序錯誤：{dict(zip(order, first))}")

    def test_tr_classification(self):
        res = list(self.sc.trs_df()["result"])
        self.assertIn("吸籌（Accumulation）", res)
        self.assertIn("派發（Distribution）", res)

    def test_signals_and_stops(self):
        s = self.sig
        sp = s[s["kind"] == "L_SPRING"]
        self.assertFalse(sp.empty, "缺少 Spring 進場訊號")
        spring_low = self.df.at[self.mark["Spring"], "low"]
        self.assertTrue((sp["stop"] < spring_low).all(), "Spring 停損應在跌破最低點之下（p224）")
        self.assertFalse(s[s["kind"] == "L_BU"].empty, "缺少 JOC 回測進場訊號")
        self.assertFalse(s[s["kind"] == "S_UT"].empty, "缺少 UT 確認出場訊號")
        self.assertFalse(s[s["kind"] == "S_LPSY"].empty, "缺少 LPSY 訊號")
        self.assertFalse(s[s["kind"] == "W_BC"].empty, "缺少搶購高潮警示")

    def test_no_lookahead_signals(self):
        # 訊號只能出現在事件確認之後（confirm <= signal idx）
        for r in self.sig.itertuples():
            self.assertLess(r.idx, len(self.df))
        self.assertTrue((self.ev["confirm"] >= self.ev["idx"]).all())

    def test_phase_labels(self):
        ph = self.sc.bars_df()["phase"]
        for lab in ("A", "B", "C", "D", "E"):
            self.assertTrue((ph == lab).any(), f"缺少 Phase {lab}")

    def test_pipeline_runs(self):
        t = backtest(self.sc)
        self.assertFalse(t.empty)
        summarize_trades(t)
        event_study(self.sc)
        card = snapshot(self.sc)
        self.assertIn("phase", card)
        trades = pd.DataFrame(dict(date=[self.df.at[self.mark["JOC"], "date"], self.df.at[self.mark["BU_trigger"] + 1, "date"]],
                                   side=["買", "買"], price=[1, 1]))
        a = audit_trades(self.sc, trades)
        self.assertIn("追高", a.at[0, "review"])


def random_walk(seed, n=1500):
    rng = np.random.default_rng(seed)
    r = rng.standard_t(4, n) * 0.015
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[100, c[:-1]] * (1 + rng.normal(0, 0.003, n))
    sp = np.abs(rng.normal(0.02, 0.008, n)) * c
    h = np.maximum(o, c) + sp * rng.uniform(0, 0.5, n)
    l = np.minimum(o, c) - sp * rng.uniform(0, 0.5, n)
    v = 1e6 * np.exp(rng.normal(0, 0.45, n)) * (1 + 4 * np.abs(r))
    return pd.DataFrame(dict(date=pd.bdate_range("2010-01-04", periods=n), open=o, high=h, low=l, close=c, volume=v))


class TestNoiseSanity(unittest.TestCase):
    """隨機漫步沒有可預測性：若回測期望值顯著為正，代表程式有未來函數。"""

    def test_random_walk_no_edge(self):
        nets, n_sig, n_bars = [], 0, 0
        for seed in range(30):
            df = random_walk(seed)
            sc = WyckoffScanner(df).run()
            t = backtest(sc)
            s = sc.signals_df()
            n_sig += int(s["kind"].isin(["L_SPRING", "L_TEST", "L_LPS", "L_BU"]).sum()) if len(s) else 0
            n_bars += len(df)
            if len(t):
                nets += list(t["net_ret"])
        nets = np.array(nets)
        self.__class__.diag = dict(trades=len(nets), mean=float(nets.mean()) if len(nets) else 0.0,
                                   se=float(nets.std(ddof=1) / np.sqrt(len(nets))) if len(nets) > 1 else 0.0,
                                   long_signals_per_1000_bars=1000 * n_sig / n_bars)
        print("\n[隨機漫步診斷]", self.diag)
        if len(nets) > 10:
            self.assertLess(nets.mean(), 3 * nets.std(ddof=1) / np.sqrt(len(nets)) + 0.002)

    def test_indicators_range(self):
        df = random_walk(99, 600)
        sc = WyckoffScanner(df).run()
        b = sc.bars_df()
        ds = b["demand_score"].dropna()
        self.assertTrue(((ds >= 0) & (ds <= 100)).all())
        self.assertTrue(b["ud_vol_ratio"].dropna().gt(0).all())


def make_channel_series():
    """暖身 → 帶波動的上升通道 → 放量跌破需求線 → 帶波動且振幅漸增的下降通道 → 放量突破供應線"""
    b = Builder(100.0)
    b.path(60, 100.0, 0.02, 0.5, 1e6, wiggle=0.005)
    m = {}
    start = b.last
    for k in range(1, 97):                                    # 上升通道：線性上漲 + 正弦波動
        c = start * (1 + 0.004 * k) * (1 + 0.035 * np.sin(2 * np.pi * k / 16))
        b.bar(c, 0.02, 0.5, 1e6)
    m["up_end"] = len(b.rows) - 1
    for f in (0.96, 0.955, 0.95):                              # 放量跌破
        b.bar(b.last * f, 0.04, 0.15, 2.6e6)
    m["break_dn"] = (m["up_end"] + 1, len(b.rows) - 1)
    start = b.last
    for k in range(1, 97):                                    # 下降通道：振幅漸增，後面的低點會跌破超賣線
        amp = 0.03 + 0.0004 * k
        c = start * (1 - 0.004 * k) * (1 + amp * np.sin(2 * np.pi * k / 16))
        b.bar(c, 0.02, 0.5, 1e6)
    m["down_end"] = len(b.rows) - 1
    for f in (1.05, 1.05, 1.05):                               # 放量突破
        b.bar(b.last * f, 0.045, 0.85, 2.6e6)
    m["break_up"] = (m["down_end"] + 1, len(b.rows) - 1)
    b.path(30, b.last * 1.05, 0.02, 0.5, 1.1e6)
    return b.df(), m


class TestChannels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.m = make_channel_series()
        d = add_features(cls.df)
        cls.ch = channels(d, zigzag(d, 2.0))
        cls.ev = pd.DataFrame(cls.ch["events"])

    def _has(self, kind, rng, strong=None):
        e = self.ev[(self.ev["kind"] == kind) & self.ev["confirm"].between(*rng)]
        if strong is not None:
            e = e[e["strong"] == strong]
        return not e.empty

    def test_up_channel_and_demand_line_break(self):
        a, z = self.m["break_dn"]
        self.assertTrue((self.ch["dir"][self.m["up_end"] - 5:self.m["up_end"]] == "up").all(), "未形成上升通道")
        self.assertTrue(self._has("DEMAND_LINE_BREAK", (a, z + 2), strong=True), "放量跌破需求線未偵測")

    def test_down_channel_oversold_and_supply_break(self):
        a, z = self.m["break_up"]
        self.assertTrue(self._has("OVERSOLD", (self.m["break_dn"][1], self.m["down_end"])), "觸及超賣線未偵測")
        self.assertTrue(self._has("SUPPLY_LINE_BREAK", (a, z + 2), strong=True), "放量突破供應線未偵測")

    def test_trendline_exit_signal(self):
        sc = WyckoffScanner(self.df).run()
        s = sc.signals_df()
        a, z = self.m["break_dn"]
        self.assertTrue(((s["kind"] == "X_TRENDLINE") & s["idx"].between(a, z + 2)).any(), "缺少 X_TRENDLINE 出場訊號")
        card = snapshot(sc)
        self.assertIn("channel", card)


class TestCausality(unittest.TestCase):
    """把資料截在第 t 根重跑：t-1 之前確認的事件與訊號必須和完整資料完全相同（無未來函數）。"""

    def _key_events(self, sc, upto):
        e = sc.events_df()
        e = e[e["confirm"] < upto]
        return sorted((r.kind, int(r.idx), int(r.confirm), round(float(r.price), 6)) for r in e.itertuples())

    def _key_signals(self, sc, upto):
        s = sc.signals_df()
        if s.empty:
            return []
        s = s[s["idx"] < upto]
        return sorted((r.kind, int(r.idx), round(float(r.stop), 6) if r.stop == r.stop else None) for r in s.itertuples())

    def test_truncation_invariance(self):
        for df in (make_cycle()[0], random_walk(11, 900)):
            full = WyckoffScanner(df).run()
            for t in (len(df) // 2, int(len(df) * 0.75), len(df) - 15):
                part = WyckoffScanner(df.iloc[:t].reset_index(drop=True)).run()
                self.assertEqual(self._key_events(full, t - 1), self._key_events(part, t - 1), f"事件在 t={t} 不一致")
                self.assertEqual(self._key_signals(full, t - 1), self._key_signals(part, t - 1), f"訊號在 t={t} 不一致")
                np.testing.assert_array_equal(full.ch["lower"][:t], part.ch["lower"][:t])
                self.assertTrue((full.wk["wk_ok_long"].iloc[:t - 1].to_numpy() == part.wk["wk_ok_long"].iloc[:t - 1].to_numpy()).all())


class TestWeeklyAndPnf(unittest.TestCase):
    def test_to_weekly_and_mapping(self):
        dates = pd.bdate_range("2024-01-01", periods=10)            # 週一～週五 ×2
        df = pd.DataFrame(dict(date=dates, open=np.arange(10) + 10.0, high=np.arange(10) + 11.0,
                               low=np.arange(10) + 9.0, close=np.arange(10) + 10.5, volume=[100.0] * 10))
        w = to_weekly(df)
        self.assertEqual(len(w), 2)
        self.assertEqual(w.at[0, "open"], 10.0)
        self.assertEqual(w.at[0, "close"], 14.5)
        self.assertEqual(w.at[0, "high"], 15.0)
        self.assertEqual(w.at[0, "volume"], 500.0)
        self.assertEqual(str(w.at[0, "date"])[:10], "2024-01-05")

    def test_weekly_context_no_lookahead_alignment(self):
        df = random_walk(3, 700)
        wk, wsc = weekly_context(add_features(df))
        self.assertEqual(len(wk), len(df))
        wb = wsc.bars_df()
        # 週三（未完成的週）只能用上一個已完成週的狀態
        d = pd.to_datetime(df["date"])
        i = int(np.where(d.dt.weekday == 2)[0][60])
        prev_week_end = wb["date"][pd.to_datetime(wb["date"]) < d[i]].max()
        expect = wb.loc[pd.to_datetime(wb["date"]) == prev_week_end, "phase"].iloc[0]
        self.assertEqual(wk.at[i, "wk_phase"], expect)

    def test_pnf_columns(self):
        cols = pnf_columns([10, 11, 12, 13, 12, 11, 10, 11, 12, 13, 14], box=1.0, reversal=3)
        self.assertEqual([c["dir"] for c in cols], ["X", "O", "X"])
        self.assertEqual(cols[0]["top"], 13.0)
        self.assertEqual(cols[2]["top"], 14.0)

    def test_cause_target(self):
        df, m = make_cycle()
        sc = WyckoffScanner(df).run()
        acc = [t for t in sc.trs if t.result.startswith("吸籌")][0]
        ct = cause_target(sc, acc)
        self.assertGreater(ct["columns"], 0)
        self.assertGreater(ct["bull_from_support"], acc.support)


class TestCsvLoader(unittest.TestCase):
    def test_twse_format(self):
        import tempfile, os
        txt = ('"日期","成交股數","成交金額","開盤價","最高價","最低價","收盤價","漲跌價差","成交筆數"\n'
               '"113/01/02","20,000,000","1,000","590.00","593.00","589.00","593.00","+0.00","10"\n'
               '"113/01/03","25,000,000","1,000","584.00","585.00","576.00","578.00","-15.00","10"\n')
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(txt)
        out = load_ohlcv_csv(f.name)
        os.unlink(f.name)
        g = list(out.values())[0]
        self.assertEqual(str(g.at[0, "date"])[:10], "2024-01-02")
        self.assertEqual(g.at[1, "volume"], 25_000_000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
