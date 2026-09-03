"""
Unit tests for tcs_options_pricing.py

Run:  python -m unittest test_pricing.py -v

These replace the previous "read the printed console output and eyeball it"
verification with actual automated checks: binomial->BS convergence,
put-call parity, IV solver round-trip accuracy, and the expiry-date guard.
"""

import unittest
from datetime import date

from tcs_options_pricing import (
    binomial_price,
    black_scholes,
    bs_greeks,
    implied_volatility,
    compute_monthly_expiry,
    resolve_expiry,
    select_atm_strike,
    load_option_chain,
    compute_skew_stats,
    unique_path,
    is_nse_trading_holiday,
    roll_back_to_trading_day,
    log_run,
)


class TestBinomialConvergence(unittest.TestCase):
    def test_converges_to_black_scholes(self):
        S, K, r, sigma, T = 2264.0, 2264.0, 0.0692, 0.2188, 12 / 365
        bs_price = black_scholes(S, K, r, sigma, T, "call")
        binom_price = binomial_price(S, K, r, sigma, T, n=500, option_type="call")
        self.assertAlmostEqual(bs_price, binom_price, delta=0.05)

    def test_converges_with_dividend_yield(self):
        S, K, r, q, sigma, T = 2264.0, 2264.0, 0.0692, 0.015, 0.2188, 30 / 365
        bs_price = black_scholes(S, K, r, sigma, T, "call", q)
        binom_price = binomial_price(S, K, r, sigma, T, n=500, option_type="call", q=q)
        self.assertAlmostEqual(bs_price, binom_price, delta=0.05)

    def test_low_step_count_still_in_right_ballpark(self):
        S, K, r, sigma, T = 2264.0, 2264.0, 0.0692, 0.2188, 12 / 365
        bs_price = black_scholes(S, K, r, sigma, T, "call")
        binom_price = binomial_price(S, K, r, sigma, T, n=10, option_type="call")
        self.assertAlmostEqual(bs_price, binom_price, delta=5.0)


class TestPutCallParity(unittest.TestCase):
    def test_black_scholes_parity(self):
        import numpy as np
        S, K, r, sigma, T = 2264.0, 2264.0, 0.0692, 0.2188, 12 / 365
        c = black_scholes(S, K, r, sigma, T, "call")
        p = black_scholes(S, K, r, sigma, T, "put")
        rhs = S - K * np.exp(-r * T)
        self.assertAlmostEqual(c - p, rhs, places=2)

    def test_black_scholes_parity_with_dividends(self):
        import numpy as np
        S, K, r, q, sigma, T = 2264.0, 2200.0, 0.0692, 0.02, 0.2188, 45 / 365
        c = black_scholes(S, K, r, sigma, T, "call", q)
        p = black_scholes(S, K, r, sigma, T, "put", q)
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        self.assertAlmostEqual(c - p, rhs, places=2)

    def test_binomial_parity(self):
        S, K, r, sigma, T, n = 2264.0, 2264.0, 0.0692, 0.2188, 12 / 365, 50
        c = binomial_price(S, K, r, sigma, T, n, "call")
        p = binomial_price(S, K, r, sigma, T, n, "put")
        import numpy as np
        rhs = S - K * np.exp(-r * T)
        self.assertAlmostEqual(c - p, rhs, places=1)

    def test_binomial_parity_with_dividends(self):
        # Mirrors test_black_scholes_parity_with_dividends, but for the CRR
        # binomial tree -- previously untested even though every report run
        # prices the headline binomial results with q potentially nonzero.
        import numpy as np
        S, K, r, q, sigma, T, n = 2264.0, 2200.0, 0.0692, 0.02, 0.2188, 45 / 365, 200
        c = binomial_price(S, K, r, sigma, T, n, "call", q)
        p = binomial_price(S, K, r, sigma, T, n, "put", q)
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        self.assertAlmostEqual(c - p, rhs, places=1)


class TestImpliedVolatilitySolver(unittest.TestCase):
    def test_recovers_known_sigma_call(self):
        S, K, r, T, true_sigma = 2264.0, 2264.0, 0.0692, 12 / 365, 0.19
        price = black_scholes(S, K, r, true_sigma, T, "call")
        solved = implied_volatility(price, S, K, r, T, "call")
        self.assertAlmostEqual(solved, true_sigma, places=4)

    def test_recovers_known_sigma_put_otm(self):
        S, K, r, T, true_sigma = 2264.0, 2400.0, 0.0692, 30 / 365, 0.35
        price = black_scholes(S, K, r, true_sigma, T, "put")
        solved = implied_volatility(price, S, K, r, T, "put")
        self.assertAlmostEqual(solved, true_sigma, places=3)

    def test_deep_otm_low_vega_does_not_crash(self):
        # Deep OTM, short-dated -> near-zero vega, Newton is unstable here;
        # this is exactly the case the bisection fallback exists for.
        S, K, r, T = 2264.0, 3500.0, 0.0692, 3 / 365
        price = black_scholes(S, K, r, 0.9, T, "call")
        solved = implied_volatility(price, S, K, r, T, "call")
        self.assertFalse(solved != solved)  # not NaN

    def test_price_outside_no_arbitrage_bounds_returns_nan(self):
        S, K, r, T = 2264.0, 2264.0, 0.0692, 12 / 365
        solved = implied_volatility(market_price=-5.0, S=S, K=K, r=r, T=T)
        self.assertTrue(solved != solved)  # is NaN


class TestExpiryCalculation(unittest.TestCase):
    def test_last_tuesday_of_month(self):
        expiry = compute_monthly_expiry(date(2026, 5, 15), expiry_weekday=1)
        self.assertEqual(expiry, date(2026, 5, 26))
        self.assertEqual(expiry.weekday(), 1)

    def test_rolls_to_next_month_if_already_passed(self):
        expiry = compute_monthly_expiry(date(2026, 5, 27), expiry_weekday=1)
        self.assertEqual(expiry.month, 6)
        self.assertEqual(expiry.weekday(), 1)

    def test_resolve_expiry_rejects_past_date(self):
        with self.assertRaises(ValueError):
            resolve_expiry(run_date=date(2026, 6, 1), override="2026-05-28", expiry_weekday=1)

    def test_resolve_expiry_accepts_future_override(self):
        expiry = resolve_expiry(run_date=date(2026, 5, 15), override="2026-06-30", expiry_weekday=1)
        self.assertEqual(expiry, date(2026, 6, 30))


class TestAtmStrikeSelection(unittest.TestCase):
    """
    Regression coverage for the K=S bug: an earlier version of this project
    set the strike equal to spot before ATM selection ran, silently ignoring
    the option chain. select_atm_strike() must be able to return a strike
    that differs from spot whenever the numerically-nearest strike fails the
    liquidity filter -- that's the whole point of filtering. These tests
    build a synthetic chain where that's guaranteed to happen, so this would
    fail immediately if that bug (or an equivalent one) were reintroduced.
    """

    def _synthetic_chain(self):
        import pandas as pd
        # Spot is 2280. The numerically-closest strike (2280) is deliberately
        # thin (fails min_oi/min_volume) -- a stale/illiquid print, exactly
        # like the deep-ITM outliers seen in real NSE chain data. 2260 is
        # also thin, so 2300 is the unique, unambiguous nearest strike that
        # actually clears the liquidity bar.
        rows = [
            # strike, call_ltp, call_iv, call_oi, call_volume
            (2260, 40.0, 0.28, 5, 0),       # thin, fails filter
            (2280, 32.9, 0.29, 5, 0),       # nearest to spot, but illiquid
            (2300, 25.6, 0.27, 800, 450),   # liquid, one step further out
            (2320, 18.4, 0.26, 700, 400),
        ]
        return pd.DataFrame(rows, columns=["strike", "call_ltp", "call_iv", "call_oi", "call_volume"])

    def test_atm_strike_can_differ_from_spot_when_nearest_is_illiquid(self):
        chain = self._synthetic_chain()
        S = 2280.0
        row, liquidity_filtered = select_atm_strike(chain, S, have_liquidity_cols=True,
                                                       min_oi=100, min_volume=1)
        self.assertTrue(liquidity_filtered)
        self.assertNotEqual(float(row["strike"]), S)
        self.assertEqual(float(row["strike"]), 2300.0)

    def test_atm_strike_equals_nearest_when_that_strike_is_liquid(self):
        import pandas as pd
        chain = pd.DataFrame([
            (2260, 40.0, 0.28, 500, 300),
            (2280, 32.9, 0.29, 900, 500),  # nearest to spot AND liquid this time
            (2300, 25.6, 0.27, 800, 450),
        ], columns=["strike", "call_ltp", "call_iv", "call_oi", "call_volume"])
        S = 2280.0
        row, liquidity_filtered = select_atm_strike(chain, S, have_liquidity_cols=True,
                                                       min_oi=100, min_volume=1)
        self.assertTrue(liquidity_filtered)
        self.assertEqual(float(row["strike"]), 2280.0)

    def test_falls_back_to_nearest_iv_when_no_liquidity_columns(self):
        import pandas as pd
        chain = pd.DataFrame([
            (2260, 40.0, 0.28),
            (2280, 32.9, 0.29),
            (2300, 25.6, 0.27),
        ], columns=["strike", "call_ltp", "call_iv"])
        S = 2280.0
        row, liquidity_filtered = select_atm_strike(chain, S, have_liquidity_cols=False)
        self.assertFalse(liquidity_filtered)
        self.assertEqual(float(row["strike"]), 2280.0)


class TestPartialLiquidityColumns(unittest.TestCase):
    """
    Regression coverage for the "OR across four optional columns" bug: a
    chain CSV with call OI/Volume but no put OI/Volume (or vice versa) used
    to set a single blanket have_liquidity_cols=True, which made
    select_atm_strike() silently treat the missing put_volume/call_volume
    column as always-zero and fall back to the IV-only heuristic -- without
    the "no liquidity columns" warning ever firing, since that check only
    looked at the blanket flag. load_option_chain() now reports call/put
    liquidity availability separately.
    """

    def _write_partial_chain(self, tmp_path, include_call_liquidity, include_put_liquidity):
        import csv as csv_mod
        header = ["STRIKE", "LTP", "IV"]
        if include_call_liquidity:
            header += ["OI", "VOLUME"]
        header += ["LTP.1", "IV.1"]
        if include_put_liquidity:
            header += ["OI.1", "VOLUME.1"]

        rows = []
        for strike, c_ltp, c_iv, c_oi, c_vol, p_ltp, p_iv, p_oi, p_vol in [
            (2240, 45.0, 18.0, 800, 400, 22.0, 24.0, 700, 350),
            (2260, 32.9, 16.23, 1200, 600, 28.5, 20.5, 1100, 500),
            (2280, 22.0, 17.0, 900, 450, 36.0, 23.0, 850, 420),
        ]:
            row = [strike, c_ltp, c_iv]
            if include_call_liquidity:
                row += [c_oi, c_vol]
            row += [p_ltp, p_iv]
            if include_put_liquidity:
                row += [p_oi, p_vol]
            rows.append(row)

        path = tmp_path
        with open(path, "w", newline="") as f:
            f.write("skiprow\n")
            writer = csv_mod.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def test_call_only_liquidity_columns_detected_independently(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = self._write_partial_chain(os.path.join(d, "chain.csv"), True, False)
            chain, liquidity_info = load_option_chain(path)
            self.assertTrue(liquidity_info["call"])
            self.assertFalse(liquidity_info["put"])

    def test_put_only_liquidity_columns_detected_independently(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = self._write_partial_chain(os.path.join(d, "chain.csv"), False, True)
            chain, liquidity_info = load_option_chain(path)
            self.assertFalse(liquidity_info["call"])
            self.assertTrue(liquidity_info["put"])

    def test_select_atm_strike_uses_call_liquidity_flag_directly(self):
        # Previously this path silently defaulted missing call_volume to 0
        # and could return liquidity_filtered=True/False inconsistently
        # with the caller's blanket flag. Now the caller passes exactly
        # liquidity_info["call"], so behaviour is unambiguous.
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = self._write_partial_chain(os.path.join(d, "chain.csv"), True, False)
            chain, liquidity_info = load_option_chain(path)
            row, filtered = select_atm_strike(chain, S=2262.0, have_liquidity_cols=liquidity_info["call"],
                                                min_oi=100, min_volume=1)
            self.assertTrue(filtered)
            self.assertEqual(float(row["strike"]), 2260.0)

    def test_skew_stats_falls_back_only_on_missing_side(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = self._write_partial_chain(os.path.join(d, "chain.csv"), True, False)
            chain, liquidity_info = load_option_chain(path)
            stats = compute_skew_stats(chain, liquidity_info, atm_strike=2260.0)
            self.assertTrue(stats["call_filtered"])
            self.assertFalse(stats["put_filtered"])
            self.assertFalse(stats["filtered"])  # overall flag requires BOTH sides


class TestLoadOptionChainErrorHandling(unittest.TestCase):
    """
    Regression tests for a real failure: an option-chain CSV that doesn't
    have a 'STRIKE' column (wrong file, or NSE's export format/header-row
    count changed) used to fail deep inside pandas with a bare
    'KeyError: [\'strike\']' -- no indication of what was actually wrong or
    which columns were found instead.
    """

    def test_missing_strike_column_raises_clear_error(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.csv")
            with open(path, "w") as f:
                f.write("Title Row\nFOO,BAR,BAZ\n1,2,3\n")
            with self.assertRaises(ValueError) as ctx:
                load_option_chain(path)
            self.assertIn("STRIKE", str(ctx.exception))
            self.assertIn("FOO", str(ctx.exception))  # names the actual columns found

    def test_utf8_bom_does_not_break_strike_detection(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bom.csv")
            with open(path, "wb") as f:
                f.write(b"\xef\xbb\xbf")  # UTF-8 BOM
                f.write(b"Title Row\nSTRIKE,LTP,IV,LTP.1,IV.1\n2260,32.9,16.23,28.5,20.5\n")
            chain, liquidity_info = load_option_chain(path)
            self.assertEqual(list(chain["strike"]), [2260.0])


class TestSkewStatsExcludesImplausibleIV(unittest.TestCase):
    """
    Regression test for a real, confirmed distortion: a deep-ITM strike near
    expiry can have real OI/volume (clearing the liquidity filter easily)
    while still being ~all intrinsic value, so inverting Black-Scholes on
    its tiny residual extrinsic value produces a meaningless annualised IV.
    Confirmed live against a real chain: a liquid (OI=107, volume=38)
    strike far below spot solved to 353% call IV on a 1-day-to-expiry
    contract. compute_skew_stats() must exclude this via the moneyness
    bound / IV ceiling, not just the OI/volume liquidity check.
    """

    def test_deep_itm_liquid_but_implausible_iv_excluded_from_call_range(self):
        import tempfile, os, csv as csv_mod

        header = ["STRIKE", "LTP", "IV", "OI", "VOLUME", "LTP.1", "IV.1", "OI.1", "VOLUME.1"]
        rows = [
            # Deep ITM, liquid, but implausible solved IV -- exactly the
            # 1800-strike/2280-spot case found live. Must be excluded.
            [1800, 502.00, 353.32, 107, 38, 0.05, 5.0, 50, 10],
            # Genuinely near-the-money, liquid, plausible IV -- must survive.
            [2240, 45.0, 24.0, 800, 400, 22.0, 26.0, 700, 350],
            [2260, 32.9, 25.0, 1200, 600, 28.5, 27.0, 1100, 500],
            [2280, 22.0, 26.0, 900, 450, 36.0, 28.0, 850, 420],
        ]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "chain.csv")
            with open(path, "w", newline="") as f:
                f.write("skiprow\n")
                writer = csv_mod.writer(f)
                writer.writerow(header)
                writer.writerows(rows)

            chain, liquidity_info = load_option_chain(path)
            stats = compute_skew_stats(chain, liquidity_info, atm_strike=2280.0,
                                        min_oi=100, min_volume=1)
            self.assertIsNotNone(stats["call_iv_max"])
            self.assertLess(stats["call_iv_max"], 0.50,
                             "the 353% deep-ITM print leaked into the reported call IV range")


class TestUniquePath(unittest.TestCase):
    def test_returns_same_path_when_free(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "report.docx")
            self.assertEqual(unique_path(path), path)

    def test_returns_incremented_path_when_taken(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "convergence_plot.png")
            open(path, "w").close()
            result = unique_path(path)
            self.assertEqual(result, os.path.join(d, "convergence_plot_1.png"))

    def test_increments_past_multiple_collisions(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "volatility_smile.png")
            open(path, "w").close()
            open(os.path.join(d, "volatility_smile_1.png"), "w").close()
            result = unique_path(path)
            self.assertEqual(result, os.path.join(d, "volatility_smile_2.png"))


class TestGreeksSanity(unittest.TestCase):
    def test_call_delta_between_0_and_1(self):
        g = bs_greeks(2264.0, 2264.0, 0.0692, 0.2188, 12 / 365)
        self.assertTrue(0 < g["delta_call"] < 1)

    def test_put_delta_between_minus1_and_0(self):
        g = bs_greeks(2264.0, 2264.0, 0.0692, 0.2188, 12 / 365)
        self.assertTrue(-1 < g["delta_put"] < 0)

    def test_gamma_positive(self):
        g = bs_greeks(2264.0, 2264.0, 0.0692, 0.2188, 12 / 365)
        self.assertGreater(g["gamma"], 0)

    def test_vega_positive(self):
        g = bs_greeks(2264.0, 2264.0, 0.0692, 0.2188, 12 / 365)
        self.assertGreater(g["vega"], 0)


class TestInvalidOptionType(unittest.TestCase):
    """
    Both pricing functions guard option_type with an explicit ValueError --
    this just confirms that guard actually fires (and stays firing if either
    function is refactored later) instead of silently falling through to a
    call price, a put price, or a confusing exception from deeper in the
    formula.
    """

    def test_black_scholes_rejects_invalid_option_type(self):
        with self.assertRaises(ValueError):
            black_scholes(2264.0, 2264.0, 0.0692, 0.2188, 12 / 365, "straddle")

    def test_binomial_rejects_invalid_option_type(self):
        with self.assertRaises(ValueError):
            binomial_price(2264.0, 2264.0, 0.0692, 0.2188, 12 / 365, 50, "straddle")


class TestHolidayAdjustedExpiry(unittest.TestCase):
    """
    Regression coverage for the NSE-holiday-adjustment fix. Cases are
    checked against the actual NSE circular (NSE/CMTR/71775, 12-Dec-2025)
    rather than invented dates, so these fail immediately if the holiday
    list or the rollback logic drifts from what NSE actually published.
    """

    def test_holiday_tuesday_rolls_back_to_previous_trading_day(self):
        # March 31, 2026 is a Tuesday AND Shri Mahavir Jayanti (an NSE
        # holiday) -- the last Tuesday of March, so this is exactly the
        # case the rollback exists for. March 30 (Monday) is not a holiday.
        expiry = compute_monthly_expiry(date(2026, 3, 15), expiry_weekday=1)
        self.assertEqual(expiry, date(2026, 3, 30))
        self.assertFalse(is_nse_trading_holiday(expiry))

    def test_non_holiday_expiry_unaffected(self):
        # May 26, 2026 is a Tuesday and NOT an NSE holiday (Bakri Id that
        # month falls on Thursday May 28, a different weekday) -- rollback
        # must not fire here.
        expiry = compute_monthly_expiry(date(2026, 5, 15), expiry_weekday=1)
        self.assertEqual(expiry, date(2026, 5, 26))

    def test_unlisted_year_does_not_crash_and_returns_unadjusted(self):
        # No NSE_HOLIDAYS_2027 calendar exists yet. This must degrade to
        # the plain calendar calculation (with a warning) rather than
        # raising -- silently wrong-by-a-holiday is recoverable by passing
        # --expiry explicitly; a crash is not.
        expiry = compute_monthly_expiry(date(2027, 5, 15), expiry_weekday=1)
        self.assertEqual(expiry.weekday(), 1)

    def test_is_nse_trading_holiday_matches_circular(self):
        self.assertTrue(is_nse_trading_holiday(date(2026, 1, 26)))   # Republic Day
        self.assertTrue(is_nse_trading_holiday(date(2026, 10, 2)))   # Gandhi Jayanti
        self.assertFalse(is_nse_trading_holiday(date(2026, 1, 27)))  # day after, not a holiday

    def test_roll_back_skips_consecutive_holidays_and_weekends(self):
        # Synthetic worst case: a run of holiday/weekend days in a row.
        # Not an NSE scenario specifically, just confirms the while-loop
        # in roll_back_to_trading_day terminates correctly and lands on
        # an actual trading day rather than looping or off-by-one.
        result = roll_back_to_trading_day(date(2026, 1, 26))  # Republic Day, Monday
        self.assertFalse(is_nse_trading_holiday(result))
        self.assertLess(result.weekday(), 5)
        self.assertLessEqual(result, date(2026, 1, 26))


class TestLogRunSchemaMigration(unittest.TestCase):
    """
    Regression coverage for the run_history.csv corruption bug: appending a
    record with a different key set than the file's existing header used
    to write rows positionally with no validation, producing a file
    pandas.read_csv cannot parse (confirmed reproducible: "Error
    tokenizing data. C error: Expected N fields..." on a real file from
    this project). log_run() must detect the mismatch and migrate instead.
    """

    def test_creates_new_file_with_header(self):
        import tempfile, os, csv as csv_mod
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "run_history.csv")
            log_run(path, {"a": 1, "b": 2})
            with open(path, newline="") as f:
                rows = list(csv_mod.reader(f))
            self.assertEqual(rows[0], ["a", "b"])
            self.assertEqual(rows[1], ["1", "2"])

    def test_appends_when_schema_matches(self):
        import tempfile, os, csv as csv_mod
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "run_history.csv")
            log_run(path, {"a": 1, "b": 2})
            log_run(path, {"a": 3, "b": 4})
            with open(path, newline="") as f:
                rows = list(csv_mod.reader(f))
            self.assertEqual(len(rows), 3)  # header + 2 rows
            self.assertEqual(rows[0], ["a", "b"])

    def test_migrates_when_new_column_added(self):
        import tempfile, os, csv as csv_mod
        import pandas as pd
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "run_history.csv")
            log_run(path, {"a": 1, "b": 2})               # old schema
            log_run(path, {"a": 3, "version": "1.0", "b": 4})  # new column inserted

            # Must be parseable by pandas -- this is the actual bug symptom
            # that was reproduced against the real corrupted file.
            df = pd.read_csv(path)
            self.assertEqual(len(df), 2)
            self.assertIn("version", df.columns)
            self.assertTrue(pd.isna(df.loc[0, "version"]) or df.loc[0, "version"] == "")
            self.assertEqual(float(df.loc[1, "version"]), 1.0)
            self.assertEqual(df.loc[1, "a"], 3)
            self.assertEqual(df.loc[1, "b"], 4)

    def test_no_data_loss_for_old_only_columns(self):
        # If an old file has a column the new record doesn't have, that
        # column must be preserved (appended at the end), not silently
        # dropped during migration.
        import tempfile, os
        import pandas as pd
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "run_history.csv")
            log_run(path, {"a": 1, "old_only_col": "keep_me"})
            log_run(path, {"a": 2, "new_col": "x"})
            df = pd.read_csv(path)
            self.assertIn("old_only_col", df.columns)
            self.assertEqual(df.loc[0, "old_only_col"], "keep_me")




if __name__ == "__main__":
    unittest.main()
