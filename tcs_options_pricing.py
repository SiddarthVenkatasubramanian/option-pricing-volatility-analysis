"""
TCS Options Pricing & Volatility Analysis
==========================================
CRR Binomial and Black-Scholes pricing using TCS market data, with:
- historical and NSE-published implied volatility
- IV solved from the observed market call premium
- liquidity-aware ATM strike selection
- dividend-yield-aware pricing
- Greeks and sensitivity analysis
- put-call parity and binomial convergence checks
- run-history CSV logging
- automatic Word-report generation from a reusable template

Example:
    python TCS_Options_Pricing_and_Volatility_Analysis_updated.py --help
"""

import argparse
import csv
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

try:
    import yfinance as yf
    HAVE_YFINANCE = True
except ImportError:
    HAVE_YFINANCE = False

# Bump this whenever build_report_values(), the pricing functions, or the
# report template's placeholder contract change. Recorded in run_history.csv
# so past rows can be told apart from ones produced by a different (possibly
# buggy) version of this script.
SCRIPT_VERSION = "1.4.0"

try:
    from docx import Document
    HAVE_DOCX = True
except ImportError:
    HAVE_DOCX = False


# =========================================================
# ASSUMPTIONS
# =========================================================

EXERCISE_STYLE = "European"


# =========================================================
# 1. EXPIRY CALCULATION
# =========================================================

# NSE weekday trading holidays for 2026, per official circular
# NSE/CMTR/71775 (Download Ref No., dated 12-Dec-2025):
# https://nsearchives.nseindia.com/content/circulars/CMTR71775.pdf
# Deliberately excludes the four 2026 holidays that fall on a Sat/Sun (Feb
# 15, Mar 21, Aug 15, Nov 8) -- those don't affect any weekday expiry
# calculation and including them would be inert but confusing.
#
# IMPORTANT: this list is specific to calendar year 2026 and does not
# extend automatically. NSE publishes a new circular every December for
# the following year -- this constant needs a manual update (with the new
# circular's Download Ref No.) at the start of each year, or expiries in
# an unlisted year will silently get zero holiday adjustment rather than
# an error. A live/external holiday-calendar source would avoid that, at
# the cost of another network dependency this project has otherwise
# avoided (see the option-chain-fetch discussion elsewhere in this
# project's history).
NSE_HOLIDAYS_2026 = {
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 3),    # Holi
    date(2026, 3, 26),   # Shri Ram Navami
    date(2026, 3, 31),   # Shri Mahavir Jayanti
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 28),   # Bakri Id
    date(2026, 6, 26),   # Muharram
    date(2026, 9, 14),   # Ganesh Chaturthi
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 11, 10),  # Diwali-Balipratipada
    date(2026, 11, 24),  # Prakash Gurpurb Sri Guru Nanak Dev
    date(2026, 12, 25),  # Christmas
}

NSE_HOLIDAY_CALENDARS = {
    2026: NSE_HOLIDAYS_2026,
}


def is_nse_trading_holiday(d):
    """True if d is a known NSE weekday trading holiday. Returns False (not
    an error) for years outside NSE_HOLIDAY_CALENDARS, with a warning
    printed by the caller -- see roll_back_to_trading_day()."""
    return d in NSE_HOLIDAY_CALENDARS.get(d.year, frozenset())


def roll_back_to_trading_day(d):
    """NSE convention: if a scheduled expiry falls on a holiday (or a
    weekend, as a defensive fallback), it moves to the previous trading
    day -- not the next one. Warns if d's year has no holiday calendar
    loaded, since in that case this function cannot actually detect a
    holiday and will silently return d unchanged."""
    if d.year not in NSE_HOLIDAY_CALENDARS:
        print(f"[WARN] No NSE holiday calendar loaded for {d.year} -- "
              f"holiday adjustment skipped for {d}. Add a "
              f"NSE_HOLIDAYS_{d.year} entry (from NSE's December circular "
              "for that year) or pass --expiry explicitly.")
        return d
    while d.weekday() >= 5 or is_nse_trading_holiday(d):
        d -= timedelta(days=1)
    return d


def compute_monthly_expiry(run_date, expiry_weekday=1):
    """Return the last occurrence of expiry_weekday in the current/following
    month, rolled back to the previous NSE trading day if that date is a
    holiday. Monday=0 ... Sunday=6. Default=Tuesday."""

    def last_weekday_of_month(year, month, weekday):
        if month == 12:
            next_month_first = date(year + 1, 1, 1)
        else:
            next_month_first = date(year, month + 1, 1)
        last_day = next_month_first - timedelta(days=1)
        offset = (last_day.weekday() - weekday) % 7
        return last_day - timedelta(days=offset)

    candidate = last_weekday_of_month(run_date.year, run_date.month, expiry_weekday)
    candidate = roll_back_to_trading_day(candidate)
    if candidate <= run_date:
        year, month = run_date.year, run_date.month + 1
        if month == 13:
            year, month = year + 1, 1
        candidate = last_weekday_of_month(year, month, expiry_weekday)
        candidate = roll_back_to_trading_day(candidate)
    return candidate


def resolve_expiry(run_date, override, expiry_weekday):
    if override is not None:
        expiry = date.fromisoformat(override)
    else:
        expiry = compute_monthly_expiry(run_date, expiry_weekday)
        calendar_status = (
            "NSE-holiday-adjusted" if expiry.year in NSE_HOLIDAY_CALENDARS
            else "NOT holiday-adjusted -- no calendar loaded for this year"
        )
        print(
            f"[INFO] --expiry not supplied; auto-computed {expiry} ({calendar_status}). "
            "Pass --expiry explicitly if this doesn't match your downloaded chain."
        )

    if expiry <= run_date:
        raise ValueError(
            f"Expiry {expiry} is not in the future relative to run date {run_date}."
        )
    return expiry


# =========================================================
# 2. HISTORICAL RETURNS + VOLATILITY
# =========================================================

def get_historical_volatility(symbol, end_date=None, lookback_days=365):
    """Download historical prices ending at the analysis date."""
    if not HAVE_YFINANCE:
        raise RuntimeError("yfinance is not installed.")

    end_date = end_date or date.today()
    start_date = end_date - timedelta(days=lookback_days + 10)

    data = yf.download(
        symbol,
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
    )["Close"].squeeze()

    data = data.dropna()
    if data.empty:
        raise ValueError(f"No valid price data downloaded for {symbol}")

    returns = np.log(data / data.shift(1)).dropna()
    mean_return = returns.mean() * 252       # descriptive only
    sigma = returns.std() * np.sqrt(252)
    return returns, mean_return, sigma


# =========================================================
# 3. CRR BINOMIAL MODEL
# =========================================================

def binomial_price(S, K, r, sigma, T, n, option_type="call", q=0.0):
    """European CRR binomial price with continuous dividend yield q."""
    if EXERCISE_STYLE != "European":
        raise ValueError("This implementation is for European options only.")
    if T <= 0 or sigma <= 0 or n < 1:
        raise ValueError("Require T>0, sigma>0 and n>=1.")

    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)

    if not (0 < p < 1):
        raise ValueError("Risk-neutral probability p is outside (0,1).")

    j = np.arange(n + 1)
    ST = S * (u ** j) * (d ** (n - j))

    if option_type == "call":
        values = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        values = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    disc = np.exp(-r * dt)
    for i in range(n - 1, -1, -1):
        values = disc * (p * values[1:i + 2] + (1 - p) * values[0:i + 1])

    return float(values[0])


# =========================================================
# 4. BLACK-SCHOLES + GREEKS
# =========================================================

def black_scholes(S, K, r, sigma, T, option_type="call", q=0.0):
    if T <= 0 or sigma <= 0:
        raise ValueError("Require T>0 and sigma>0.")

    d1 = (
        np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T
    ) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call = (
        S * np.exp(-q * T) * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )
    put = (
        K * np.exp(-r * T) * norm.cdf(-d2)
        - S * np.exp(-q * T) * norm.cdf(-d1)
    )

    if option_type == "call":
        return float(call)
    if option_type == "put":
        return float(put)
    raise ValueError("option_type must be 'call' or 'put'.")


def bs_greeks(S, K, r, sigma, T, q=0.0):
    """Vega is per 1% volatility move; theta is per day; rho per 1% rate."""
    d1 = (
        np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T
    ) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta_call = np.exp(-q * T) * norm.cdf(d1)
    delta_put = np.exp(-q * T) * (norm.cdf(d1) - 1)
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100

    theta_call = (
        -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2)
        + q * S * np.exp(-q * T) * norm.cdf(d1)
    ) / 365

    theta_put = (
        -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        + r * K * np.exp(-r * T) * norm.cdf(-d2)
        - q * S * np.exp(-q * T) * norm.cdf(-d1)
    ) / 365

    rho_call = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        "delta_call": float(delta_call),
        "delta_put": float(delta_put),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta_call": float(theta_call),
        "theta_put": float(theta_put),
        "rho_call": float(rho_call),
        "rho_put": float(rho_put),
    }


# =========================================================
# 5. IMPLIED VOLATILITY
# =========================================================

def implied_volatility(
    market_price, S, K, r, T, option_type="call", q=0.0,
    tol=1e-6, max_iter=100, sigma_bounds=(1e-4, 5.0)
):
    """Solve IV from market price using Newton-Raphson with bisection fallback."""
    lo, hi = sigma_bounds
    price_lo = black_scholes(S, K, r, lo, T, option_type, q)
    price_hi = black_scholes(S, K, r, hi, T, option_type, q)

    if not (price_lo - tol <= market_price <= price_hi + tol):
        return np.nan

    sigma = 0.30
    for _ in range(max_iter):
        price = black_scholes(S, K, r, sigma, T, option_type, q)
        vega = bs_greeks(S, K, r, sigma, T, q)["vega"] * 100
        diff = price - market_price

        if abs(diff) < tol:
            return float(sigma)
        if vega < 1e-8:
            break

        sigma -= diff / vega
        if not (lo < sigma < hi):
            break
    else:
        return float(sigma)

    # Bisection fallback
    a, b = lo, hi
    for _ in range(200):
        mid = (a + b) / 2
        price_mid = black_scholes(S, K, r, mid, T, option_type, q)
        if abs(price_mid - market_price) < tol:
            return float(mid)
        if price_mid < market_price:
            a = mid
        else:
            b = mid
    return float((a + b) / 2)


# =========================================================
# 6. OPTION CHAIN + ATM SELECTION
# =========================================================

def load_option_chain(path):
    """
    Returns (chain, liquidity_info) where liquidity_info = {"call": bool, "put": bool}.

    liquidity_info["call"] is True only when BOTH call_oi and call_volume are
    present (and likewise for "put"). A single blanket "have_liquidity_cols"
    flag used to be set True if ANY one of the four optional columns showed
    up -- so a chain CSV with call OI but no call volume (or vice versa)
    looked "liquidity-aware" globally, while select_atm_strike() silently
    treated the missing column as all-zero volume, always failed the
    min_volume check, and fell back to the IV-only heuristic without ever
    printing the "no liquidity columns" warning (that warning was gated on
    the same blanket flag, which was True). Tracking call/put availability
    separately means the fallback path and its warning now agree with what
    actually happened.
    """
    # encoding='utf-8-sig' strips a leading UTF-8 BOM if present (Excel and
    # some browsers add one on save/export) -- without this, a BOM makes the
    # first column read as '\ufeffSTRIKE' instead of 'STRIKE', which fails
    # the exact-match lookup below silently and produces a confusing
    # downstream KeyError with no indication of what actually went wrong.
    df = pd.read_csv(path, skiprows=1, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    col_map = {
        "STRIKE": "strike", "LTP": "call_ltp", "IV": "call_iv",
        "LTP.1": "put_ltp", "IV.1": "put_iv",
    }
    optional_cols = {
        "OI": "call_oi", "VOLUME": "call_volume",
        "OI.1": "put_oi", "VOLUME.1": "put_volume",
    }

    keep = {k: v for k, v in col_map.items() if k in df.columns}
    for k, v in optional_cols.items():
        if k in df.columns:
            keep[k] = v

    if "STRIKE" not in keep:
        raise ValueError(
            f"Could not find a 'STRIKE' column in {path}. This usually means "
            f"either the file isn't the standard NSE option-chain export (a "
            f"title row, then a header row containing STRIKE/LTP/IV/...), or "
            f"the number of header rows to skip has changed on NSE's site "
            f"(currently hardcoded as skiprows=1). Columns actually found "
            f"after skipping 1 row: {list(df.columns)}"
        )

    chain = df[list(keep.keys())].copy()
    chain.columns = list(keep.values())

    for col in chain.columns:
        chain[col] = pd.to_numeric(
            chain[col].astype(str)
            .str.replace(",", "")
            .str.replace("%", "")
            .str.strip()
            .str.replace(r"^-+$", "", regex=True),
            errors="coerce",
        )

    chain = chain.dropna(subset=["strike"]).sort_values("strike")
    if "call_iv" in chain:
        chain["call_iv"] = chain["call_iv"] / 100
    if "put_iv" in chain:
        chain["put_iv"] = chain["put_iv"] / 100

    liquidity_info = {
        "call": "call_oi" in chain.columns and "call_volume" in chain.columns,
        "put": "put_oi" in chain.columns and "put_volume" in chain.columns,
    }

    return chain, liquidity_info


def _filter_trustworthy_strikes(df, iv_col, oi_col, vol_col, has_liquidity_cols,
                                 atm_strike, min_oi=100, min_volume=1,
                                 moneyness_bound=0.15, iv_ceiling=1.50):
    """
    Shared "is this strike's IV trustworthy enough to summarise or plot"
    filter, used by both compute_skew_stats() (the report's Section 9 text)
    and the volatility-smile chart. These used to be two independent
    filters -- a liquidity-only one here, a crude flat IV%-bound one in the
    chart code -- that could silently disagree about which strikes to trust,
    and neither caught the actual failure mode below.

    Three checks, since each catches a different problem:
      - liquidity (OI/Volume, when the CSV has those columns): excludes
        strikes nobody is actually trading.
      - moneyness bound: excludes strikes far enough from spot that the
        option is almost pure intrinsic value. This is the one that matters
        most near expiry -- a deep-ITM strike can have real OI and volume
        (clearing the liquidity check easily) and still produce a
        meaningless IV, because inverting Black-Scholes on a tiny residual
        extrinsic value blows up. Confirmed live: a 1-day-to-expiry chain
        with spot ~2280 had a liquid (OI=107, volume=38) strike at 1800 --
        ~96% intrinsic value -- solving to 353% annualised IV.
      - IV ceiling: a backstop for whatever the first two miss.
    """
    side = df.dropna(subset=[iv_col])
    if has_liquidity_cols:
        side = side[
            (side[oi_col].fillna(0) >= min_oi)
            & (side[vol_col].fillna(0) >= min_volume)
        ]
    side = side[
        (side[iv_col] > 0.01)
        & (side[iv_col] <= iv_ceiling)
        & ((side["strike"] - atm_strike).abs() <= moneyness_bound * atm_strike)
    ]
    return side


def compute_skew_stats(chain, liquidity_info, atm_strike, min_oi=100, min_volume=1,
                        moneyness_bound=0.15, iv_ceiling=1.50):
    """
    Summarise the volatility skew from the option chain for use in the report
    narrative, instead of hand-typed percentages that only describe one past
    run. Returns None entries (formatted as "N/A" downstream) where there
    isn't enough trustworthy data to say something meaningful.
    """
    def liquid_side(df, iv_col, oi_col, vol_col, has_liquidity_cols):
        return _filter_trustworthy_strikes(
            df, iv_col, oi_col, vol_col, has_liquidity_cols, atm_strike,
            min_oi, min_volume, moneyness_bound, iv_ceiling,
        )

    calls = liquid_side(chain, "call_iv", "call_oi", "call_volume", liquidity_info["call"])
    puts = liquid_side(chain, "put_iv", "put_oi", "put_volume", liquidity_info["put"])

    stats = {
        "call_iv_min": calls["call_iv"].min() if not calls.empty else None,
        "call_iv_max": calls["call_iv"].max() if not calls.empty else None,
        "put_iv_min": puts["put_iv"].min() if not puts.empty else None,
        "put_iv_max": puts["put_iv"].max() if not puts.empty else None,
        "filtered": liquidity_info["call"] and liquidity_info["put"],
        "call_filtered": liquidity_info["call"],
        "put_filtered": liquidity_info["put"],
    }

    # ATM-adjacent put/call IV gap: use the strike closest to the actual ATM
    # strike among rows where BOTH sides clear the liquidity bar, so the
    # comparison is apples-to-apples rather than pairing a liquid call with
    # a stale put print (or vice versa). Requires both sides' columns to
    # actually be present -- partial columns fall back to the IV-only filter.
    both = chain.dropna(subset=["call_iv", "put_iv"])
    if liquidity_info["call"] and liquidity_info["put"]:
        both = both[
            (both["call_oi"].fillna(0) >= min_oi) & (both["call_volume"].fillna(0) >= min_volume)
            & (both["put_oi"].fillna(0) >= min_oi) & (both["put_volume"].fillna(0) >= min_volume)
        ]
    both = both[
        (both["call_iv"] > 0.01) & (both["call_iv"] <= iv_ceiling)
        & (both["put_iv"] > 0.01) & (both["put_iv"] <= iv_ceiling)
        & ((both["strike"] - atm_strike).abs() <= moneyness_bound * atm_strike)
    ]
    if not both.empty:
        row = both.loc[(both["strike"] - atm_strike).abs().idxmin()]
        stats["atm_gap"] = float(row["put_iv"] - row["call_iv"])
        stats["atm_gap_strike"] = float(row["strike"])
    else:
        stats["atm_gap"] = None
        stats["atm_gap_strike"] = None

    return stats


def select_atm_strike(chain, S, have_liquidity_cols, min_oi=100, min_volume=1):
    """
    have_liquidity_cols here refers specifically to the CALL side (both
    call_oi and call_volume present) -- that's the only side this function
    selects on. Callers passing the old combined flag from before this fix
    would have silently gotten a False positive when only the put columns
    were present; pass liquidity_info["call"] from load_option_chain().
    """
    if have_liquidity_cols:
        liquid = chain.dropna(subset=["call_iv"]).copy()
        liquid = liquid[
            (liquid["call_oi"].fillna(0) >= min_oi)
            & (liquid["call_volume"].fillna(0) >= min_volume)
            & (liquid["call_iv"] > 0.01)
        ]
        if not liquid.empty:
            row = liquid.loc[(liquid["strike"] - S).abs().idxmin()]
            return row, True

    valid_iv = chain.dropna(subset=["call_iv"])
    valid_iv = valid_iv[valid_iv["call_iv"] > 0.10]
    if valid_iv.empty:
        raise ValueError("No valid call-IV quote available for ATM selection.")
    row = valid_iv.loc[(valid_iv["strike"] - S).abs().idxmin()]
    return row, False


# =========================================================
# 7. RUN HISTORY
# =========================================================

def log_run(log_path, record):
    """
    Append a run to the CSV log, migrating the header if the record's
    schema doesn't match what's already on disk (e.g. script_version was
    added after the file already existed). Without this, csv.DictWriter in
    append mode writes new rows positionally with no validation against the
    existing header -- silently producing a file where some rows have more
    fields than the header names, which pandas.read_csv cannot parse at
    all (confirmed: raises "Error tokenizing data. C error: Expected N
    fields...").

    Reads the existing file with a plain csv.reader (not DictReader) and
    reconciles rows by raw field count, rather than assuming the file is
    already well-formed under its own header. This matters because a file
    can already be in the broken state this function is meant to prevent
    (some rows written under an old, shorter header; others already
    written with extra fields by a version of the script that added a
    column, all sitting under a stale header line) -- DictReader chokes on
    exactly that shape: rows longer than the header get their overflow
    dumped into a `None` key, which then crashes DictWriter with "dict
    contains fields not in fieldnames: None". This function instead
    disambiguates each row by comparing its raw length against both the
    file's stale header and the *current* record's field count, and heals
    the file in one pass -- it does not require the file to already be
    correct, only that this function has been called with it at least
    once.
    """
    new_fields = list(record.keys())

    if not os.path.isfile(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=new_fields)
            writer.writeheader()
            writer.writerow(record)
        return

    with open(log_path, "r", newline="") as f:
        raw = list(csv.reader(f))

    if not raw:
        existing_header, raw_rows = [], []
    else:
        existing_header, raw_rows = raw[0], raw[1:]

    if existing_header == new_fields and all(len(r) == len(new_fields) for r in raw_rows):
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=new_fields)
            writer.writerow(record)
        return

    old_only = [c for c in existing_header if c not in new_fields]
    unified_fields = new_fields + old_only
    n_new, n_old = len(new_fields), len(existing_header)

    migrated_rows = []
    unrecognised = 0
    for raw_row in raw_rows:
        if len(raw_row) == n_old:
            # Well-formed under the file's own (possibly stale) header.
            row_dict = dict(zip(existing_header, raw_row))
        elif len(raw_row) == n_new:
            # Already written in the *current* record's field order/count
            # (e.g. by a run that happened after a column was added, before
            # the header was ever migrated) -- map against new_fields, not
            # the stale header.
            row_dict = dict(zip(new_fields, raw_row))
        else:
            # Genuinely ambiguous shape -- don't crash the run and don't
            # guess; keep the raw values under generic column names so
            # nothing is silently dropped, and flag it for manual review.
            row_dict = {f"_unrecognised_col_{i}": v for i, v in enumerate(raw_row)}
            for extra_col in row_dict:
                if extra_col not in unified_fields:
                    unified_fields.append(extra_col)
            unrecognised += 1
        migrated_rows.append(row_dict)

    print(f"[WARN] {log_path} schema changed or was already inconsistent -- "
          f"migrating {len(migrated_rows)} existing row(s) to a unified header. "
          f"Missing values are left blank."
          + (f" {unrecognised} row(s) had an unrecognised shape and were "
             "preserved under generic column names -- check these manually."
             if unrecognised else ""))

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=unified_fields, restval="")
        writer.writeheader()
        for row in migrated_rows:
            writer.writerow(row)
        writer.writerow(record)


# =========================================================
# 8. SENSITIVITY
# =========================================================

def sensitivity_table(
    S, K, r, sigma, T, q=0.0,
    r_shifts=(-0.01, -0.005, 0, 0.005, 0.01),
    sigma_shifts=(-0.02, -0.01, 0, 0.01, 0.02),
):
    rows = []
    base = black_scholes(S, K, r, sigma, T, "call", q)

    for dr in r_shifts:
        price = black_scholes(S, K, r + dr, sigma, T, "call", q)
        rows.append({
            "variable": "r",
            "shifted_value": r + dr,
            "call_price": price,
            "delta_vs_base": price - base,
        })

    for ds in sigma_shifts:
        price = black_scholes(S, K, r, sigma + ds, T, "call", q)
        rows.append({
            "variable": "sigma",
            "shifted_value": sigma + ds,
            "call_price": price,
            "delta_vs_base": price - base,
        })

    return pd.DataFrame(rows)


# =========================================================
# 9. WORD REPORT AUTOMATION
# =========================================================

def _replace_placeholder_in_paragraph(paragraph, values):
    """Replace placeholders even if Word split them across multiple runs."""
    for key, value in values.items():
        while key in paragraph.text:
            texts = [run.text for run in paragraph.runs]
            full = "".join(texts)
            start = full.find(key)
            if start < 0:
                break
            end = start + len(key)

            pos = 0
            first = last = None
            first_local = last_local = None
            for i, txt in enumerate(texts):
                r0, r1 = pos, pos + len(txt)
                if first is None and r0 <= start < r1:
                    first = i
                    first_local = start - r0
                if r0 < end <= r1:
                    last = i
                    last_local = end - r0
                    break
                pos = r1

            if first is None:
                break
            if last is None:
                last = first
                last_local = first_local + len(key)

            if first == last:
                paragraph.runs[first].text = (
                    texts[first][:first_local] + str(value) + texts[first][last_local:]
                )
            else:
                paragraph.runs[first].text = texts[first][:first_local] + str(value)
                for j in range(first + 1, last):
                    paragraph.runs[j].text = ""
                paragraph.runs[last].text = texts[last][last_local:]


def _iter_report_paragraphs(doc):
    for p in doc.paragraphs:
        yield p
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    yield p
    for section in doc.sections:
        for p in section.header.paragraphs:
            yield p
        for p in section.footer.paragraphs:
            yield p


def unique_path(path):
    """
    If path doesn't exist, return it unchanged. Otherwise append _1, _2, ...
    before the extension until a free name is found, and print a warning so
    it's clear a collision was avoided rather than a previous run's output
    being silently overwritten.
    """
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while True:
        candidate = f"{base}_{i}{ext}"
        if not os.path.exists(candidate):
            print(f"[WARN] {path} already exists; writing to {candidate} instead "
                  "to avoid overwriting a previous run's output.")
            return candidate
        i += 1


def generate_word_report(template_path, output_path, values):
    if not HAVE_DOCX:
        raise RuntimeError("python-docx is not installed.")
    if not os.path.isfile(template_path):
        raise FileNotFoundError(f"Report template not found: {template_path}")

    doc = Document(template_path)
    for paragraph in _iter_report_paragraphs(doc):
        _replace_placeholder_in_paragraph(paragraph, values)
    doc.save(output_path)
    return output_path


# =========================================================
# 10. REPORT VALUE BUILDING
# =========================================================

def _fmt_date(d):
    """Format a date as '15 May 2026' without relying on the '%-d' strftime
    flag, which is a glibc/Linux extension and raises ValueError on Windows."""
    return f"{d.day} {d.strftime('%b %Y')}"


def build_report_values(
    run_date, expiry, stock_df, returns, mean_return, sigma,
    S, K, r, q, n, days_to_expiry, published_iv, solved_iv,
    market_ltp, results, skew_stats=None,
):
    values = {
        "{{DATA_DATE}}": _fmt_date(run_date),
        "{{EXPIRY_DATE}}": _fmt_date(expiry),
        "{{DATA_START}}": _fmt_date(stock_df["DATE"].iloc[-1]),
        # Bare day-of-month for the run date, for use in "15-28 May" style
        # inline date ranges that describe the option's life (run_date -> expiry),
        # not the one-year historical lookback window.
        "{{DATA_DATE_DAY}}": str(run_date.day),
        "{{TRADING_DAYS}}": str(len(returns)) if returns is not None else "N/A",
        "{{S}}": f"₹{S:,.2f}",
        "{{ATM_STRIKE}}": f"{K:.0f}",
        "{{R_PCT}}": f"{r * 100:.2f}%",
        "{{Q_PCT}}": f"{q * 100:.2f}%",
        "{{SIGMA}}": f"{sigma * 100:.2f}%" if sigma is not None else "N/A",
        "{{PUBLISHED_IV}}": f"{published_iv * 100:.2f}%",
        "{{SOLVED_IV}}": f"{solved_iv * 100:.2f}%" if not np.isnan(solved_iv) else "N/A",
        "{{T_DAYS}}": f"{days_to_expiry} days",
        "{{T_DAYS_NUM}}": str(days_to_expiry),
        "{{N_STEPS}}": str(n),
        "{{MEAN_RETURN}}": f"{mean_return * 100:.2f}%" if mean_return is not None else "N/A",
        "{{MARKET_LTP}}": f"₹{market_ltp:.2f}",
    }

    if returns is not None:
        values["{{ROLLING_30}}"] = f"{returns.rolling(30).std().iloc[-1] * np.sqrt(252) * 100:.2f}%"
        values["{{ROLLING_90}}"] = f"{returns.rolling(90).std().iloc[-1] * np.sqrt(252) * 100:.2f}%"
    else:
        values["{{ROLLING_30}}"] = values["{{ROLLING_90}}"] = "N/A"

    mapping = {
        "binomial_call_hist_vol": "{{CALL_HIST}}",
        "binomial_put_hist_vol": "{{PUT_HIST}}",
        "binomial_call_published_iv": "{{CALL_PUBLISHED}}",
        "binomial_put_published_iv": "{{PUT_PUBLISHED}}",
        "binomial_call_solved_iv": "{{CALL_SOLVED}}",
        "binomial_put_solved_iv": "{{PUT_SOLVED}}",
        "bs_call_hist_vol": "{{BS_CALL_HIST}}",
        "bs_put_hist_vol": "{{BS_PUT_HIST}}",
        "bs_call_published_iv": "{{BS_CALL_PUBLISHED}}",
        "bs_put_published_iv": "{{BS_PUT_PUBLISHED}}",
        "bs_call_solved_iv": "{{BS_CALL_SOLVED}}",
        "bs_put_solved_iv": "{{BS_PUT_SOLVED}}",
    }
    for key, placeholder in mapping.items():
        values[placeholder] = f"₹{results[key]:.2f}" if key in results else "N/A"

    # Convergence
    if sigma is not None:
        conv_steps = [3, 5, 10, 25, 50, 100, 200, 500]
        for step in conv_steps:
            p = binomial_price(S, K, r, sigma, T=days_to_expiry / 365, n=step, option_type="call", q=q)
            values[f"{{{{CONV_N{step}}}}}"] = f"{p:.2f}"
        bs_ref = black_scholes(S, K, r, sigma, days_to_expiry / 365, "call", q)
        p500 = float(values["{{CONV_N500}}"])
        values["{{CONV_DIFF}}"] = f"{abs(p500 - bs_ref):.2f}"
    else:
        for step in [3, 5, 10, 25, 50, 100, 200, 500]:
            values[f"{{{{CONV_N{step}}}}}"] = "N/A"
        values["{{CONV_DIFF}}"] = "N/A"

    # Put-call parity
    # NOTE: each row's LHS (C-P) is computed independently from that row's own
    # model call/put prices, and compared against the theoretical RHS. Do not
    # reuse the RHS as a stand-in for LHS -- that would make the table assert
    # parity instead of actually testing it, and would silently hide any future
    # bug that broke agreement between the call and put pricing paths.
    rhs = S * np.exp(-q * days_to_expiry / 365) - K * np.exp(-r * days_to_expiry / 365)
    values["{{PARITY_VAL}}"] = f"{rhs:.2f}"

    parity_rows = {
        "HIST": ("binomial_call_hist_vol", "binomial_put_hist_vol"),
        "PUBLISHED": ("binomial_call_published_iv", "binomial_put_published_iv"),
        "SOLVED": ("binomial_call_solved_iv", "binomial_put_solved_iv"),
        "BS_HIST": ("bs_call_hist_vol", "bs_put_hist_vol"),
        "BS_PUBLISHED": ("bs_call_published_iv", "bs_put_published_iv"),
        "BS_SOLVED": ("bs_call_solved_iv", "bs_put_solved_iv"),
    }
    for label, (call_key, put_key) in parity_rows.items():
        if call_key in results and put_key in results:
            lhs = results[call_key] - results[put_key]
            values[f"{{{{PARITY_LHS_{label}}}}}"] = f"{lhs:.2f}"
            values[f"{{{{PARITY_DIFF_{label}}}}}"] = f"{abs(lhs - rhs):.2f}"
        else:
            values[f"{{{{PARITY_LHS_{label}}}}}"] = "N/A"
            values[f"{{{{PARITY_DIFF_{label}}}}}"] = "N/A"

    # Greeks: historical and solved IV.
    T = days_to_expiry / 365
    if sigma is not None:
        g = bs_greeks(S, K, r, sigma, T, q)
        values.update({
            "{{DELTA_CALL_HIST}}": f"{g['delta_call']:.3f}",
            "{{DELTA_PUT_HIST}}": f"{g['delta_put']:.3f}",
            "{{GAMMA_HIST}}": f"{g['gamma']:.4f}",
            "{{VEGA_HIST}}": f"{g['vega']:.3f}",
            "{{THETA_CALL_HIST}}": f"{g['theta_call']:.3f}",
            "{{THETA_PUT_HIST}}": f"{g['theta_put']:.3f}",
            "{{RHO_CALL_HIST}}": f"{g['rho_call']:.3f}",
            "{{RHO_PUT_HIST}}": f"{g['rho_put']:.3f}",
        })
    else:
        for p in [
            "{{DELTA_CALL_HIST}}", "{{DELTA_PUT_HIST}}", "{{GAMMA_HIST}}",
            "{{VEGA_HIST}}", "{{THETA_CALL_HIST}}", "{{THETA_PUT_HIST}}",
            "{{RHO_CALL_HIST}}", "{{RHO_PUT_HIST}}",
        ]:
            values[p] = "N/A"

    if not np.isnan(solved_iv):
        g = bs_greeks(S, K, r, solved_iv, T, q)
        values.update({
            "{{DELTA_CALL_SOLVED}}": f"{g['delta_call']:.3f}",
            "{{DELTA_PUT_SOLVED}}": f"{g['delta_put']:.3f}",
            "{{GAMMA_SOLVED}}": f"{g['gamma']:.4f}",
            "{{VEGA_SOLVED}}": f"{g['vega']:.3f}",
            "{{THETA_CALL_SOLVED}}": f"{g['theta_call']:.3f}",
            "{{THETA_PUT_SOLVED}}": f"{g['theta_put']:.3f}",
            "{{RHO_CALL_SOLVED}}": f"{g['rho_call']:.3f}",
            "{{RHO_PUT_SOLVED}}": f"{g['rho_put']:.3f}",
        })

    # Sensitivity uses the historical-volatility headline price.
    if sigma is not None:
        base = black_scholes(S, K, r, sigma, T, "call", q)

        values["{{SENS_Q_BASE}}"] = f"₹{base:.2f}"
        q_deltas = []
        for label, q_value in [("2", 0.02), ("35", 0.035), ("5", 0.05)]:
            p = black_scholes(S, K, r, sigma, T, "call", q_value)
            delta = p - base
            q_deltas.append(delta)
            values[f"{{{{SENS_Q_{label}}}}}"] = f"{p:.2f}"
            values[f"{{{{SENS_Q_{label}_D}}}}"] = f"{delta:+.3f}"

        # Narrative figures: derived from the actual +1pp shifts above, not
        # hand-typed, so the "roughly ₹X per pp" commentary in the report
        # stays in sync with the tables on every run.
        vol_p100 = black_scholes(S, K, r, sigma + 0.01, T, "call", q) - base
        rate_p100 = black_scholes(S, K, r + 0.01, sigma, T, "call", q) - base
        values["{{SENS_VOL_PER_PP}}"] = f"{abs(vol_p100):.2f}"
        values["{{SENS_RATE_PER_PP}}"] = f"{abs(rate_p100):.2f}"
        if abs(rate_p100) > 1e-9:
            values["{{SENS_VOL_VS_RATE_RATIO}}"] = f"{abs(vol_p100) / abs(rate_p100):.0f}"
        else:
            values["{{SENS_VOL_VS_RATE_RATIO}}"] = "N/A"

        abs_q_deltas = [abs(d) for d in q_deltas]
        values["{{SENS_Q_RANGE_LOW}}"] = f"{min(abs_q_deltas):.2f}"
        values["{{SENS_Q_RANGE_HIGH}}"] = f"{max(abs_q_deltas):.2f}"
    else:
        for ph in [
            "{{SENS_VOL_PER_PP}}", "{{SENS_RATE_PER_PP}}", "{{SENS_VOL_VS_RATE_RATIO}}",
            "{{SENS_Q_RANGE_LOW}}", "{{SENS_Q_RANGE_HIGH}}", "{{SENS_Q_BASE}}",
        ]:
            values[ph] = "N/A"
        for label in ["2", "35", "5"]:
            values[f"{{{{SENS_Q_{label}}}}}"] = "N/A"
            values[f"{{{{SENS_Q_{label}_D}}}}"] = "N/A"

    # --- Volatility skew (Section 9): computed from the actual chain data,
    # restricted to liquid strikes, instead of hand-typed percentages that
    # only ever described one specific past run. See compute_skew_stats().
    def _pct_or_na(x):
        return f"{x * 100:.1f}%" if x is not None else "N/A"

    if skew_stats is not None:
        values["{{SKEW_CALL_IV_MIN}}"] = _pct_or_na(skew_stats.get("call_iv_min"))
        values["{{SKEW_CALL_IV_MAX}}"] = _pct_or_na(skew_stats.get("call_iv_max"))
        values["{{SKEW_PUT_IV_MIN}}"] = _pct_or_na(skew_stats.get("put_iv_min"))
        values["{{SKEW_PUT_IV_MAX}}"] = _pct_or_na(skew_stats.get("put_iv_max"))
        atm_gap = skew_stats.get("atm_gap")
        if atm_gap is not None:
            values["{{SKEW_ATM_GAP}}"] = f"{abs(atm_gap) * 100:.1f}"
            values["{{SKEW_ATM_GAP_STRIKE}}"] = f"{skew_stats['atm_gap_strike']:.0f}"
            values["{{SKEW_ATM_GAP_SIDE}}"] = "put" if atm_gap > 0 else "call"
        else:
            values["{{SKEW_ATM_GAP}}"] = "N/A"
            values["{{SKEW_ATM_GAP_STRIKE}}"] = "N/A"
            values["{{SKEW_ATM_GAP_SIDE}}"] = "N/A"
        if skew_stats.get("filtered"):
            values["{{SKEW_LIQUIDITY_NOTE}}"] = (
                "restricted to strikes that clear the OI/Volume liquidity filter"
            )
        elif skew_stats.get("call_filtered") or skew_stats.get("put_filtered"):
            missing_side = "put" if skew_stats.get("call_filtered") else "call"
            values["{{SKEW_LIQUIDITY_NOTE}}"] = (
                f"restricted to the liquidity filter on the side that had OI/Volume "
                f"columns; the {missing_side} side had none, so those figures are "
                f"based on all quoted {missing_side} strikes"
            )
        else:
            values["{{SKEW_LIQUIDITY_NOTE}}"] = (
                "based on all quoted strikes -- the chain CSV had no OI/Volume "
                "columns to filter on, so thin or stale quotes are not excluded"
            )
    else:
        for ph in [
            "{{SKEW_CALL_IV_MIN}}", "{{SKEW_CALL_IV_MAX}}", "{{SKEW_PUT_IV_MIN}}",
            "{{SKEW_PUT_IV_MAX}}", "{{SKEW_ATM_GAP}}", "{{SKEW_ATM_GAP_STRIKE}}",
            "{{SKEW_ATM_GAP_SIDE}}",
        ]:
            values[ph] = "N/A"
        values["{{SKEW_LIQUIDITY_NOTE}}"] = "not available for this run"

    # --- Directional comparisons (Sections 9 & 15): which side of the market
    # premium each model volatility lands on. Previously asserted as fixed
    # prose ("historical vol overprices, IV underprices") that only held for
    # one specific run's numbers and broke the moment relative vol levels
    # differed (as happened on the 2026-08-18 run).
    def _direction_word(model_price, market_price, tol_frac=0.02, tol_min=0.5):
        tol = max(tol_min, tol_frac * market_price)
        diff = model_price - market_price
        if abs(diff) <= tol:
            return "roughly matches"
        return "overprices" if diff > 0 else "underprices"

    def _comparative_phrase(model_price, market_price, tol_frac=0.02, tol_min=0.5):
        tol = max(tol_min, tol_frac * market_price)
        diff = model_price - market_price
        if abs(diff) <= tol:
            return "about the same option value as"
        return "a higher option value than" if diff > 0 else "a lower option value than"

    if "bs_call_hist_vol" in results:
        values["{{HIST_VS_MARKET_DIR}}"] = _direction_word(results["bs_call_hist_vol"], market_ltp)
        values["{{HIST_VS_MARKET_COMPARATIVE}}"] = _comparative_phrase(results["bs_call_hist_vol"], market_ltp)
    else:
        # These placeholders sit mid-sentence as a verb ("... volatility
        # {{HIST_VS_MARKET_DIR}} the ATM call ..."), so a bare "N/A" reads
        # as broken English when historical vol wasn't available for this
        # run (e.g. --skip-yfinance). Keep the substitute grammatical.
        values["{{HIST_VS_MARKET_DIR}}"] = "cannot be compared to"
        values["{{HIST_VS_MARKET_COMPARATIVE}}"] = "no comparable option value to"

    if "bs_call_published_iv" in results:
        values["{{PUBLISHED_VS_MARKET_DIR}}"] = _direction_word(results["bs_call_published_iv"], market_ltp)
    else:
        values["{{PUBLISHED_VS_MARKET_DIR}}"] = "cannot be compared to"

    return values


# =========================================================
# 11. MAIN
# =========================================================

def build_arg_parser():
    p = argparse.ArgumentParser(description="TCS options pricing & volatility analysis")
    p.add_argument("--symbol", default="TCS.NS")
    p.add_argument("--stock-csv", default="tcs_stock_data.csv")
    p.add_argument("--chain-csv", default="option_chain_data.csv")
    p.add_argument("--risk-free-rate", type=float, default=0.0692)
    p.add_argument("--dividend-yield", type=float, default=0.0)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--expiry", default=None, help="YYYY-MM-DD override")
    p.add_argument("--expiry-weekday", type=int, default=1)
    p.add_argument("--min-oi", type=int, default=100)
    p.add_argument("--min-volume", type=int, default=1)
    p.add_argument("--output-dir", default="outputs",
                    help="Top-level directory. Each run creates its own timestamped "
                         "subfolder inside this (report, charts, sensitivity CSV all "
                         "live there). run_history.csv is the one exception -- it's "
                         "cumulative, so it stays directly in this top-level folder. "
                         "Created automatically if it doesn't exist.")
    p.add_argument("--log-csv", default="run_history.csv")
    p.add_argument("--run-date", default=None, help="YYYY-MM-DD snapshot date; defaults to latest stock CSV date")
    p.add_argument("--report-template", default="TCS_Options_Pricing_Report_template.docx")
    p.add_argument("--report-docx", default=None, help="Optional report filename/path")
    p.add_argument("--skip-yfinance", action="store_true")
    return p


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Every run gets its own timestamped subfolder inside --output-dir, so
    # a run's report/charts/sensitivity CSV are grouped together and two
    # runs can never land in the same folder. run_history.csv is the one
    # exception -- it's a cumulative log meant to span every run, so it
    # stays at the top level of --output-dir, not inside this subfolder.
    run_started_at = datetime.now()
    run_folder_name = run_started_at.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.output_dir, run_folder_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run output folder = {run_dir}")

    # ---- Stock snapshot ----
    stock_df = pd.read_csv(args.stock_csv)
    stock_df["CLOSE"] = pd.to_numeric(
        stock_df["CLOSE"].astype(str).str.replace(",", ""), errors="coerce"
    )
    stock_df["DATE"] = pd.to_datetime(stock_df["DATE"], format="%d-%b-%Y")
    stock_df = stock_df.dropna(subset=["DATE", "CLOSE"]).sort_values("DATE", ascending=False)
    if stock_df.empty:
        raise ValueError("No valid observations in stock CSV.")

    snapshot_date = stock_df["DATE"].iloc[0].date()
    run_date = date.fromisoformat(args.run_date) if args.run_date else snapshot_date

    # ---- Expiry ----
    expiry = resolve_expiry(run_date, args.expiry, args.expiry_weekday)
    days_to_expiry = (expiry - run_date).days
    T = days_to_expiry / 365

    print(f"Exercise style = {EXERCISE_STYLE}")
    print(f"Run date       = {run_date}")
    print(f"Expiry         = {expiry} ({days_to_expiry} days)")

    # ---- Historical volatility ----
    sigma = mean_return = returns = None
    if not args.skip_yfinance:
        try:
            returns, mean_return, sigma = get_historical_volatility(args.symbol, end_date=run_date)
            print(f"Mean return (descriptive) = {mean_return:.4f}")
            print(f"Historical volatility     = {sigma:.4f}")
        except Exception as exc:
            print(f"[WARN] Historical vol unavailable: {exc}")

    S = float(stock_df["CLOSE"].iloc[0])
    r = args.risk_free_rate
    q = args.dividend_yield
    n = args.n_steps

    # ---- Option chain and ATM selection ----
    chain, liquidity_info = load_option_chain(args.chain_csv)
    if not liquidity_info["call"]:
        print("[WARN] Chain CSV missing call OI/Volume columns; ATM selection "
              "falls back to the IV-bound heuristic.")
    if not liquidity_info["put"]:
        print("[WARN] Chain CSV missing put OI/Volume columns; put-side skew "
              "stats fall back to the IV-bound heuristic.")

    atm_row, atm_liquidity_filtered = select_atm_strike(
        chain, S, liquidity_info["call"], args.min_oi, args.min_volume
    )
    # IMPORTANT FIX: K must be the selected option-chain strike.
    # Previously K=S was set before ATM selection, creating a mismatch when
    # the nearest liquid ATM strike differed from spot (e.g. S=2264, K=2260).
    K = float(atm_row["strike"])
    market_ltp = float(atm_row["call_ltp"])
    published_iv = float(atm_row["call_iv"])

    skew_stats = compute_skew_stats(chain, liquidity_info, K, args.min_oi, args.min_volume)

    print(f"S              = {S:.2f}")
    print(f"Selected K     = {K:.2f}")
    print(f"Market call LTP= {market_ltp:.2f}")
    print(f"Published IV   = {published_iv:.4f}")

    solved_iv = implied_volatility(market_ltp, S, K, r, T, "call", q)
    print(f"Solved IV      = {solved_iv:.4f}")

    # ---- Model prices ----
    results = {}
    volatility_inputs = [
        ("hist_vol", sigma),
        ("published_iv", published_iv),
        ("solved_iv", solved_iv),
    ]

    for label, vol in volatility_inputs:
        if vol is None or np.isnan(vol):
            continue
        results[f"binomial_call_{label}"] = binomial_price(S, K, r, vol, T, n, "call", q)
        results[f"binomial_put_{label}"] = binomial_price(S, K, r, vol, T, n, "put", q)
        results[f"bs_call_{label}"] = black_scholes(S, K, r, vol, T, "call", q)
        results[f"bs_put_{label}"] = black_scholes(S, K, r, vol, T, "put", q)

    print("\n--- Model prices ---")
    for key, value in results.items():
        print(f"{key:30s} = {value:.2f}")

    # ---- Greeks on historical and solved IV ----
    if sigma is not None:
        print("\n--- Greeks: historical volatility ---")
        for key, value in bs_greeks(S, K, r, sigma, T, q).items():
            print(f"{key:15s} = {value:.4f}")

    if not np.isnan(solved_iv):
        print("\n--- Greeks: solved IV ---")
        for key, value in bs_greeks(S, K, r, solved_iv, T, q).items():
            print(f"{key:15s} = {value:.4f}")

    # ---- Sensitivity CSV ----
    if sigma is not None:
        sens = sensitivity_table(S, K, r, sigma, T, q)
        # Lives inside this run's own timestamped folder now, so it can't
        # collide with another run's table -- unique_path() stays as a
        # cheap belt-and-suspenders check in case two runs land in the
        # same folder (e.g. a fixed --output-dir path reused deliberately).
        sens_path = unique_path(os.path.join(run_dir, "sensitivity_analysis.csv"))
        sens.to_csv(sens_path, index=False)
        print(f"Sensitivity table written to {sens_path}")

    # ---- Run history ----
    # This is the one file that stays at the top level of --output-dir
    # rather than inside run_dir -- it's a single cumulative log meant to
    # span every run, not a per-run artifact. output_folder records which
    # subfolder this row's report/charts/sensitivity CSV actually landed in.
    log_record = {
        "run_timestamp": pd.Timestamp.now().isoformat(),
        "script_version": SCRIPT_VERSION,
        "output_folder": run_folder_name,
        "run_date": str(run_date),
        "expiry": str(expiry),
        "S": S,
        "K": K,
        "r": r,
        "q": q,
        "T": T,
        "hist_vol": sigma,
        "published_iv": published_iv,
        "solved_iv": solved_iv,
        "market_ltp": market_ltp,
        "atm_liquidity_filtered": atm_liquidity_filtered,
        **results,
    }
    log_path = os.path.join(args.output_dir, args.log_csv)
    log_run(log_path, log_record)
    print(f"Run appended to {log_path}")

    # ---- Word report ----
    report_values = build_report_values(
        run_date, expiry, stock_df, returns, mean_return, sigma,
        S, K, r, q, n, days_to_expiry, published_iv, solved_iv,
        market_ltp, results, skew_stats,
    )

    if args.report_docx:
        report_path = args.report_docx
        if not os.path.isabs(report_path):
            report_path = os.path.join(run_dir, report_path)
    else:
        report_path = os.path.join(run_dir, "TCS_Options_Pricing_Report.docx")
    report_path = unique_path(report_path)

    try:
        generate_word_report(args.report_template, report_path, report_values)
        print(f"Word report written to {report_path}")
    except Exception as exc:
        print(f"[WARN] Word report generation skipped: {exc}")

    # ---- Charts ----
    if sigma is not None:
        steps_list = [3, 5, 10, 25, 50, 100, 200, 500]
        prices = [binomial_price(S, K, r, sigma, T, s, "call", q) for s in steps_list]
        bs_ref = black_scholes(S, K, r, sigma, T, "call", q)

        plt.figure(figsize=(8, 5))
        plt.plot(steps_list, prices, marker="o", label="Binomial Price")
        plt.axhline(y=bs_ref, linestyle="--", label="Black-Scholes Price")
        plt.xscale("log")
        plt.xlabel("Number of Binomial Steps (log scale)")
        plt.ylabel("Call Option Price (₹)")
        plt.title("Binomial Convergence to Black-Scholes")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        convergence_path = unique_path(os.path.join(run_dir, "convergence_plot.png"))
        plt.savefig(convergence_path)
        plt.close()
        print(f"Convergence chart written to {convergence_path}")

        rolling_30 = returns.rolling(30).std() * np.sqrt(252)
        rolling_90 = returns.rolling(90).std() * np.sqrt(252)
        plt.figure(figsize=(10, 5))
        plt.plot(rolling_30.index, rolling_30, label="30-Day Rolling Vol")
        plt.plot(rolling_90.index, rolling_90, label="90-Day Rolling Vol")
        plt.axhline(y=sigma, linestyle="--", label="1-Year Historical Vol")
        plt.axhline(y=published_iv, linestyle=":", label="ATM IV (NSE)")
        plt.xlabel("Date")
        plt.ylabel("Annualised Volatility")
        plt.title("Rolling Volatility of TCS Returns")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        rolling_vol_path = unique_path(os.path.join(run_dir, "rolling_volatility.png"))
        plt.savefig(rolling_vol_path)
        plt.close()
        print(f"Rolling volatility chart written to {rolling_vol_path}")

    # Same trustworthy-strike filter used for the report's Section 9 skew
    # stats (see _filter_trustworthy_strikes docstring) -- previously this
    # chart used an independent, cruder filter (a flat 5%-100% IV bound with
    # no liquidity or moneyness check), so it could show a different set of
    # strikes than the numbers the report text actually quotes.
    call_plot = _filter_trustworthy_strikes(
        chain, "call_iv", "call_oi", "call_volume", liquidity_info["call"], K,
        args.min_oi, args.min_volume,
    )
    put_plot = _filter_trustworthy_strikes(
        chain, "put_iv", "put_oi", "put_volume", liquidity_info["put"], K,
        args.min_oi, args.min_volume,
    )

    plt.figure(figsize=(11, 5))
    if not call_plot.empty:
        plt.plot(call_plot["strike"], call_plot["call_iv"] * 100, marker="o", label="Call IV")
    if not put_plot.empty:
        plt.plot(put_plot["strike"], put_plot["put_iv"] * 100, marker="o", label="Put IV")
    plt.axvline(x=S, linestyle="--", label=f"Spot (₹{round(S, 0)})")
    plt.axvline(x=K, linestyle=":", label=f"Selected ATM strike (₹{round(K, 0)})")
    plt.xlabel("Strike Price (₹)")
    plt.ylabel("Implied Volatility (%)")
    plt.title(f"TCS Implied Volatility Curve — {expiry.isoformat()} Expiry")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    smile_path = unique_path(os.path.join(run_dir, "volatility_smile.png"))
    plt.savefig(smile_path)
    plt.close()
    print(f"Volatility smile chart written to {smile_path}")


if __name__ == "__main__":
    main()
