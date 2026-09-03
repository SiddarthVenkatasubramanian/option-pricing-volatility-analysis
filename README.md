# TCS Options Pricing & Volatility Analysis

Cox-Ross-Rubinstein binomial and Black-Scholes option pricing for TCS
(NSE) — cross-validated against each other and against a real NSE
options-chain snapshot, with Greeks, an independent implied-volatility
solver, sensitivity analysis, and an automated Word report generated from
the same run.

This started as a straightforward "does Black-Scholes actually reproduce
the market price" question and grew into a small pricing pipeline as real
bugs kept surfacing against real data (a hardcoded expiry date that
silently breaks after it passes, an NSE exchange-holiday shift that a
naive "last Tuesday of the month" rule misses, a strike-selection bug that
quietly assumed K=S). The commit history / project notes below are honest
about what's fixed and what's still a known limitation, on purpose.

## Features

- **Binomial (CRR) and Black-Scholes pricing**, both supporting a
  continuous dividend yield `q` (defaults to 0)
- **Market-implied volatility solver** — solves directly from the
  observed market premium rather than trusting NSE's published IV as
  ground truth (Newton-Raphson, with a bisection fallback for the
  low-vega cases — deep OTM, very short-dated — where Newton is unstable)
- **Greeks** — closed-form Black-Scholes analytical Greeks (delta, gamma,
  vega, theta, rho); *not* binomial finite-difference Greeks, see
  [Notes on methodology](#notes-on-methodology)
- **Sensitivity analysis** — price response to shifts in the risk-free
  rate, volatility, and dividend yield, written to
  `sensitivity_analysis.csv` each run. The dividend-yield piece also
  appears in the Word report itself (Section 12); rate/volatility
  sensitivity don't, since they'd just restate what the report's Greeks
  section (rho, vega) already shows — the CSV is where to look for the
  full rate/vol grid
- **Convergence analysis** — binomial price vs. step count, converging to
  the Black-Scholes limit
- **Put-call parity validation** — for both models, with and without a
  dividend yield
- **Liquidity-aware ATM strike selection** — nearest strike to spot is
  only used if it clears an open-interest/volume bar; otherwise the
  nearest *liquid* strike is selected instead, and this is logged
- **NSE-holiday-adjusted monthly expiry** — rolls back to the previous
  trading day when the calendar weekday lands on a known NSE holiday (see
  [Notes on methodology](#notes-on-methodology) for the current
  limitation here)
- **Volatility smile plotting, restricted to trustworthy strikes** — the
  chart and the report's skew statistics (Section 9) now share one filter:
  liquidity (OI/Volume), a moneyness bound, and an implied-volatility
  ceiling. The moneyness bound matters most: a deep-ITM strike near expiry
  can have real OI and volume yet still solve to a meaningless annualised
  IV, since almost all of its premium is intrinsic value — confirmed live
  against a real 1-day-to-expiry chain, where a liquid deep-ITM strike
  solved to 353% before this filter was added
- **Automated Word report generation** from a `.docx` template, populated
  directly from the run's own numbers — no manual copy-paste
- **Run history log** (`run_history.csv`) for tracking results across
  multiple runs over time, with automatic schema migration if the logged
  fields change between versions
- **Collision-safe output filenames** — a report or chart is never
  silently overwritten; a new run gets a new filename
- **43 automated tests** (`unittest`) covering convergence, parity (with
  and without dividends), the IV solver's edge cases, expiry calculation
  (including holiday rollback), ATM strike selection regressions, skew
  stats correctly excluding a liquid-but-implausible deep-ITM IV, and the
  run-history schema migration

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+.

## Usage

Basic run (auto-detects the latest date in your stock CSV, computes the
next NSE monthly expiry, prices everything, writes a report):

```bash
python tcs_options_pricing.py
```

With explicit inputs (recommended — matches your actual downloaded data):

```bash
python tcs_options_pricing.py \
    --stock-csv tcs_stock_data.csv \
    --chain-csv option_chain_data.csv \
    --expiry 2026-08-25 \
    --run-date 2026-08-19
```

Useful flags:

| Flag | Default | Purpose |
|---|---|---|
| `--symbol` | `TCS.NS` | yfinance ticker for historical prices |
| `--stock-csv` | `tcs_stock_data.csv` | NSE historical price export |
| `--chain-csv` | `option_chain_data.csv` | NSE options chain export |
| `--risk-free-rate` | `0.0692` | Annualised risk-free rate |
| `--dividend-yield` | `0.0` | Continuous dividend yield `q` |
| `--n-steps` | `50` | Binomial tree steps |
| `--expiry` | auto | `YYYY-MM-DD` override — use this to match your downloaded chain exactly |
| `--expiry-weekday` | `1` (Tuesday) | NSE's current monthly-expiry weekday; change if NSE's policy changes again |
| `--min-oi` / `--min-volume` | `100` / `1` | Liquidity thresholds for ATM strike selection |
| `--skip-yfinance` | off | Skip the live price download (useful offline/for testing; historical/rolling volatility won't be computed) |
| `--dividend-yield`, `--output-dir`, `--log-csv`, `--report-template`, `--report-docx` | — | See `--help` for the full list |

## Input data format

**`tcs_stock_data.csv`** — NSE historical price export with `DATE`
(`DD-MMM-YYYY`) and `CLOSE` columns (comma-formatted numbers are handled).

**`option_chain_data.csv`** — NSE option chain export (the standard
"Export" from NSE's option chain page), with a title row followed by a
header row containing `STRIKE`, `LTP`, `IV` (call side) and `LTP.1`,
`IV.1` (put side). `OI`/`VOLUME` columns (call and put) are optional but
enable liquidity-aware strike selection and skew stats — the script warns
and degrades gracefully if either side's liquidity columns are missing.

Both files are **not** committed to this repo (see `.gitignore`) — they're
point-in-time market snapshots you download fresh before each run.

## Output

Each run gets its own timestamped subfolder inside `--output-dir`
(default `outputs/2026-08-25_07-22-08/`, etc.), so a run's files are
grouped together and two runs can never collide:

- `TCS_Options_Pricing_Report.docx` — the full report
- `convergence_plot.png`, `rolling_volatility.png`, `volatility_smile.png`
- `sensitivity_analysis.csv`

`run_history.csv` is the one exception — it's a single cumulative log
meant to span every run, so it stays at the top level of `--output-dir`
rather than inside a per-run subfolder. Each row records which subfolder
that run's report/charts/CSV actually landed in.

## Testing

```bash
python -m unittest test_pricing.py -v
```

43 tests, no network or market-data files required — everything is
either a pure calculation or uses synthetic in-memory data.

## Notes on methodology (known limitations, stated plainly)

- **Greeks are Black-Scholes analytical Greeks**, not binomial-tree
  Greeks. The two would need to agree closely for a well-converged tree,
  but this project reports the closed-form values because they're exact
  rather than an approximation of an approximation.
- **NSE holiday adjustment is year-specific and manually maintained.**
  `NSE_HOLIDAY_CALENDARS` currently only covers 2026 (from NSE's official
  December 2025 circular). Running this in an unlisted year prints a
  warning and silently skips holiday adjustment rather than erroring —
  pass `--expiry` explicitly in that case. NSE publishes next year's
  calendar every December; this needs a corresponding code update then.
- **NSE's option chain is not fetched live.** An earlier version of this
  project attempted to scrape NSE's API directly; it was abandoned after
  NSE's anti-bot measures made it unreliable across networks regardless of
  headers/session handling. The current, more honest approach: download
  the chain manually from NSE's website (a real browser gets through
  fine) and pass it via `--chain-csv`.
- **Published IV vs. solved IV can disagree meaningfully.** This project
  treats NSE's published ATM IV as one data point, not ground truth, and
  solves independently from the observed market premium — the two have
  differed by several percentage points across different runs of this
  project, which is itself part of the analysis (see the report's
  volatility comparison section).
- No dividend *schedule* is modeled — `q` is a flat continuous yield,
  which is an approximation for a stock with discrete dividend dates.

## Project structure

```
tcs_options_pricing.py    # main script
test_pricing.py           # unittest suite (43 tests)
TCS_Options_Pricing_Report_template.docx   # report template (placeholders filled at runtime)
run_history.csv            # cumulative run log, committed directly (lives in outputs/)
requirements.txt
```
