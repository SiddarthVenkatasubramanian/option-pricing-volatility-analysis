# TCS Options Pricing & Volatility Analysis

Cox-Ross-Rubinstein binomial and Black-Scholes option pricing for TCS
(NSE) — cross-validated against each other and against a real NSE
options-chain snapshot, with Greeks, an independent implied-volatility
solver, and an automated Word report generated from each run.

This started as a straightforward "does Black-Scholes actually reproduce
the market price" question and grew into a small pricing pipeline as real
bugs kept surfacing against real data — a hardcoded expiry date that
silently breaks once it passes, an NSE holiday/expiry-day shift a naive
calendar rule misses, a strike-selection bug that quietly assumed K=S.

## Features

- **Binomial (CRR) and Black-Scholes pricing**, both supporting a
  continuous dividend yield `q`
- **Market-implied volatility solver** — solves directly from the
  observed market premium rather than trusting NSE's published IV as
  ground truth
- **Greeks** — closed-form Black-Scholes analytical Greeks (delta, gamma,
  vega, theta, rho)
- **Sensitivity analysis** — price response to shifts in the risk-free
  rate, volatility, and dividend yield, written to
  `sensitivity_analysis.csv` each run
- **Convergence analysis** — binomial price vs. step count, converging to
  the Black-Scholes limit
- **Put-call parity validation** — for both models, with and without a
  dividend yield
- **Liquidity-aware ATM strike selection** — nearest strike to spot is
  only used if it clears an open-interest/volume bar; otherwise the
  nearest *liquid* strike is used instead
- **NSE-holiday-adjusted monthly expiry** — rolls back to the previous
  trading day when the calendar weekday lands on a known NSE holiday
- **Volatility smile plotting, restricted to trustworthy strikes** — the
  chart and the report's skew statistics share one filter: liquidity,
  moneyness, and an implied-volatility ceiling, so a stale deep-ITM quote
  can't distort the reported skew
- **Automated Word report generation** from a `.docx` template, populated
  directly from the run's own numbers — no manual copy-paste
- **Run history log** (`run_history.csv`) for tracking results across
  multiple runs over time
- **Collision-safe output filenames** — a report or chart is never
  silently overwritten

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
| `--min-oi` / `--min-volume` | `100` / `1` | Liquidity thresholds for ATM strike selection |
| `--skip-yfinance` | off | Skip the live price download (historical/rolling volatility won't be computed) |

Run `--help` for the full list of flags.

## Input data format

**`tcs_stock_data.csv`** — NSE historical price export with `DATE`
(`DD-MMM-YYYY`) and `CLOSE` columns.

**`option_chain_data.csv`** — NSE option chain export (the standard
"Export" from NSE's option chain page), with a title row followed by a
header row containing `STRIKE`, `LTP`, `IV` (call side) and `LTP.1`,
`IV.1` (put side). `OI`/`VOLUME` columns are optional but enable
liquidity-aware strike selection and skew stats.

Neither file is committed to this repo — they're point-in-time market
snapshots you download fresh before each run.

## Output

Each run gets its own timestamped subfolder inside `outputs/`, so a run's
files are grouped together and two runs can never collide:

- `TCS_Options_Pricing_Report.docx` — the full report
- `convergence_plot.png`, `rolling_volatility.png`, `volatility_smile.png`
- `sensitivity_analysis.csv`

`run_history.csv` is the exception — a single cumulative log spanning
every run, so it stays directly inside `outputs/` rather than inside a
per-run subfolder.

See `sample_outputs/` for two real generated reports from past runs, if
you'd like to see the output without running anything yourself.

## Notes on methodology (known limitations)

- **Greeks are Black-Scholes analytical Greeks**, not binomial-tree
  Greeks — reported because they're exact, not an approximation of one.
- **NSE holiday adjustment is manually maintained** and currently only
  covers 2026. Pass `--expiry` explicitly for other years.
- **NSE's option chain is not fetched live** — download it manually from
  NSE's website and pass it via `--chain-csv`.
- **Published IV vs. solved IV can disagree meaningfully** — this project
  treats NSE's published ATM IV as one data point, not ground truth, and
  solves independently from the observed market premium.
- No dividend *schedule* is modeled — `q` is a flat continuous yield.

## Project structure

```
tcs_options_pricing.py     # main script
TCS_Options_Pricing_Report_template.docx   # report template
requirements.txt
outputs/run_history.csv    # cumulative run log
sample_outputs/            # two example generated reports
```
