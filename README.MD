# 📊 Portfolio Strategy Dashboard

A Streamlit app for tracking a tiered, multi-market stock portfolio (US + India),
generating rules-based "Value Buy" / "Momentum Buy" signals for a DCA / scale-in
investing approach, and backtesting whether those signals actually have any edge.

Live app: https://pavanstocks.streamlit.app

> **Not financial advice.** This is a personal signal-tracking tool, not a
> recommendation engine. All thresholds are simplifications, all backtests are
> approximate, and every signal should be sanity-checked before acting on it.

---

## What it does

- **Portfolio Scan** — checks every ticker in your US or India tier list at
  once and flags today's signal (💎 Value Buy, 🚀 Momentum Buy, or 😴 Wait),
  so you can see what's actionable without clicking through each stock.
- **Single Stock Deep Dive** — full price chart, signal history, a simple
  forward-return backtest, and per-signal stop/target outcome tracking for
  whichever ticker you select.
- **Click-to-navigate** — click any row in a scan result table and the app
  jumps straight to that ticker's deep dive below.
- **Tunable thresholds** — every signal rule (RSI bands, volume multipliers,
  ATR stop/target multipliers) is a sidebar slider, not a hardcoded constant.

---

## How signals work

Every trading day is judged **independently** — this is a DCA / scale-in tool
(buy again each time conditions are met), not a single buy-once/hold-until-stop
system. There's no "open position" being tracked; each signal day gets its own
suggested Stop/Target based on that day's own price and ATR.

**💎 Value Buy** fires when, on a given day:
- Price is below the 20-day EMA
- RSI is in a "genuinely soft but not a falling knife" band (default 25–40)
- Volume is above its 30-day average (confirms real participation in the dip,
  not just a quiet low-volume day)
- The day itself wasn't a >3% single-day crash

**🚀 Momentum Buy** fires when:
- Price is above the 20-day EMA
- Volume is above its 30-day average by a larger multiplier (default 1.3x)
- RSI is below a cap (default 75, to avoid buying an already-extended move)

**Conviction score** (0–100) scales with how far the day's numbers exceed the
minimum threshold — it is *not* a fixed value per signal type.

**Stop / Target** are computed per-signal-day from that day's Close ± a
multiple of the 10-day ATR (defaults: 3x for stop, 6x for target) — they are
suggested risk levels for that specific tranche, not a tracked open position.

---

## Backtest & Tranche Outcomes

- **Signal Backtest** — for every historical signal, checks the actual price
  20 trading days later. Reports both **Raw Occurrences** (every signal day)
  and **Independent Episodes** (consecutive signal days within 3 trading days
  of each other collapsed into one event) — trust the Episodes count over Raw
  Occurrences, since consecutive days in the same dip aren't independent
  samples.
- **Tranche Outcomes** — retrospectively checks, for every past signal,
  whether price ever hit that signal's own stop or target level. Purely
  informational; does not affect future signals.
- **Condition Breakdown** — shows the pass rate of each individual Value Buy
  condition, so if a ticker shows zero signals you can see which specific
  threshold is the bottleneck rather than guessing.

None of this is a proper walk-forward backtest — treat it as a sanity check on
whether a signal type has *any* historical edge on a given ticker, not as a
guarantee.

---

## Managing the ticker universe (`tickers.csv`)

The full list of tickers/tiers lives in **`tickers.csv`**, not in the Python
code. To add, remove, or re-tier a stock, just edit that file and push —
no code changes needed.

**Columns:**

| Column | Description |
|---|---|
| `Market` | `US` or `India` |
| `Tier` | Plain tier label, e.g. `Tier 1: Core Compounders` or `Other / Watchlist` |
| `Name` | Display label shown in the dropdown, e.g. `Walmart (WMT)` |
| `Ticker` | The actual yfinance symbol — **include `.NS` for India tickers**, e.g. `TCS.NS` |

**To add a stock:** add a new row with the right `Market`/`Tier`/`Name`/`Ticker`.
**To add a whole new tier:** just use a new `Tier` value on some rows — it will
automatically appear as a new option in the dropdown and get picked up by the
relevant market's Portfolio Scan.
**To reorder tiers or the watchlist:** reorder the rows in the CSV — the
dropdown follows file order, not alphabetical order.
**To remove a stock:** delete its row.

The CSV must live in the same folder as `app.py` in the repo — both need to be
pushed together. Tier data is cached for 5 minutes (`ttl=300`), so edits show
up quickly without needing a full app restart.

---

## Sidebar controls

- **Portfolio Tier / Select Stock** — pick a tier (US or India, flag-prefixed)
  then a ticker within it, or choose "-- Enter Manually --" for any ticker not
  in the CSV.
- **⚙️ Signal Thresholds** *(collapsed by default)* — every RSI/volume/ATR
  threshold described above, live-tunable.
- **🧮 Interactive Risk-Reward Calculator** *(collapsed by default)* — a
  manual "what if I buy X shares at price Y" calculator, independent of the
  signal engine. Mainly useful for sizing a purchase made outside of an actual
  signal day.
- **🔄 Clear cache & retry data fetch** — forces a fresh pull from Yahoo
  Finance. Useful if a ticker fails to load (Yahoo occasionally has a
  transient hiccup on a cold start, and `st.cache_data` will otherwise cache
  that failure for up to an hour).

---

## Known limitations

- **Data source is Yahoo Finance via `yfinance`.** Smaller-cap India tickers
  in particular can have thin or gappy history — a "not enough historical
  data" warning is usually a genuine data-availability issue on Yahoo's side,
  not a bug.
- **Backtest sample sizes are often small and partially non-independent**
  (see Independent Episodes above). Don't over-trust win rates from tickers
  with only a handful of episodes.
- **No real position/trade tracking.** The app doesn't know what you actually
  bought — it only shows what the rules would have flagged. Tranche Outcomes
  and the backtest are both purely retrospective and informational.
- **This is not investment advice.** All thresholds, stop/target multipliers,
  and signal definitions are simplifications chosen for this project — they
  are not validated trading rules.

---

## Tech stack

- [Streamlit](https://streamlit.io) — UI framework
- [yfinance](https://github.com/ranaroussi/yfinance) — price data
- [pandas-ta](https://github.com/twopirllc/pandas-ta) — technical indicators (EMA, RSI, ATR)
- [Plotly](https://plotly.com/python/) — candlestick chart

Requires `streamlit>=1.35` (for the row-selection / click-to-navigate feature
on scan result tables).

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying

Push `app.py`, `tickers.csv`, and `requirements.txt` to the same folder in
your GitHub repo. If deployed via Streamlit Community Cloud connected to that
repo, pushes trigger an automatic redeploy.