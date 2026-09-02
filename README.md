# German Power Generation & CO₂ Forecasting

A 16-day-ahead forecasting pipeline for the German electricity system. It predicts
demand and renewable output, derives the **residual load** that thermal plants must
cover, and then forecasts how that residual is split across **fossil gas, hard coal,
and lignite** — which yields an hourly **CO₂ emissions** forecast.

The economic idea behind the last step: when a residual load has to be served, *which*
fossil plants run is a merit-order decision driven by relative fuel costs. So the final
model sees not only how much power is needed, but what gas, coal, and carbon cost.

```
                 ENTSO-E                     Open-Meteo
          (load, generation,              (weather archive
        capacity, hydro forecast)          and forecast)
                    │                            │
                    └──────────┬─────────────────┘
                               ▼
      ┌────────────────────────────────────────────────┐
      │  1. Demand        N-BEATS (level)               │
      │                 + XGBoost (weather-driven shape)│
      │  2. Solar         XGBoost → capacity factor     │
      │  3. Wind onshore  XGBoost → capacity factor     │
      │  4. Wind offshore XGBoost → capacity factor     │
      │  5. Hydro         ENTSO-E day-ahead forecast    │
      └────────────────────────────────────────────────┘
                               ▼
              Residual load = demand − renewables
                               │
              + fuel prices (TTF gas, coal, EUA carbon)
              + calendar and Fourier time features
                               ▼
      ┌────────────────────────────────────────────────┐
      │  6. Multi-output XGBoost                        │
      │     → Fossil Gas · Hard coal · Lignite (MW)     │
      └────────────────────────────────────────────────┘
                               ▼
        CO₂ = Σ generationᵢ × emission factorᵢ  (kg CO₂)
```

---

## Table of contents

- [Data sources](#data-sources)
- [The models](#the-models)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Rebuilding the data directory](#rebuilding-the-data-directory)
- [Running a forecast](#running-a-forecast)
- [Repository status](#repository-status)

---

## Data sources

| Source | What it provides | Access |
|---|---|---|
| **[ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)** | Actual generation by fuel type, actual load, installed capacity, hydro day-ahead forecast (bidding zone `DE`) | Free token — see [`.env.example`](.env.example) |
| **[Open-Meteo](https://open-meteo.com/)** | Weather archive, forecast, historical forecasts, and previous model runs | No key required |
| **Commodity prices** | TTF gas futures, API2 coal, EUA carbon allowances | CSVs under `data/`, optionally synced from Supabase |

**Weather is sampled at representative sites**, not averaged over the whole country —
each technology gets the locations that actually drive it:

| Model | Weather sites |
|---|---|
| Demand | Germany centroid (51.5 °N, 10.5 °E) |
| Solar | Germany centroid |
| Wind onshore | Hanover, Berlin, Bavaria |
| Wind offshore | North Sea, Baltic Sea |

ENTSO-E returns generation at 15-minute resolution; [`data_completer.py`](data_completer.py)
resamples to hourly and de-duplicates the repeated column names ENTSO-E emits for
multi-level fuel categories. Every `complete_*` function is **incremental** — it reads
the existing CSV, requests only the gap between its last timestamp and the target end
date, and appends. Re-running is cheap and safe.

---

## The models

### 1. Demand — N-BEATS + XGBoost, decomposed

The load forecast is deliberately split into two jobs:

- **N-BEATS** ([darts](https://unit8co.github.io/darts/)) sees 32 days of history
  (`input_chunk_length = 16×24×2`) and predicts the next 16 days. It supplies the
  **level** — where demand sits overall.
- **XGBoost** is trained on load with the **16-day block mean subtracted**
  (`groupby(Grouper(freq='16D')).transform(x - x.mean())`), so it only ever learns the
  **deviation** from that level, driven by weather, calendar, and holiday features.

At prediction time the two are added: `XGBoost deviation + mean(N-BEATS level)`. The
recurrent model handles the slow trend it is good at; the tree model handles the sharp
weather and calendar response it is good at. Features include apparent temperature,
humidity, wind speed, one-hot WMO weather conditions (thunderstorm, clear, fog, rain,
cloud, snow), and German public holidays via the `holidays` package.

### 2–4. Renewables — XGBoost on capacity factors

Solar, onshore wind, and offshore wind each get their own `XGBRegressor`
(`objective='reg:absoluteerror'`, 1500 trees, early stopping at 100 rounds).

The important modelling choice: **the target is normalized by installed capacity**
(`Solar/Capacity`, `Wind_power`), so the model learns a *capacity factor* — a physical
relationship between weather and yield that is stable over time. Predictions are then
multiplied back by hourly interpolated installed capacity from ENTSO-E. Without this,
a model trained on 2016 data would systematically under-predict 2024 output simply
because Germany has built more panels and turbines since.

Predictions are clipped at zero, and MAE is optimized rather than MSE — renewable
output is spiky, and absolute error is more robust to those spikes.

Features are weather (temperature, humidity, dew point, cloud cover at three levels,
wind speed, pressure, precipitation) plus calendar features and **Fourier terms** —
sine/cosine pairs on the daily, monthly, and annual cycles, which give the trees a
smooth, continuous encoding of periodicity instead of forcing them to split on raw
hour and month integers.

### 5. Hydro

Taken directly from the ENTSO-E day-ahead generation forecast rather than modelled.

### 6. Fossil split and CO₂

A multi-output XGBoost predicts the three fossil series simultaneously from residual
load, fuel prices, and time features. Price features are engineered in
[`Preprocessing_Data.py`](Preprocessing_Data.py): TTF gas futures are mapped from their
contract symbols to expiry dates (month codes `F`…`Z`) and summarized as mean prices
over the 3-month window before expiry, giving a leak-free view of what the market
expected. A composite `Gas_Cost = 0.5 × EUA + TTF` approximates the carbon-inclusive
cost of gas generation, and `residuals/Prices` captures the demand-to-cost ratio that
drives dispatch.

Emissions use standard factors, in kg CO₂/MWh:

| Fuel | Factor |
|---|---|
| Fossil gas | 185 |
| Hard coal | 920 |
| Lignite | 1183 |

---

## Repository layout

| File | Role |
|---|---|
| [`main.py`](main.py) | Operational driver — loads models, runs the chain, plots actual vs predicted and CO₂. Interactive: prompts for forecast vs backtest mode |
| [`data_completer.py`](data_completer.py) | All ENTSO-E and Open-Meteo ingestion, plus the per-technology feature builders. The largest and most-used module |
| [`model_training.py`](model_training.py) | `train_*` and `predict_*` for load, solar, onshore, offshore, hydro, and the residual chain |
| [`Preprocessing_Data.py`](Preprocessing_Data.py) | Commodity price preprocessing (gas futures, coal, EUA), residual construction, calendar and Fourier features |
| [`Import_Data_Supabase.py`](Import_Data_Supabase.py) | Incremental sync of price tables from Supabase into `data/*.csv` |
| [`Training_Inference.py`](Training_Inference.py) | Optuna search for the fossil-split model and a standalone CO₂ helper. **Currently inert** — see below |

Two directories are git-ignored but expected at runtime:

- **`data/`** — ~120 MB of CSVs, all regenerable (see below).
- **`models/`** — trained artefacts: five XGBoost JSONs (load, solar, onshore,
  offshore, fossil-split) and the N-BEATS checkpoint (`nbeats_load_model.pt`, 77 MB).
  Regenerable via the `train_*` functions in `model_training.py`.

---

## Getting started

```bash
git clone <this-repo>
cd german-power-generation-forecasting

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then fill in ENTSOE_API_KEY
set -a && source .env && set +a
```

`ENTSOE_API_KEY` is required — `main.py` exits with an explanatory error if it is
missing. Supabase variables are optional and only needed if you sync prices rather
than supplying the price CSVs yourself.

> **Note on credentials.** Earlier revisions of this project hard-coded API keys. They
> have been moved to environment variables and the affected keys should be treated as
> compromised and rotated.

---

## Rebuilding the data directory

`data/` is not committed. Rebuild it with the incremental completers — each call fetches
only what is missing and appends to the CSV, so an interrupted run resumes cleanly:

```python
import pandas as pd
from entsoe import EntsoePandasClient
from data_completer import *

client = EntsoePandasClient(api_key=os.environ["ENTSOE_API_KEY"])
end    = pd.Timestamp.now(tz="UTC").floor("H")
archive_url = "https://archive-api.open-meteo.com/v1/archive"

# Generation, load and residual load
gen  = complete_generation(client, end=end, start=pd.Timestamp("2016-01-01", tz="UTC"))
load = complete_load(client,       end=end, start=pd.Timestamp("2016-01-01", tz="UTC"))
resid = complete_resid(client, gen=gen, load=load, end=end)

# Installed capacity (annual → interpolated hourly)
cap = complete_capacity_data(client, end=end)

# Weather per technology
weather_load     = complete_weather_for_load(archive_url,          end=end)
weather_solar    = complete_weather_for_solar(archive_url,         end=end)
weather_onshore  = complete_weather_for_wind_onshore(archive_url,  end=end)
weather_offshore = complete_weather_for_wind_offshore(archive_url, end=end)
```

`main.py` contains worked examples of these calls (in commented blocks) with the exact
CSV paths each one reads and writes.

Then train, using the `prepare_train_data_*` builders to join capacity, weather, and
generation before fitting:

```python
from model_training import *

solar_train = prepare_train_data_solar(cap, weather_solar, gen)
solar_model = train_xgboost_solar_model(solar_train)
solar_model.save_model("models/solar_xgboost_model.json")
```

---

## Running a forecast

```bash
python main.py
```

It prompts:

```
do you want predictions or past 0(predictions) 1(past) :
```

- **`0`** — forecast forward from today. Calls `predict_residuals`, which chains load →
  solar → onshore → offshore → hydro into a residual load, joins fuel prices, and runs
  the fossil-split model.
- **`1`** — backtest a past date (prompts for `yyyy-mm-dd`). Uses the Open-Meteo
  *historical forecast* endpoints so the weather inputs are the forecasts that were
  actually available at the time, not the archive — which keeps the backtest honest.

Output: matplotlib panels of actual vs predicted generation per fuel and total CO₂,
MAE/RMSE printed per target, and a `data/comparison.csv` with the joined series.

---

## Repository status

Being straight about the state of the code, since that is more useful than a polished
façade:

- **`main.py` is a driver script, not a library.** It runs top-to-bottom with
  interactive `input()` prompts, and several stages (data completion, individual
  training runs) are preserved as commented-out blocks that document how each step was
  invoked. It is best read as an operations log — the reusable logic lives in
  `data_completer.py` and `model_training.py`.
- **`Training_Inference.py` is currently inert.** The entire file is wrapped in a
  triple-quoted string, so importing it defines nothing, and `main.py` never calls its
  functions. It is retained because it documents the Optuna search used to tune the
  fossil-split model. Unwrap it to run that search.
- **No test suite and no CI.** Correctness has been checked by eyeballing forecast
  plots against actuals.
- **Backtest metrics are not committed.** MAE/RMSE are printed at runtime rather than
  logged to a results file.

The clearest next steps would be to give `main.py` a proper CLI (`argparse` with
`--mode forecast|backtest --date …`) instead of `input()` prompts, move the hard-coded
paths and date ranges into a config file, and persist backtest metrics so model changes
can be compared over time.

---

## License

[MIT](LICENSE). Data carries its own terms: ENTSO-E Transparency Platform, Open-Meteo
(CC-BY 4.0), and your commodity price provider.
