# Market Regime Classifier

Classifies the current market regime (Bull / Bear / Sideways / High Volatility)
from price and volatility data, then backtests a simple regime-aware
allocation strategy against buy-and-hold SPY.

## Setup

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the full pipeline

```bash
python main.py
```

This will:
1. Download SPY, QQQ, VIX (cached to `data/raw/`)
2. Engineer features and label regimes (`data/processed/features.csv`)
3. Train a tuned Random Forest classifier (`models/regime_model.pkl`)
4. Evaluate it (accuracy, precision, recall, confusion matrix -> `figures/`)
5. Backtest the regime strategy vs. buy-and-hold, out-of-sample, using
   the model's actual predictions (`figures/equity_curve.png`)

## Project structure
market-regime-classifier/
├── data/
│   ├── raw/            # cached raw downloads
│   └── processed/      # engineered feature sets
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── data_loader.py         # download/cache market data
│   ├── feature_engineering.py # build features + regime labels
│   ├── train_model.py         # train classifier
│   ├── evaluate.py            # metrics + confusion matrix
│   └── backtest.py            # regime strategy vs buy & hold
├── figures/             # saved plots
├── models/              # saved model artifacts
├── requirements.txt
└── main.py               # one-click pipeline

## Regime definitions

| Regime | Rule |
|---|---|
| Bull | Forward 20-day return > +5% |
| Bear | Forward 20-day return < -5% |
| Sideways | Everything else |
| High Volatility | Top 15% of **forward-looking** 20-day volatility (overrides the above) |

Both the return-based labels and the volatility-based label are defined
using *future* data relative to each row (`close.shift(-20)` for returns,
a rolling std shifted by -20 for volatility). This was a deliberate fix —
an earlier version defined High_Volatility from *trailing* volatility,
which made it partly circular (a feature already in the model was also
driving one of the labels). See Experiments below.

## Experiments & Model Selection

Baseline model (Random Forest, `class_weight="balanced"`, ±5% Bull/Bear
thresholds) on the full feature set:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Bear | 0.05 | 0.21 | 0.08 | 38 |
| Bull | 0.21 | 0.38 | 0.27 | 147 |
| High_Volatility | 0.41 | 0.36 | 0.38 | 150 |
| Sideways | 0.89 | 0.61 | 0.73 | 733 |
| **Accuracy** | | | **0.53** | 1068 |

Confusion matrix analysis showed the model wasn't confusing Bear with Bull
(direction errors were rare) — it was mostly mistaking ordinary Sideways
days for Bear/Bull, i.e. overreacting to noise in flat markets rather than
getting the market's direction backwards.

**Experiment 1 — widen Bull/Bear thresholds to ±8%:**
Hypothesis: more decisive labels would give the model a cleaner signal.
Result: rejected. Widening the threshold shrank Bear from 162 to 31 rows
total (only 7 in the test set) and Bull from 476 to 85 — both classes
became too sparse to evaluate meaningfully. Reverted to ±5%.

**Experiment 2 — model type and class weighting (4-way comparison):**

| Config | Macro F1 | Bear F1 | Bull F1 | High_Vol F1 | Accuracy |
|---|---|---|---|---|---|
| Random Forest (balanced) | 0.39 | 0.00* | 0.07* | 0.59 | 0.82 |
| Random Forest (unweighted) | 0.31 | 0.00 | 0.00 | 0.34 | 0.82 |
| Logistic Regression (balanced) | 0.32 | 0.02 | 0.11 | 0.40 | 0.56 |
| Logistic Regression (unweighted) | 0.34 | 0.00 | 0.00 | 0.46 | 0.83 |

*\*Run under the (later-reverted) ±8% thresholds, so Bear/Bull scores here
are noisy due to low sample size — included for the RF-vs-LR and
balanced-vs-unweighted comparison, not as final numbers.*

Result: **Random Forest with `class_weight="balanced"` retained** as the
production model. Both unweighted configs scored higher raw accuracy but
only by predicting Sideways/High_Volatility almost exclusively and
abandoning Bear/Bull entirely (F1 = 0.00) — a "safe but useless" outcome.
Balanced Logistic Regression overcorrected in the other direction,
sacrificing Sideways precision without meaningfully rescuing Bear/Bull.
Balanced Random Forest was the best available tradeoff.

**Takeaway:** raw accuracy is a misleading metric on this dataset given
~70% class imbalance toward Sideways. Macro-averaged F1 and per-class
precision/recall are the metrics that actually matter here, and are what
should be used going forward when comparing future changes.

### Out-of-sample backtest (initial)

The confusion matrix for the ±5%-threshold baseline model showed that
misclassifications were mostly "safe" (real Sideways days being called
Bear/Bull) rather than directional errors (real Bear days being called
Bull, or vice versa) — a relatively benign failure mode, though still
costly.

Backtesting the baseline model's **actual predictions** (not the true
labels — see caveat below) on the held-out test period, using a simple
regime-based allocation (Bull=100%, Sideways=50%, High_Volatility=25%,
Bear=0% SPY):

| | Buy & Hold | Regime Strategy (baseline model) |
|---|---|---|
| Total return | 87.3% | 29.7% |
| Annualized return | 15.95% | 6.33% |
| Annualized volatility | 17.48% | 11.92% |
| Sharpe ratio | 0.913 | 0.531 |
| Max drawdown | -22.1% | -20.6% |

*Important methodology note:* an earlier sanity-check version of this
backtest used the **true** regime labels instead of predictions, which
produced an artificially excellent equity curve (~20x growth) because the
true labels are built from `close.shift(-20)` — i.e. they already "know"
the next 20 days. That version is not a valid performance estimate and was
replaced with this one, which uses only the model's predictions on the
untouched test period.

The strategy underperforms buy-and-hold on both raw and risk-adjusted
return, consistent with the weak Bear/Bull precision above — the model
reduces volatility somewhat but isn't accurate enough yet to time
allocation changes profitably.

**Experiment 3 — confidence-based allocation filtering:**
Hypothesis: only acting on high-confidence predictions (`predict_proba`
≥ 0.45, else default to neutral 50% Sideways allocation) would filter out
noisy guesses and improve risk-adjusted return.
Result: rejected. 64.7% of predictions cleared the threshold (mean
confidence 0.524), ruling out "stuck at neutral" as the cause — instead,
total return fell to 21.9% and max drawdown *worsened* to -26.0%. Likely
explanation: the filter can't distinguish protective low-confidence
predictions (e.g. a correct-but-unsure Bear call, which zeroes allocation
right before a real drawdown) from noisy ones, and Bear's already-low
recall (21%) meant a disproportionate share of the model's few genuine
signals got overridden back to neutral. **Takeaway: confidence and
usefulness aren't the same thing under severe class imbalance** — naive
confidence filtering isn't automatically an improvement.

**Experiment 4 — hyperparameter tuning (GridSearchCV + TimeSeriesSplit):**
Grid search over `n_estimators` (100/300/500), `max_depth` (4/6/8/10), and
`min_samples_leaf` (1/5/10), scored on macro-F1 (not accuracy, to avoid
rewarding majority-class-only configs) using `TimeSeriesSplit` (5 folds)
to keep validation chronologically after training in every fold. The
untouched test set was never seen during search.

Best params: `max_depth=8, min_samples_leaf=10, n_estimators=300`
(vs. the original untuned `max_depth=6`).

| Class | Baseline F1 | Tuned F1 |
|---|---|---|
| Bear | 0.08 | 0.08 |
| Bull | 0.27 | 0.27 |
| High_Volatility | 0.38 | 0.41 |
| Sideways | 0.73 | 0.75 |
| **Accuracy** | 0.53 | 0.56 |

Result: **adopted.** Sideways and High_Volatility both improved modestly;
Bear and Bull were unchanged. This is an informative negative result on
its own — it indicates Bear/Bull's weakness is a **data scarcity**
problem (only 162/476 total rows respectively), not a tuning problem, so
further hyperparameter search is unlikely to help those two classes
without either more data, different features, or a simpler (e.g. binary)
label scheme.

**Experiment 5 — extend historical data range (1993 vs. 2005 start):**
Hypothesis: Bear's small sample size (162 rows) was a data scarcity
problem (per Experiment 4's takeaway); extending the data back to 1993
(SPY's inception) to capture the dot-com crash would give the model more
Bear examples to learn from.

| Class | 2005-start F1 | 1993-start F1 |
|---|---|---|
| Bear | 0.08 (recall 0.16) | 0.12 (recall 0.44) |
| Bull | 0.27 | 0.30 |
| High_Volatility | 0.41 | 0.57 |
| Macro F1 | 0.38 | 0.42 |

Result: **adopted.** Bear recall nearly tripled and High_Volatility F1
improved substantially — more historical examples of real distress
clearly helped, confirming Experiment 4's diagnosis that Bear/Bull were
data-limited rather than tuning-limited.

Out-of-sample backtest also improved:

| | Buy & Hold | Regime Strategy |
|---|---|---|
| Sharpe ratio | 0.839 | 0.799 |
| Max drawdown | -33.7% | -15.6% |

*Caveat:* this isn't a perfectly clean comparison to the earlier backtest.
The train/test split is 80/20 **by row count**, so extending the data
start date shifts where the split falls — the test period here starts
around 2019 rather than 2022, meaning it now includes the COVID crash.
Buy-and-hold's much deeper drawdown (-33.7% vs. the prior test period's
-22.1%) is largely that crash appearing in the benchmark, not a change in
market behavior the model had to work harder against. The strategy's
improved drawdown avoidance is still a genuine, positive signal (same
underlying skill — reacting to real volatility spikes — improved per the
classification metrics above), but the two backtests are testing the
model against different market conditions, not just different model
versions.

**Note on methodology:** starting here, the train/test split uses a
fixed date boundary (`test_start_date="2018-01-01"` in `train_model.py`)
instead of an 80/20 row-count split. This was changed because the
percentage-based split meant the test period silently moved whenever the
amount of historical training data changed — exactly what happened
between Experiments 4 and 5. All results from this point forward are
directly comparable.

## Final Results (current)

As of Experiment 5, the train/test split changed from a row-count
percentage to a **fixed date boundary** (test period: 2018-01-01 onward,
2116 rows) — this fixes a subtle bug where extending the historical data
range was silently shifting which market conditions ended up in the test
set. All results below use this fixed boundary and are directly
comparable to each other and to any future experiment.

Test set class distribution: Sideways 1467, High_Volatility 316,
Bull 263, Bear 70.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Bear | 0.06 | 0.41 | 0.10 |
| Bull | 0.23 | 0.30 | 0.26 |
| High_Volatility | 0.51 | 0.50 | 0.51 |
| Sideways | 0.88 | 0.57 | 0.69 |
| **Accuracy** | | | **0.52** |

Out-of-sample backtest (test period: 2018-01-01 to present, covering the
Dec 2018 selloff, COVID crash, and 2022 bear market):

| | Buy & Hold | Regime Strategy |
|---|---|---|
| Total return | 218.8% | 73.8% |
| Annualized return | 14.81% | 6.81% |
| Annualized volatility | 19.19% | 9.75% |
| Sharpe ratio | 0.772 | 0.698 |
| Max drawdown | -33.7% | -14.8% |

The strategy trails buy-and-hold on raw and risk-adjusted return, but
cuts max drawdown by more than half (-14.8% vs. -33.7%). Given the
model's precision limitations (especially Bear at 0.06), this reads as a
genuine, if modest, risk-reduction result rather than a market-timing
edge — the model is more useful for damping volatility than for
capturing upside.

## Notes / next steps

- The train/test split in `train_model.py` is **chronological**, not
  shuffled, and hyperparameter tuning used `TimeSeriesSplit` for the same
  reason — this matters for time series so the model never trains or
  validates on future data relative to what it's being evaluated against.
- Labels use FORWARD-looking windows (20-day return, and now 20-day
  volatility too), which means the last 20 rows of any dataset can't be
  labeled and are dropped — expected behavior, not a bug.
- Bear and Bull remain the weakest classes (F1 0.08 and 0.27) and did not
  improve under hyperparameter tuning, pointing to a data scarcity issue
  (162 and 476 total rows respectively) rather than a modeling one.
  Candidate next steps, not yet tried:
  - **Binary reformulation** — e.g. "Bull vs. Not-Bull" or "high-risk
    (Bear+High_Vol) vs. low-risk (Bull+Sideways)" — would likely raise
    precision substantially at the cost of granularity.
  - **XGBoost comparison** — installed but not yet benchmarked against
    the tuned Random Forest.
  - **Feature pruning** — `rel_volume` (importance ~0.02) and
    `vix_change_5d` (~0.03) contribute little and may be adding noise.
  - **More/different features** — seasonality (day-of-week, month),
    sector dispersion, or macro data beyond VIX.
  - **More historical data** — extending the start date back further
    than 2005, if data quality allows, to give Bear more examples.
- Optional plotting/stats upgrades: seaborn for nicer plots, scipy for
  statistical tests on regime transitions (both already installed).