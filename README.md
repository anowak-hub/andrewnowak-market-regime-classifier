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

**Experiment 6 — binary reformulation (Risk_Off vs. Risk_On), as an addition:**
Hypothesis: the 4-class model's Bear precision (0.06) was crippled by
having to discriminate against 3 other classes with very few examples
(312 Bear rows). Collapsing to two classes — Risk_Off (Bear +
High_Volatility) vs. Risk_On (Bull + Sideways), derived from the existing
`regime` column with no new thresholds — should let the model concentrate
on a single, better-populated decision boundary.

Trained and evaluated on the exact same fixed 2018-01-01 test split as
the 4-class model, for direct comparability. Kept as an addition
alongside the 4-class model, not a replacement — `regime_model.pkl` and
`regime_model_binary.pkl` are both saved.

| | Risk_Off (binary) | Bear (4-class) |
|---|---|---|
| Precision | 0.40 | 0.06 |
| Recall | 0.59 | 0.41 |
| F1 | 0.48 | 0.10 |
| Overall accuracy | 0.77 | 0.52 |

Out-of-sample backtest (Risk_On=100% SPY, Risk_Off=0% SPY):

| | Buy & Hold | 4-class Strategy | Binary Strategy |
|---|---|---|---|
| Total return | 218.8% | 73.8% | 101.0% |
| Sharpe ratio | 0.772 | 0.698 | 0.761 |
| Max drawdown | -33.7% | -14.8% | -14.0% |

Result: **adopted as the stronger of the two models.** Confirms
Experiment 4's diagnosis precisely — Bear's weakness was about having too
few examples spread across too many competing classes, not a modeling
limitation. The binary model achieves near-benchmark Sharpe ratio (0.761
vs. 0.772) while roughly halving max drawdown versus buy-and-hold — the
strongest risk-adjusted result in the project so far.

**Tradeoff:** the binary model loses the graduated allocation (100%/50%/
25%/0%) of the 4-class version — it's a blunter, all-in/all-out signal.
Both models are kept in the repo (`regime_model.pkl` and
`regime_model_binary.pkl`) so this tradeoff — nuance vs. precision — is
visible and comparable rather than hidden by picking just one.

**Experiment 7 — XGBoost vs. Random Forest (both targets):**
Hypothesis: gradient boosting might outperform Random Forest, given it's
generally a more powerful algorithm on tabular data. Tested on both the
4-class and binary targets, same fixed test period, with manual
per-sample class weighting (XGBoost has no native `class_weight="balanced"`
equivalent for multiclass — weights were computed using the standard
`n_samples / (n_classes * class_count)` formula to approximate RF's
built-in balancing).

| Target | Metric | Random Forest | XGBoost |
|---|---|---|---|
| 4-class | Macro F1 | 0.39 | 0.37 |
| 4-class | Bear F1 | 0.10 | 0.03 |
| Binary | Risk_Off F1 | 0.48 | 0.42 |
| Binary | Risk_Off recall | 0.59 | 0.46 |

Result: **rejected on both targets.** XGBoost's manual class weighting
appears less effective than sklearn's native `class_weight="balanced"` —
on the 4-class target it achieved higher raw accuracy (0.66 vs. 0.52)
only by collapsing Bear almost entirely (F1 0.03), the same "safe but
useless" failure mode as the unweighted configs in Experiment 2. On the
binary target, with identical overall accuracy, Random Forest still
caught meaningfully more real Risk_Off periods (recall 0.59 vs. 0.46).
Random Forest with `class_weight="balanced"` remains the model of record
for both targets.

## Final Results (current)

Both the original 4-class model and the binary reformulation (Experiment
6) are kept in the repo — `models/regime_model.pkl` and
`models/regime_model_binary.pkl` — since they represent a genuine
nuance-vs-precision tradeoff rather than one being a strictly better
replacement for the other.

Both are trained and evaluated on the same fixed test period
(2018-01-01 onward, 2116 rows — chosen to include the Dec 2018 selloff,
COVID crash, and 2022 bear market alongside calm stretches, per the
methodology fix in Experiment 5) for direct comparability.

### Binary model (Risk_Off vs. Risk_On) — best risk-adjusted result

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Risk_Off | 0.40 | 0.59 | 0.48 |
| Risk_On | 0.90 | 0.81 | 0.85 |
| **Accuracy** | | | **0.77** |

| | Buy & Hold | Binary Strategy |
|---|---|---|
| Total return | 218.8% | 101.0% |
| Annualized return | 14.81% | 8.67% |
| Annualized volatility | 19.19% | 11.40% |
| Sharpe ratio | 0.772 | 0.761 |
| Max drawdown | -33.7% | -14.0% |

Nearly matches buy-and-hold's Sharpe ratio while cutting max drawdown by
more than half. The strongest, most credible result in the project: not
a market-timing edge, but a real, defensible risk-reduction strategy —
close to market-level risk-adjusted return at roughly a third of the
downside.

### 4-class model (Bull / Bear / Sideways / High_Volatility) — more nuance, weaker precision

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Bear | 0.06 | 0.41 | 0.10 |
| Bull | 0.23 | 0.30 | 0.26 |
| High_Volatility | 0.51 | 0.50 | 0.51 |
| Sideways | 0.88 | 0.57 | 0.69 |
| **Accuracy** | | | **0.52** |

| | Buy & Hold | 4-class Strategy |
|---|---|---|
| Total return | 218.8% | 73.8% |
| Annualized return | 14.81% | 6.81% |
| Annualized volatility | 19.19% | 9.75% |
| Sharpe ratio | 0.772 | 0.698 |
| Max drawdown | -33.7% | -14.8% |

Trails the binary model on both return and Sharpe, largely because Bear's
very low precision (0.06) means the model frequently de-risks on false
alarms, giving up upside without a corresponding accuracy benefit.
Retained for its finer-grained allocation (100%/50%/25%/0% vs. binary's
100%/0%), which may be preferable in contexts where a graduated response
is more useful than a blunt in/out signal — that tradeoff is a judgment
call, not something the metrics alone resolve.

### Takeaway

Across all six experiments, the clearest lesson was that **Bear/Bull's
weakness was consistently a data scarcity and class-imbalance problem,
not a model-capacity one** — confirmed by hyperparameter tuning barely
moving those classes (Experiment 4) while adding more historical data
(Experiment 5) and reducing the number of competing classes (Experiment
6) both produced large, real improvements. Neither model beats
buy-and-hold on raw return, which is an honest and expected result for
a first-pass regime classifier — the binary model's near-benchmark
Sharpe ratio with substantially reduced drawdown is a legitimate,
if modest, edge.

## Notes / next steps

- The train/test split in `train_model.py` uses a **fixed date boundary**
  (`test_start_date="2018-01-01"`), not a percentage — this was a
  deliberate fix (Experiment 5) so the test period stays a controlled
  constant across experiments, regardless of how much historical data is
  loaded.
- `GridSearchCV` in the (now-deleted) tuning script used `TimeSeriesSplit`
  rather than default k-fold, for the same reason: validation must always
  come chronologically after training.
- Labels use FORWARD-looking windows (20-day return, 20-day volatility),
  so the last 20 rows of any dataset are dropped — expected, not a bug.
- Both models (`regime_model.pkl`, `regime_model_binary.pkl`) are kept
  and documented as a genuine tradeoff (precision vs. granularity), not
  because one replaces the other.

**Untried, in rough priority order:**
- **Feature pruning** — `rel_volume` (~0.02 importance) and
  `vix_change_5d` (~0.03) contribute the least; dropping them is a quick,
  low-risk experiment, unlikely to move metrics much either way.
- **Graduated binary allocation** — the binary model currently uses a
  blunt 100%/0% allocation. Using `predict_proba()` to size the position
  continuously (e.g. 100% at high Risk_On confidence, tapering toward 0%
  as confidence drops) could recover some of the 4-class model's nuance
  without reintroducing its precision problem. Note: Experiment 3 already
  showed naive confidence *filtering* backfires — this would need to be
  confidence-based *sizing* instead, a different mechanism worth testing
  carefully rather than assuming it'll work.
- **More/different features** — seasonality (day-of-week, month), sector
  dispersion, or macro data beyond VIX (e.g. Treasury yields, credit
  spreads) could give the model genuinely new information rather than
  recombinations of existing price/volatility signals.
- **Transaction costs** — the current backtest assumes frictionless
  rebalancing. Adding a per-trade cost assumption (e.g. 5-10 bps) would
  be a more realistic test of whether the binary model's drawdown
  reduction survives real-world trading costs, especially since Risk_Off
  triggers relatively often (386 of 2116 test rows).
- Optional plotting/stats upgrades: seaborn for nicer plots, scipy for
  statistical tests on regime transitions (both already installed, not
  yet used).

**Housekeeping:**
- `notebooks/exploration.ipynb` still reflects the original single-model
  setup — worth updating to also visualize the binary model's
  Risk_Off/Risk_On periods against price, alongside the existing 4-regime
  chart.
- `requirements.txt` should be re-frozen (`pip freeze > requirements.txt`)
  since xgboost/seaborn/scipy were installed after the last freeze.