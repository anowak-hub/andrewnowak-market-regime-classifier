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
| Logistic Regression (balanced) | 0.32 | 0.02 |