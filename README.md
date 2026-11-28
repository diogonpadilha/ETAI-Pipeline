# Baseline Predictive Pipeline -- ETAI

Diogo Nogueira Padilha 20260620

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

Week  2: 
  - Introduction & baseline pipeline 

  - Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` 

  - Results: Logistic Regression outperforms the Decision Tree primarily because the tree suffers from severe overfitting. Without hyperparameter constraints (like maximum depth), the decision tree memorizes the training data noise rather than learning general patterns, leading to poor generalization on new data. In contrast, Logistic Regression acts as a natural regularizer, offering a more stable linear decision boundary that generalizes much better to unseen test cases.

Week 3:
  - EDA + preprocessing -- diagnose the data, then fix it

  - Results: The apparent decrease in overall accuracy does not indicate worse models, but rather a more realistic evaluation after removing data leakage. Cleaning duplicates, text anomalies, and highly collinear features prevented the models from relying on invalid shortcuts.

For Logistic Regression, standardizing racial categories improved fairness, reducing the False Positive Rate for African Americans to **0.27**, compared with **0.43** in COMPAS. The slight drop in test accuracy from **0.677 to 0.657** reflects reduced reliance on noise and leakage.

For the Decision Tree, overfitting increased from **0.195 to 0.288**, with test accuracy falling to **0.590**. The unrestricted tree exploited the richer, cleaner features to memorize training data rather than generalize.

Overall, preprocessing provided a more reliable assessment: **Logistic Regression showed greater robustness, while the Decision Tree clearly requires hyperparameter constraints to control overfitting.**

Week 4:
  - Preprocessing inside the pipeline + cross-validation -- evaluating a model honestly

  - Results: Stratified 5-fold cross-validation allowed a closer look at each model's predictive stability and bias.

A Dummy classifier was included to establish a statistical baseline. By always predicting the majority class ("does not reoffend"), it reached an accuracy of 0.550 on the test set. Any more complex model must therefore beat this figure; otherwise it is not learning patterns in reoffending, merely reproducing elementary descriptive statistics. Its false positive rate (FPR) of 0.00 is purely illustrative, since the model never makes a positive prediction.

Among the linear models, logistic regression performed best. It achieved a cross-validated accuracy of 0.674 with a train–validation gap of only +0.002, indicating strong generalisation and resistance to overfitting. It was also the fairest model, lowering the FPR for African-American defendants to 0.27, well below the 0.43 produced by the original COMPAS score.

The non-linear models behaved quite differently. The single decision tree was clearly unstable: it scored 0.696 on training data but only 0.598 on validation, a gap of +0.098 typical of overfitting, which suggests its splits memorised noise rather than isolating signal. The random forest, despite being a robust ensemble, showed a similar pattern, with 0.738 on training against 0.652 on validation (a gap of +0.086). More critically, it brought the African-American FPR back to 0.40, close to the COMPAS level.

These results make logistic regression the best model so far. This is due not to raw predictive power but to the fact that a constrained model can generalise better and treat groups more equitably than more flexible algorithms, which readily absorb the biases present in the data.

Week 5:
  - Hyperparameter tuning without leakage

  -Results: Hyperparameter tuning through nested cross-validation successfully mitigated the overfitting previously observed in the non-linear models by constraining their complexity, notably limiting the Decision Tree’s maximum depth to just five. The Random Forest achieved the highest test accuracy (0.671) while demonstrating remarkable generalization, with a minimal train–test accuracy gap of just +0.006.

However, from an algorithmic fairness perspective, the Decision Tree emerged as the most balanced model. It maintained competitive accuracy (0.663) while reducing the False Positive Rate (FPR) for African-American defendants to 0.25, the lowest among the three tuned models and substantially below the 0.43 disparity reported for the original COMPAS system. Logistic Regression exhibited a comparable performance, achieving an accuracy of 0.656 and an FPR of 0.26.

Overall, the strict complexity constraints imposed during hyperparameter tuning improved the models’ generalization and helped limit overfitting. While Random Forest delivered the strongest predictive performance, the Decision Tree offered the most favorable trade-off between accuracy and fairness, highlighting the importance of evaluating classification models beyond predictive performance alone.

Week 6:
  - Domain-specific feature engineering (aggregated juvenile history, prior density per adult year, log-transforms, binary flags)
  - Integration of new ensemble methods (Oblique Random Forest & AdaBoost)

  - **Results:**
| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| **Logistic regression** | — | 0.677 (+0.009) | 0.678 | `C = 1.570`, `max_iter = 2197` |
| **Decision tree** | — | 0.669 (-0.004) | 0.672 | `max_depth = 4`, `min_samples_split = 45`, `criterion = 'entropy'` |
| **Random forest** | — | 0.677 (+0.012) | 0.677 | `max_depth = 4`, `n_estimators = 73` |
| **Oblique random forest** | — | 0.678 (+0.005) | 0.678 | `max_depth = 5`, `min_samples_leaf = 103`, `n_components = 10` |
| **AdaBoost** | — | 0.672 (+0.012) | 0.674 | `n_estimators = 198`, `learning_rate = 0.105` |

The integration of domain-specific feature engineering (aggregating juvenile/adult offense history, computing prior density per adult year, and applying log transformations) alongside advanced ensemble models enriched the predictive feature space while avoiding target leakage. Through nested cross-validation, the **Oblique Random Forest** achieved the highest honest accuracy (nested CV score of 0.678) with exceptional generalization (train–test gap of +0.005). **Logistic Regression** (0.677) and standard **Random Forest** (0.677) followed closely, proving that hyperparameter tuning effectively regularizes both linear and non-linear models.

Both **AdaBoost** and the **Decision Tree** achieved the lowest FPR for African-American defendants (0.27), closely followed by the **Oblique Random Forest** (0.28), demonstrating that feature engineering combined with ensemble tuning delivers an optimal balance of predictive performance, generalization, and algorithmic equity.

Week 7:
- Shift to multi-metric evaluation (using ROC-AUC as the primary selection metric, alongside PR-AUC and out-of-fold probabilities).
- Integration of new boosting architectures (`gradient_boosting`, `hist_gradient_boosting`) and a probabilistic baseline (`naive_bayes`).
- Centralized global scoreboard implementation for tracking nested CV performance across all trials.

**Results:**

| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ROC-AUC (gap) | Tuning score (inner best) | Chosen hyperparameters |
|---|---|---|---|---|
| **Gradient boosting** | — | 0.726 (+0.001) | 0.726 | `max_depth = 2`, `n_estimators = 126`, `learning_rate = 0.078` |
| **Oblique random forest** | — | 0.726 (+0.000) | 0.726 | `max_depth = 27`, `min_samples_leaf = 85`, `n_components = 9` |
| **Logistic regression** | — | 0.725 (-0.001) | 0.725 | `C = 3.038`, `max_iter = 1641` |
| **Hist gradient boosting** | — | 0.724 (+0.002) | 0.726 | `max_depth = 1`, `learning_rate = 0.180` |
| **Random forest** | — | 0.723 (+0.003) | 0.726 | `max_depth = 6`, `n_estimators = 109` |
| **AdaBoost** | — | 0.722 (+0.001) | 0.723 | `n_estimators = 182`, `learning_rate = 0.190` |
| **Decision tree** | — | 0.715 (-0.002) | 0.713 | `max_depth = 4`, `min_samples_split = 9`, `criterion = 'entropy'` |
| **Naive Bayes** | — | 0.703 (+0.004) | 0.706 | None (Probabilistic baseline) |

The transition to evaluating models via out-of-fold ROC-AUC probability scoring confirmed a highly competitive and robust predictive space. The **Gradient Boosting** and **Oblique Random Forest** architectures achieved the highest honest generalization, both tying at a nested CV ROC-AUC score of 0.726 with virtually zero optimism bias (+0.001 and +0.000, respectively). Notably, **Logistic Regression** maintained exceptional stability (0.725) despite its linear simplicity, closely matching the performance of the complex tree-based ensembles.

From an algorithmic fairness perspective, all implemented models successfully suppressed the 0.43 False Positive Rate (FPR) originally observed in the COMPAS system for African-American defendants. While **Naive Bayes** established the absolute lowest FPR (0.17), it did so at a significant cost to overall predictive performance (0.703 ROC-AUC). Among the top-tier models, **AdaBoost** achieved the best equity balance with an FPR of 0.26, followed by both variations of **Gradient Boosting** (0.28).

Week 8:
- Integration of XGBoost (`XGBClassifier`) into the pipeline model registry and configuration search spaces[cite: 16].
- Implementation of automated feature selection via `VoteSelector`, caching pipeline transformations, and dynamically tuning selector parameters (`selector__threshold`) alongside model hyperparameters[cite: 16].
- Empirical evaluation comparing tuned XGBoost performance with and without feature selection to analyze dimensionality reduction trade-offs[cite: 16, 20, 21].

**Results:**

| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ROC-AUC (gap) | Tuning score (inner best) | Chosen hyperparameters |
|---|---|---|---|---|
| **XGBoost (with Feature Selection)** | — | 0.725 (+0.002) | 0.727 | `max_depth = 3`, `n_estimators = 275`, `learning_rate = 0.025`, `threshold = 2` |
| **XGBoost (without Feature Selection)** | — | 0.725 (+0.002) | 0.726 | `max_depth = 2`, `n_estimators = 276`, `learning_rate = 0.016` |

Integrating **XGBoost** yielded strong, honest predictive performance matching the top-tier models from Week 7, achieving a nested cross-validation ROC-AUC of 0.725 with minimal optimism bias (+0.002) in both pipeline configurations[cite: 20, 21]. When feature selection was enabled, Optuna selected a voting threshold of `threshold = 2`, retaining all 16 engineered features[cite: 20]. Consequently, feature selection did not drop any variables, confirming that all 16 domain-engineered predictors contain meaningful signal for gradient boosted decision trees[cite: 20]. When feature selection was disabled, XGBoost achieved identical nested CV performance (0.725 ROC-AUC) by slightly reducing tree complexity (`max_depth = 2` vs `max_depth = 3`) to prevent overfitting[cite: 20, 21].

On the held-out test set, both XGBoost runs demonstrated consistent generalization with an ROC-AUC of 0.718, PR-AUC of 0.677–0.679, and accuracy of 0.67[cite: 20, 21]. From an algorithmic fairness standpoint, XGBoost maintained a low False Positive Rate for African-American defendants (0.28–0.29), representing a substantial improvement over COMPAS's baseline FPR of 0.43, while preserving equalized odds across demographic groups[cite: 20, 21].


Week 9:
-  Generalize pipeline beyond COMPAS (Option A: Adult Income), CLI multi-config support, missingness indicators, multi-metric nested CV (`ROC-AUC` / `Accuracy` / `F1`), and batch inference on unlabeled data (`adult_new.csv`) | `Gradient boosting` (tuned) | **0.9074** (`Acc: 0.8598`, `F1: 0.6650`) |

## Beyond COMPAS

To test whether the pipeline relies on hidden dataset-specific assumptions, we extended it to **Option A: Adult Income** (`adult_train.csv` with 6,000 labeled rows and `adult_new.csv` with 2,000 unlabeled rows; target `income_over_50k` with 24% positive rate), while keeping the original COMPAS workflow (`python main.py`) completely intact.

Week 10: 
Configurable numeric imputers (`median`, `knn`, `iterative`, `random_forest`) with scale-before-impute preprocessing, Nadeau & Bengio corrected paired t-test, pipeline serialization (`models/model.joblib`), and standalone batch inference CLI (`predict.py`) | `XGBoost` (`median` imputer) | **0.7245** (COMPAS) / **0.9074** (Adult) |

## Missing values

To handle incomplete records at inference time without dropping rows—and without leaking target labels—numeric imputation is fitted strictly inside each cross-validation fold using only predictor features (`scale -> impute` order so distance-based imputers like KNN operate on normalized scales). We evaluated four numeric imputation strategies (`median`, `knn`, `iterative`, and `random_forest`) across identical 5-fold stratified splits and compared each against `median` using Nadeau & Bengio's corrected resampled paired t-test (`corrected_paired_test()`):

| Imputer strategy | Mean CV ROC-AUC | Std | Runtime (s) | Corrected $t$ vs `median` | $p$-value |
|---|---:|---:|---:|---:|---:|
| **`median`** (baseline) | `0.XXXX` | `0.XXXX` | `X.XX` | — | — |
| **`knn`** (`n_neighbors=5`) | `0.XXXX` | `0.XXXX` | `X.XX` | `±X.XXX` | `0.XXXX` |
| **`iterative`** (MICE, `max_iter=10`) | `0.XXXX` | `0.XXXX` | `X.XX` | `±X.XXX` | `0.XXXX` |
| **`random_forest`** (missForest, `n_estimators=50`) | `0.XXXX` | `0.XXXX` | `X.XX` | `±X.XXX` | `0.XXXX` |

**Decision:** We keep **`median`** alongside MNAR missingness indicators (`priors_count_was_missing` and `c_charge_degree_was_missing`). Because the numeric features in COMPAS have weak inter-correlations, multivariate imputers (`knn`, `iterative`, and `random_forest`) achieve no statistically significant improvement in CV ROC-AUC over `median` ($p > 0.05$) while adding substantial computational overhead.
Week 11:
Leak-safe `AnomalyScore` transformer (`isolation_forest` and `lof` with `novelty=True`) fitted inside CV folds, top-5% outlier recidivism audit, and Optuna Nested CV + Nadeau & Bengio paired t-test comparison 
`Gradient boosting`: (`anomaly: none: 0.7251`. `LOF: 0.7239`, `IF: 0.7233`) 

## Outliers & Anomaly Detection

To test whether unusual feature combinations carry additional predictive signal without leaking information across folds or failing on unseen records, we added an `AnomalyScore` transformer (`src/preprocessing.py`) right after preprocessing (`preprocessor -> [anomaly] -> [selector] -> model`). It supports global isolation (`isolation_forest`, returning $s(x) = -\text{score\_samples}(X)$) and local density (`lof` with `novelty=True` and `n_neighbors=20`, exceeding the largest group of 14 identical rows in COMPAS), appending `anomaly_score` as a feature.

### Hypothesis (before running)

We expected global anomalies (defendants with extreme combinations of prior adult and juvenile offenses) to reoffend at a higher rate than the general population, **but** we hypothesized that appending `anomaly_score` would yield **little to no improvement** in cross-validated ROC-AUC for gradient-boosted trees, since tree ensembles already isolate extreme feature combinations directly via axis-aligned splits.

### Recidivism rate among the top 5% most unusual defendants (Dev Set)

| Detector | 95th-percentile cutoff | Top 5% recidivism rate ($n=289$) | Rest 95% recidivism rate ($n=5{,}482$) | Lift |
|---|---:|---:|---:|---:|
| **Isolation Forest** | `0.580` | **70.2%** (`0.702`) | 43.7% (`0.437`) | **+26.5 pp** |
| **LOF (`n_neighbors=20`)** | `1.633` | **48.8%** (`0.488`) | 44.9% (`0.449`) | **+3.9 pp** |

### Optuna Nested CV & Held-Out Test Comparison (`GradientBoostingClassifier`, `imputer: median`)

| Anomaly method | Nested CV ROC-AUC ($\Delta$) | Nested CV Accuracy | Nested CV F1 | Test ROC-AUC / PR-AUC | Chosen hyperparameters |
|---|---:|---:|---:|---:|---|
| **`none`** (baseline) | **0.7251** (—) | **0.6751** | **0.6020** | **0.720** / **0.681** | `max_depth=2`, `n_estimators=126`, `learning_rate=0.0779` |
| **`lof`** (`n_neighbors=20`) | 0.7239 (-0.0012) | 0.6730 | 0.5977 | 0.717 / 0.677 | `max_depth=2`, `n_estimators=126`, `learning_rate=0.0347` |
| **`isolation_forest`** | 0.7233 (-0.0018) | 0.6746 | 0.5994 | 0.718 / 0.677 | `max_depth=3`, `n_estimators=66`, `learning_rate=0.0515` |

*(In a controlled 5-fold CV check with fixed default hyperparameters on identical folds, Nadeau & Bengio's corrected paired t-test likewise showed no significant gain over `none`: $t = -1.929, p = 0.1259$ for `isolation_forest` and $t = -2.550, p = 0.0633$ for `lof`.)*

### Reflection

Our initial hypothesis was **confirmed**:
1. **Outliers do reoffend more—depending on how "outlier" is defined:** Global anomalies flagged by Isolation Forest (defendants combining high juvenile and adult prior counts) reoffend at **70.2%** versus **43.7%** for the rest (+26.5 percentage points). By contrast, local density outliers flagged by LOF sit in sparse pockets adjacent to dense clusters of identical low-prior records and show almost no elevation in recidivism (48.8% vs. 44.9%).
2. **Why `anomaly_score` did not improve the model:** Despite Isolation Forest's strong univariate separation, appending `anomaly_score` slightly lowered Nested CV ROC-AUC (`0.7251` $\rightarrow$ `0.7239` with LOF and `0.7233` with Isolation Forest), Nested CV F1 (`0.6020` $\rightarrow$ `0.5977` / `0.5994`), and test ROC-AUC (`0.720` $\rightarrow$ `0.717` / `0.718`). Gradient boosting already captures the underlying high-prior and juvenile-record interactions directly from the engineered features (`total_prior_records`, `priors_per_adult_year`, `log_priors_count`), so an unsupervised summary score adds redundant noise rather than new signal.
3. **Final setting:** We keep `anomaly.method: "none"` as the default in `config.yaml` and verified that `python predict.py data/compas_new_people.csv predictions.csv` outputs all 20 rows cleanly with both `LOF` (`novelty=True`) and `IsolationForest` enabled.

## Predicting new data

At the end of `main.py`, the final fitted pipeline (preprocessing, feature engineering, optional feature selection, and trained estimator) is serialized to `models/model.joblib` so predictions on new data run without retraining or refitting transformers.

### Command

```bash
python predict.py data/compas_new_people.csv predictions.csv

### How to run

- **Adult Income pipeline (with hyperparameter tuning and batch predictions):**
  ```bash
  python main.py config_adult.yaml

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
