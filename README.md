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
