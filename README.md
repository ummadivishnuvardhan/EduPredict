# EduPredict

EduPredict is a beginner-friendly Streamlit prototype that estimates final mathematics grades and offers a configurable teacher-review prompt. It uses the real UCI Student Performance mathematics data (`data/student-mat.csv`). It does not estimate a probability of failure.

## Prediction stages

- **Earlier stage:** excludes G2 and G3, using G1 plus selected study, attendance, and support indicators.
- **Later stage:** includes G1 and G2 plus the same indicators. G2 is a strong predictor, so this is a late-stage estimate.

Sex, age, and family background are omitted from model inputs. The 10/20 risk threshold is an illustrative grade threshold, not an institution-wide rule. Outputs are support signals for human review; do not use them to penalize, rank, or exclude students.

## Setup and run

Python 3.11 or 3.12 is recommended for deployment. This project was also run locally with Python 3.13. Community Cloud currently defaults to Python 3.12; use that version to match deployment.

```powershell
py -0p
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If a previous failed setup left an incomplete `.venv` folder, create the environment under a fresh name such as `.venv-clean` and substitute that name in the commands. Activation is optional; calling the environment's `python.exe` directly avoids PowerShell PATH issues. Model training is explicit and saves fitted pipelines and held-out evaluation files in `models/`. The app loads these files and does not retrain during reruns.

## Tests

```powershell
python -m pytest
```

Tests cover dataset loading, training/inference, prediction range handling, threshold logic, and batch validation.

## Evaluation design

Each stage uses a reproducible 80/20 random train/test split (`random_state=42`), appropriate because records do not form a time series. A random split still does not measure transfer to another school or year. DummyRegressor, LinearRegression, and RandomForestRegressor are compared using held-out MAE, RMSE, and R². Five-fold cross-validation on the training partition selects between the two learned models; the held-out test partition is reserved for final reporting. Metrics are not claims of real-world validation.

## Deploy to Streamlit Community Cloud

Create an empty GitHub repository first, then from this project directory:

```powershell
git init -b main
git add .gitignore README.md requirements.txt app.py src tests data/README.md data/student-mat.csv models/early.joblib models/late.joblib models/evaluation.json models/early-holdout.csv models/late-holdout.csv
git status --short
git diff --cached --check
git commit -m "Build EduPredict student performance app"
git remote add origin https://github.com/YOUR-ACCOUNT/YOUR-REPOSITORY.git
git push -u origin main
```

Before `git push`, inspect the staged file list and confirm it contains no secrets or identifying student data. The listed model and evaluation artifacts are needed at app startup and are not ignored by `.gitignore`.

In Streamlit Community Cloud, create an app from that repository, choose `app.py` as the entry point, and use Python 3.12 (the current Community Cloud default) or select another supported version in advanced settings. Deploy and verify the Home, individual prediction, and batch prediction views.

No deployment has been performed by this project setup. Do not commit identifying student data, secrets, virtual environments, or caches.

## Data attribution and limitations

See [data/README.md](data/README.md). The dataset is the mathematics subset of Cortez (2008), UCI Student Performance, licensed CC BY 4.0. It contains 395 historical Portuguese school records. It may not generalize to other schools or populations and has not been validated for real-world intervention. Uploaded records are processed in memory by the app; do not include student names, IDs, or other identifying data.
