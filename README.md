# EduPredict

EduPredict is a beginner-friendly Streamlit prototype that estimates final mathematics grades and offers a configurable teacher-review prompt. It uses the real UCI Student Performance mathematics data (`data/student-mat.csv`). It does not estimate a probability of failure.

**Live app:** [edupredict-uvv-2026.streamlit.app](https://edupredict-uvv-2026.streamlit.app/)

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

## Deployment

The app is deployed on Streamlit Community Cloud from the public GitHub repository [ummadivishnuvardhan/EduPredict](https://github.com/ummadivishnuvardhan/EduPredict). It uses Python 3.12 and `app.py` as its entry point. Changes pushed to the repository are picked up by Streamlit Community Cloud.

Do not commit identifying student data, secrets, virtual environments, or caches. Uploaded CSV records are processed in memory; do not include student names, IDs, or other identifying data.

## Data attribution and limitations

See [data/README.md](data/README.md). The dataset is the mathematics subset of Cortez (2008), UCI Student Performance, licensed CC BY 4.0. It contains 395 historical Portuguese school records. It may not generalize to other schools or populations and has not been validated for real-world intervention. Uploaded records are processed in memory by the app; do not include student names, IDs, or other identifying data.
