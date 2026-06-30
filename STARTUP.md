# Startup

From zero to a running dashboard. Run these once, in order, from the project root.
See `README.md` for the full reference.

## 1. Environment

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `Activate.ps1` is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

## 2. Build the data

```powershell
python prep_data.py
```

Generates `data/clean/` from `data/raw/`. Required before anything else.

## 3. Train the models

```powershell
python machine-learning/train.py
```

Saves models to `machine-learning/models/`. Required for the dashboard's machine learning tab.

## 4. Run the dashboard

```powershell
streamlit run dashboard.py
```

Opens at http://localhost:8501. Stop with Ctrl+C. Do not use `python dashboard.py`.
