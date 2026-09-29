# ⚡ EnerVision AI
### Smart Energy Forecasting & Automated Demand Management System

EnerVision AI helps solar-powered homes manage their energy intelligently instead of reactively.

**The problem:** Solar generation usually peaks at midday, when household demand is low, and demand rises in the evening, right when solar output drops. Traditional home energy systems only react *after* this imbalance happens — wasting clean energy and increasing grid dependency.

**Our solution:** A two-part system:
- **Now** — a real-time status classifier that computes the current Net Energy Balance (Solar − Demand) directly from live data, and labels it Surplus / Balance / Shortage.
- **Predict 24h** — an AI model that forecasts household Demand 24 hours ahead, combined with a historical-average solar estimate, so the system can recommend actions *before* an imbalance occurs.

We benchmarked three models — XGBoost, Bi-LSTM, and PatchTSMixer — and evaluated them with a fair, task-matched 24-hour test. **PatchTSMixer** was selected as the final model: it forecasts all 24 hours natively in one pass, while XGBoost's recursive approach broke down over a full day despite strong short-term accuracy (see [Results](#-results-summary) below).

The final model is deployed in an interactive **Streamlit dashboard**.

---

## 🗂️ Data Source

**Smart Home Dataset with Weather Information** (`HomeC.csv`), originally collected and published by [Taranvee on Kaggle](https://www.kaggle.com/datasets/taranvee/smart-home-dataset-with-weather-information).

It contains one year of **minute-resolution** household energy data (whole-house usage, solar generation, and per-appliance sub-meters) alongside co-located weather observations (temperature, humidity, pressure, cloud cover, wind speed, etc.) for a single smart home.

> The raw dataset is not included in this repository due to its size. Download it from the Kaggle link above, place it in `Data/`, and run `SDA_EnerVision_AI_DataPreparation.ipynb` to generate `HomecCleaned.csv`, used by the models and the dashboard.

---

## 📁 Repository Structure

```
EnerVision_AI/
├── SDA_EnerVision_AI_DataPreparation.ipynb   # Cleaning, EDA, feature engineering
│
├── Data/
│   ├── HomeC.csv                             # Raw dataset (download from Kaggle)
│   └── HomecCleaned.csv                      # Cleaned dataset (output of the prep notebook)
│
├── Models/
│   ├── XGBoost/
│   │   ├── EnerVisionAI_XGBoost_Model.ipynb
│   │   └── xgboost_Ver2.pkl                  # Trained pipeline (benchmark only, not deployed)
│   ├── BiLSTM/
│   │   ├── EnerVisionAI_BiLSTM_Model.ipynb
│   │   ├── bilstm_model.pt
│   │   ├── scaler_X.joblib
│   │   └── scaler_y.joblib
│   ├── PatchTSMixer/
│   │   ├── EnerVisionAI_PatchTSMixer_Model.ipynb
│   │   ├── patchtsmixer_model.pt             # Deployed model weights
│   │   └── patchtsmixer_scaler.joblib        # Deployed model scaler
│   ├── Visualizations & Model Comparison.ipynb
│   ├── EnerVisionAI_Fair24h_Comparison.ipynb # Fair recursive-vs-native 24h test
│   ├── model_results.csv                     # Each model's native-task metrics
│   └── model_results_24h_fair.csv            # Fair 24h comparison metrics
│
└── UI/
    ├── EnerVision_AI_app.py                  # Streamlit dashboard (run this)
    ├── requirements.txt
    ├── logo.png
    ├── HomecCleaned.csv                      # Copy of the cleaned dataset used by the app
    ├── patchtsmixer_model.pt                 # Copy of the deployed model
    └── patchtsmixer_scaler.joblib            # Copy of the deployed scaler
```

> ⚠️ **Before pushing to GitHub**, make sure `UI/enervision_env/`, `UI/enervision2_env/` (virtual environments) and `UI/.agents/` are **excluded** — add them to `.gitignore`. They are large, machine-specific, and should never be committed.
>
> ```
> # .gitignore
> enervision_env/
> enervision2_env/
> .agents/
> __pycache__/
> *.pyc
> ```

---

## 🚀 How to Run the Dashboard

### 1. Get the project files
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>/UI
```

### 2. Create a virtual environment

**macOS / Linux (Terminal):**
```bash
python3 -m venv enervision_env
source enervision_env/bin/activate
```

**Windows (Command Prompt or PowerShell):**
```bash
python -m venv enervision_env
enervision_env\Scripts\activate
```

### 3. Install dependencies (same command on both systems)
```bash
pip install -r requirements.txt
```

### 4. Run the app

**macOS / Linux:**
```bash
python3 -m streamlit run EnerVision_AI_app.py
```

**Windows:**
```bash
python -m streamlit run EnerVision_AI_app.py
```

The dashboard opens automatically in your browser (usually `http://localhost:8501`).

> ⚠️ `EnerVision_AI_app.py` expects `HomecCleaned.csv`, `patchtsmixer_model.pt`, `patchtsmixer_scaler.joblib`, and `logo.png` to be in the **same folder** (`UI/`). If you rename or move the cleaned CSV, update the filename inside `EnerVision_AI_app.py` to match.

---

## 📊 Results Summary

Models were first benchmarked on their own native task, then re-evaluated on a **fair 24-hour test** (same 276 anchor points, same ground truth) to check how they actually perform on the dashboard's real job.

| Model | Native-task MAE | Fair 24h MAE | Fair 24h R² |
|---|---|---|---|
| XGBoost (recursive) | 0.134 kW | 0.451 kW | **−0.46** |
| Bi-LSTM | 0.214 kW | *not evaluated (no hourly deployment path)* | — |
| **PatchTSMixer (native)** | 0.300 kW | **0.301 kW** | **0.150** |

**Key finding:** XGBoost's strong short-term accuracy did not hold up over a full 24-hour forecast — its recursive method performed worse than simply predicting the historical average. PatchTSMixer, forecasting all 24 hours natively in a single pass, was the only model that held up — which is why it was selected for deployment.

---

## 🧠 Models

| Model | Type | Input | Output |
|---|---|---|---|
| XGBoost | Gradient-boosted trees | Single row of features | Next minute |
| Bi-LSTM | Recurrent neural network | 24-minute window | Next minute |
| PatchTSMixer | Transformer-style (patch-based) | 96-hour window | Full 24 hours, one pass |

---

## ⚠️ Known Limitations

- PatchTSMixer's accuracy is modest (R² ≈ 0.15) — limited by the small amount of hourly training data (~8,000 rows after resampling).
- Bi-LSTM has no hourly/recursive deployment path, so it was not included in the fair 24-hour test.
- The dashboard's future Solar estimate is a historical average (climatology), not a model prediction.
- All models were trained on a single household — generalization to other homes is untested.
- Bi-LSTM and PatchTSMixer do not use a fixed random seed, so results may vary slightly between reruns.

---

## 👥 Team

Ahdab Albishri · Israa Alaryani · Norah Algethami · Reema Alamri
