# Architecture Specification: Xetra Two-Step Stock Prediction Engine (v5.0)

## 1. Project Overview
This document provides a detailed, modular architecture for a batch-processed, two-step machine learning pipeline designed to predict German stock price movements. It incorporates model diagnostics, Kolomogorov-Smirnov (KS) optimized cutoffs, 10-quantile lift charts, robust imputation, Z-scaling, comprehensive JSON reporting, and a **massive 360-degree global macroeconomic variable universe**. It is formatted to be ingested by an autonomous coding AI to generate a complete, production-ready Python repository.

**Core Parameters & Default Configurations:**
*   **Target Classes:** `UP` (1) vs `NOT_UP` (0)
*   **Prediction Horizon:** 126 trading days (~0.5 years)
*   **Target Threshold:** 15% (+0.15)
*   **Step 1 Data History:** 10 years (Target OHLCV + Global Macroeconomic Universe)
*   **Step 2 Data History:** 4 years (Fundamental financial statement data)
*   **Feature Selection Limit:** 12 variables per step
*   **Model Algorithm:** Logistic Regression (`class_weight='balanced'`, `solver='liblinear'`)
*   **Validation:** Strictly `TimeSeriesSplit` (Prevents financial look-ahead bias)
*   **Classification Cutoff:** Dynamically optimized via KS-Statistic.

---

## 2. Directory Structure

Instruct the code generator to create the following exact directory tree:

```text
xetra_predictor/
│
├── .env                     # NEW: Local environment variables (e.g., GEMINI_API_KEY)
├── config/
│   ├── __init__.py
│   ├── settings.yaml        # Centralized configuration variables
│   └── universe.py          # NEW: The 360-Degree Macro Variable Universe
│
├── data/
│   ├── raw/                 # Downloaded Xetra T7 dumps
│   └── processed/           # Filtered lists and prediction outputs
│
├── outputs/                 
│   ├── diagnostics/         # Sub-folders per ticker for Confusion Matrices / Lift Charts
│   └── predictions/         # JSON report files generated per ticker
│
├── src/
│   ├── __init__.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── xetra_t7.py      # Xetra T7 dump download and parsing
│   │   ├── global_macro.py  # NEW: Fetches and caches the 360-degree universe ONCE
│   │   ├── market_api.py    # Fetches Target OHLCV and merges with global macro cache
│   │   └── funds_api.py     # Fundamental data API integration (Step 2)
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── qualifier.py     # Stock qualification logic for German retail investors
│   │   └── features.py      # Technical indicators, target engineering
│   │
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── base_pipeline.py # Imputer, Z-Scaler, ANOVA, SFS, LogReg Pipeline
│   │   ├── diagnostics.py   # KS Statistic, Cutoff, Confusion Matrix, Lift Charts
│   │   ├── step1_macro.py   # Execution logic for Step 1
│   │   └── step2_funds.py   # Execution logic for Step 2
│   │
│   └── orchestration/
│       ├── __init__.py
│       ├── json_exporter.py # Aggregates weights, params, and metrics into JSON
│       └── batch_runner.py  # Loops over tickers, manages the cascade & API caching
│
├── main.py                  # CLI Entry point
├── requirements.txt
└── README.md
```

-----

## 3. Module Specifications for Code Generation

### 3.1 `config/settings.yaml`

**Purpose:** Store all default hyperparameters.
**Agent Instructions:** Create this YAML file with:
*   `horizon_days`: 126
*   `threshold`: 0.15
*   `features_to_select`: 12
*   `step1_history_years`: 10
*   `step2_history_years`: 4
*   `quantiles`: 10
*   `xetra_t7_url`: "https://www.xetra.com/resource/blob/136704/8cc7402dc022e3369d7210e53a3eb2e5/data/t7-xfra-allTradableInstruments.csv"

### 3.2 `config/universe.py` (NEW COMPONENT)

**Purpose:** Define the expanded, 360-degree macroeconomic variable space.
**Agent Instructions:** Create this file. Use `python-dotenv` to load the `GEMINI_API_KEY`. Transcribe the exact lists provided in the system prompt (`MACRO_INDICATORS`, `COMMODITIES`, `AGRI_COMMODITIES`, `CREDIT_RISK`, `SECTORS_AND_INDICES`, `MORE_SECTORS`, `REAL_ESTATE`, `CRYPTO`, `TICKERS_US/DE/UK/JP`, `FRED_INDICATORS...` etc). 

Crucially, define these two aggregate variables at the bottom of the file for the ingestion engine to use:
1.  `ALL_YF_TICKERS`: A combined list of all Yahoo Finance symbols.
2.  `ALL_FRED_INDICATORS`: A combined list of all FRED series IDs.

### 3.3 `src/ingestion/global_macro.py` & `src/ingestion/market_api.py`

**Purpose:** Data extraction optimized for massive variable universes without hitting API bans.
**Agent Instructions:**
*   **Macro Caching Optimization (CRITICAL):** The agent MUST implement `fetch_global_macro_universe(history_years)` in `global_macro.py`. This function:
    1. Downloads `ALL_YF_TICKERS` (Adjusted Close only) via `yfinance`. 
    2. **Stationarity Fix:** Converts these absolute Yahoo Finance index prices into 20-day rolling returns (`.pct_change(20)`). Raw index prices are non-stationary and will destroy the Logistic Regression weights.
    3. Downloads `ALL_FRED_INDICATORS` via `pandas_datareader` (These rates/indices remain as-is).
    4. Merges them into a single `global_macro_df`.
    5. Applies strictly `.resample('B').ffill()` to map everything to business days, and returns it.
*   **`market_api.fetch_step1_data(ticker, global_macro_df)`**: Fetches *only* the specific target stock's OHLCV data. It then applies technical feature engineering to the target stock, and finally left-joins the target stock's dataframe with the injected `global_macro_df` on the date index. 

### 3.4 `src/modeling/base_pipeline.py`

**Purpose:** Standardize the reusable ML architecture with rigorous imputation and scaling.
**Agent Instructions:** Construct the following strict Scikit-learn Pipeline:
1.  `SimpleImputer(strategy='median')`: Fills any edge-case NaNs (e.g., assets that IPO'd 8 years ago within the 10-year window) *before* standardizing.
2.  `StandardScaler()`: Applies strict **Z-scaling** to normalize all variables natively so coefficients and weights are perfectly comparable.
3.  `SelectKBest(score_func=f_classif, k=40)`: ANOVA pre-filter (Set to 40 because the global universe is massive; this prevents SFS from taking hours).
4.  `SequentialFeatureSelector(LogisticRegression(class_weight='balanced', solver='liblinear'), cv=TimeSeriesSplit(n_splits=3), n_features_to_select=12)`
5.  `LogisticRegression(class_weight='balanced', solver='liblinear')` configured to output `predict_proba`.

### 3.5 `src/modeling/diagnostics.py`

**Purpose:** Evaluate probabilities, dynamic cutoffs, metrics, and visual artifacts.
**Agent Instructions:**
*   **`calculate_ks_and_cutoff(y_true, y_prob_1)`**: Maximize the KS statistic using ROC curve logic: `max(TPR - FPR)`. Return `(ks_stat, optimal_cutoff)`.
*   **`calculate_cv_accuracy(y_true, y_prob_1, optimal_cutoff)`**: Convert probabilities to binary `1` and `0` using the KS cutoff, then return `accuracy_score`.
*   **Visuals**: Implement `generate_confusion_matrix` and `generate_lift_chart` (splitting into 10 quantiles using `pd.qcut(..., duplicates='drop')`).

### 3.6 `src/modeling/step1_macro.py` & `src/modeling/step2_funds.py`

**Purpose:** Execution modules that extract model weights and trigger cascades.
**Agent Instructions:**
*   Fit models and generate CV probabilities via `cross_val_predict(..., method='predict_proba')`. Calculate KS Cutoffs and CV Accuracy.
*   **Feature Extraction Logic (CRITICAL):** Chain the boolean masks of `sfs.get_support()` and `anova.get_support()` back to the original DataFrame columns. Extract LogReg weights from `clf.coef_[0]`. Return a dictionary of `{"feature_name": float(weight)}`.
*   Cascade filter Step 1 dates where `y_prob >= ks_optimal_cutoff` to Step 2.

### 3.7 `src/orchestration/json_exporter.py`

**Purpose:** Format and write all model outputs to disk in a machine-readable format.
**Agent Instructions:**
*   Save to `outputs/predictions/{ticker}_prediction.json`.
*   Enforce this strict JSON Schema:
    ```json
    {
      "stock_name": "SAP.DE",
      "prediction_date": "2026-06-17",
      "applied_parameters": {
        "horizon_days": 126,
        "threshold": 0.15,
        "step1_history_years": 10,
        "step2_history_years": 4,
        "features_to_select": 12
      },
      "step1_model": {
        "cv_accuracy": 0.68,
        "ks_cutoff": 0.54,
        "predicted_class": "UP",
        "selected_predictors_and_weights": {
          "Ret_50D": 0.42,
          "T10Y2Y": -0.15,
          "EURUSD=X": 0.22
        }
      },
      "step2_model": {
        "cv_accuracy": 0.71,
        "ks_cutoff": 0.61,
        "predicted_class": "UP",
        "selected_predictors_and_weights": {
          "FreeCashFlow": 0.88,
          "TotalDebt": -0.34
        }
      },
      "final_prediction": "UP"
    }
    ```

### 3.8 `src/orchestration/batch_runner.py`

**Purpose:** Coordinates caching, cascade execution, and JSON payload creation.
**Agent Instructions:**
1. Call `global_macro.fetch_global_macro_universe()` **exactly once** before the stock loop begins and hold it in memory.
2. For each qualified `.DE` ticker (using `tqdm` for progress tracking), run Step 1, passing the cached `global_macro_df`.
3. If Step 1 evaluates the most recent unobservable date as `UP`, run Step 2.
4. Pass aggregated data to `json_exporter.py`. If final prediction is `UP`, update `data/processed/final_buy_signals.csv`.

-----

## 4. Required Dependencies (`requirements.txt`)

```text
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
yfinance>=0.2.31
pandas-datareader>=0.10.0
requests>=2.31.0
pyyaml>=6.0
tqdm>=4.65.0
matplotlib>=3.7.0      
seaborn>=0.12.0        
scipy>=1.10.0          
python-dotenv>=1.0.0   # Required for loading the GEMINI_API_KEY from .env
```

## 5. Strict Agent Constraints & Guardrails

*   **Global Macro Caching:** The agent MUST NOT put the global macro downloading inside the ticker loop. Fetching 80+ static indicators thousands of times will result in an immediate IP ban from Yahoo Finance.
*   **Stationarity Mandate:** The agent must explicitly convert Yahoo Finance macro assets to rolling percentage changes before passing them to the pipeline. 
*   **Z-Scaling Isolation:** The `StandardScaler()` MUST be placed *inside* the Scikit-learn Pipeline so it executes *after* train/test splits during CV to prevent data leakage.
*   **JSON Serialization Types:** Numpy floats and arrays are not naturally JSON serializable. The agent MUST cast weights and metrics to Python `float` before writing the JSON payload.
*   **KS Accuracy Evaluation:** Scikit-Learn `.score()` assumes a `0.5` cutoff. The agent MUST generate probabilities via `predict_proba()[:, 1]`, manually apply the optimal KS cutoff to convert them to 1s and 0s, and *then* calculate `accuracy_score`.