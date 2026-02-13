"""
ED Volume Forecasting — Forecast & Validation Output
=====================================================
Loads trained models and aggregated data produced by train_model.py.

Outputs:
    Forecast CSVs (predicted vs actual 2025 data):
        forecast_7day.csv
        forecast_30day.csv
        forecast_60day.csv

    Monthly summary CSVs (with % error):
        summary_7day.csv
        summary_30day.csv
        summary_60day.csv

Run train_model.py first to generate the required .pkl files.
"""

import pandas as pd
import numpy as np
import joblib

# ── Constants (must match train_model.py) ───────────────────────
BLOCK_NAMES = {
    0: 'Night (00-06)',
    1: 'Morning (06-12)',
    2: 'Afternoon (12-18)',
    3: 'Evening (18-24)'
}
DAY_NAMES = {
    0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
    3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
}
MONTH_NAMES = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}

ENC_FEATURES = [
    'Year', 'Month', 'DayOfWeek', 'Hour_Block',
    'Month_sin', 'Month_cos',
    'DayOfWeek_sin', 'DayOfWeek_cos',
    'Hour_Block_sin', 'Hour_Block_cos',
    'Enc_Lag_1Week', 'Enc_Lag_1Month',
    'Enc_Rolling_7Day', 'Enc_Rolling_30Day'
]

ADM_FEATURES = ENC_FEATURES + [
    'Total_Enc',
    'Adm_Lag_1Week', 'Adm_Lag_1Month',
    'Adm_Rolling_7Day', 'Adm_Rolling_30Day'
]


# ── Load Models and Data ─────────────────────────────────────────
print("Loading models and data...")
model_enc = joblib.load("xgb_encounter_forecast.pkl")
model_adm = joblib.load("xgb_admissions_forecast.pkl")
agg       = joblib.load("agg_data.pkl")

# Split into pre-2025 (training history) and 2025 (actual test data)
train_history = agg[agg['Year'] < 2025]
actual_2025   = agg[agg['Year'] >= 2025].copy()

# Latest row from training data — used for lag/rolling values
latest = train_history.iloc[-1]


# ── Forecast vs Actual Function ──────────────────────────────────
def generate_forecast_vs_actual(days_out):
    """
    Slices the first N unique days from actual 2025 data,
    runs predictions, and returns a comparison DataFrame.
    """
    # Get the first N unique (Month, DayOfWeek) day combinations
    unique_days = (
        actual_2025[['Year', 'Month', 'DayOfWeek']]
        .drop_duplicates()
        .head(days_out)
    )

    subset = actual_2025.merge(unique_days, on=['Year', 'Month', 'DayOfWeek']).copy()

    # Stage 1: Predict encounters
    subset['Predicted_Enc'] = (
        model_enc.predict(subset[ENC_FEATURES]).clip(0).round().astype(int)
    )

    # Stage 2: Predict admissions using predicted encounters
    adm_input = subset[ENC_FEATURES].copy()
    adm_input['Total_Enc']         = subset['Predicted_Enc']
    adm_input['Adm_Lag_1Week']     = latest['Adm_Lag_1Week']
    adm_input['Adm_Lag_1Month']    = latest['Adm_Lag_1Month']
    adm_input['Adm_Rolling_7Day']  = latest['Adm_Rolling_7Day']
    adm_input['Adm_Rolling_30Day'] = latest['Adm_Rolling_30Day']

    subset['Predicted_Adm'] = (
        model_adm.predict(adm_input[ADM_FEATURES]).clip(0).round().astype(int)
    )

    # Readable labels
    subset['Month_Name']      = subset['Month'].map(MONTH_NAMES)
    subset['Day_Name']        = subset['DayOfWeek'].map(DAY_NAMES)
    subset['Hour_Block_Name'] = subset['Hour_Block'].map(BLOCK_NAMES)

    # Rename actuals for clarity
    subset = subset.rename(columns={
        'Total_Enc':      'Actual_Enc',
        'Total_Admitted': 'Actual_Adm'
    })

    # Difference columns
    subset['Enc_Difference'] = subset['Actual_Enc'] - subset['Predicted_Enc']
    subset['Adm_Difference'] = subset['Actual_Adm'] - subset['Predicted_Adm']

    return subset[[
        'Year', 'Month_Name', 'Day_Name', 'Hour_Block_Name',
        'Actual_Enc', 'Predicted_Enc', 'Enc_Difference',
        'Actual_Adm', 'Predicted_Adm', 'Adm_Difference'
    ]]


# ── Summary Function ─────────────────────────────────────────────
def summarize(forecast_df, group_col, group_label, label):
    summary = forecast_df.groupby(group_col).agg(
        Actual_Enc=('Actual_Enc', 'sum'),
        Predicted_Enc=('Predicted_Enc', 'sum'),
        Actual_Adm=('Actual_Adm', 'sum'),
        Predicted_Adm=('Predicted_Adm', 'sum')
    ).reset_index()

    summary['Enc_Difference'] = summary['Actual_Enc'] - summary['Predicted_Enc']
    summary['Adm_Difference'] = summary['Actual_Adm'] - summary['Predicted_Adm']
    summary['Enc_Pct_Error']  = (summary['Enc_Difference'] / summary['Actual_Enc'] * 100).round(2)
    summary['Adm_Pct_Error']  = (summary['Adm_Difference'] / summary['Actual_Adm'] * 100).round(2)

    summary = summary.rename(columns={group_col: group_label})

    print(f"\n── {label} Summary by {group_label} ──")
    print(summary.to_string(index=False))
    return summary


# ── Generate All Three Horizons ──────────────────────────────────
print("\nGenerating forecasts...")

forecast_7  = generate_forecast_vs_actual(days_out=7)
forecast_30 = generate_forecast_vs_actual(days_out=30)
forecast_60 = generate_forecast_vs_actual(days_out=60)

print("\n── 7-Day Forecast vs Actual ──")
print(forecast_7.to_string(index=False))
print("\n── 30-Day Forecast vs Actual ──")
print(forecast_30.to_string(index=False))
print("\n── 60-Day Forecast vs Actual ──")
print(forecast_60.to_string(index=False))


# ── Generate Summaries for All Three Groupings x Three Horizons ─
horizons = {
    '7day':  forecast_7,
    '30day': forecast_30,
    '60day': forecast_60,
}

groupings = [
    ('Month_Name',      'Month'),
    ('Day_Name',        'Day_of_Week'),
    ('Hour_Block_Name', 'Hour_Block'),
]

all_summaries = {}

for horizon_label, forecast_df in horizons.items():
    for group_col, group_label in groupings:
        key = f"{group_label}_{horizon_label}"
        summary = summarize(forecast_df, group_col, group_label, f"{horizon_label} — {group_label}")
        all_summaries[key] = summary


# ── Save All CSVs ────────────────────────────────────────────────

# Detailed row-level forecasts
forecast_7.to_csv("forecast_7day.csv",   index=False)
forecast_30.to_csv("forecast_30day.csv", index=False)
forecast_60.to_csv("forecast_60day.csv", index=False)

# Summary CSVs for each grouping x horizon
for key, summary_df in all_summaries.items():
    filename = f"summary_{key}.csv"
    summary_df.to_csv(filename, index=False)
    print(f"Saved: {filename}")

"""
This will produce 9 summary CSVs — one for each combination of grouping and horizon:

summary_Month_7day.csv          summary_Month_30day.csv         summary_Month_60day.csv
summary_Day_of_Week_7day.csv    summary_Day_of_Week_30day.csv   summary_Day_of_Week_60day.csv
summary_Hour_Block_7day.csv     summary_Hour_Block_30day.csv    summary_Hour_Block_60day.c
"""
