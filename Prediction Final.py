"""
ED Volume Forecasting — Prediction Script
==========================================
Instructions:
    1. Place this file in the same directory as:
            xgb_encounter_forecast.pkl
            xgb_admissions_forecast.pkl
    2. Update INPUT_FILE to point to your dataset
    3. Run: python predict.py

Input:
    Parquet file with columns: Site, Date, Hour, REASON_VISIT_NAME, ED Enc, ED Enc Admitted

Output:
    predictions_detail.csv      — Predictions for every date/day/hour block combination
    predictions_by_month.csv    — Predictions summarized by month
    predictions_by_dow.csv      — Predictions summarized by day of week
    predictions_by_hourblock.csv— Predictions summarized by hour block
"""

import pandas as pd
import numpy as np
import joblib
import calendar
from pandas.tseries.holiday import USFederalHolidayCalendar

# ── Configuration ────────────────────────────────────────────────
INPUT_FILE = "DSU-Dataset.csv"   # <-- only line you need to change

# ── Constants ────────────────────────────────────────────────────
BLOCKS_PER_DAY = 4

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
    'Year', 'Month', 'Week', 'DayOfWeek', 'Hour_Block',
    'Month_sin', 'Month_cos',
    'Week_sin', 'Week_cos',
    'DayOfWeek_sin', 'DayOfWeek_cos',
    'Hour_Block_sin', 'Hour_Block_cos',
    'Enc_Lag_1Week', 'Enc_Lag_1Month',
    'Enc_Rolling_7Day', 'Enc_Rolling_30Day', 'Is_Holiday'
]

ADM_FEATURES = ENC_FEATURES + [
    'Total_Enc',
    'Adm_Lag_1Week', 'Adm_Lag_1Month',
    'Adm_Rolling_7Day', 'Adm_Rolling_30Day'
]


# ── Load Data ────────────────────────────────────────────────────
def load_and_prepare(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df['Date']      = pd.to_datetime(df['Date'])
    df['Month']     = df['Date'].dt.month
    df['Year']      = df['Date'].dt.year
    df['DayOfWeek'] = df['Date'].dt.dayofweek

    def hour_to_block(hour):
        if 0 <= hour < 6:     return 0  # Night
        elif 6 <= hour < 12:  return 1  # Morning
        elif 12 <= hour < 18: return 2  # Afternoon
        else:                 return 3  # Evening

    df['Hour_Block'] = df['Hour'].apply(hour_to_block)
    return df


# ── Aggregate ────────────────────────────────────────────────────
def aggregate(df):
    agg = df.groupby(['Date', 'Year', 'Month', 'DayOfWeek', 'Hour_Block']).agg(
        Total_Enc=('ED Enc', 'sum'),
        Total_Admitted=('ED Enc Admitted', 'sum')
    ).reset_index()
    return agg


# ── Feature Engineering ──────────────────────────────────────────
def engineer_features(agg):
    # Cyclical encoding
    agg['Month_sin']      = np.sin(2 * np.pi * agg['Month'] / 12)
    agg['Month_cos']      = np.cos(2 * np.pi * agg['Month'] / 12)
    agg['DayOfWeek_sin']  = np.sin(2 * np.pi * agg['DayOfWeek'] / 7)
    agg['DayOfWeek_cos']  = np.cos(2 * np.pi * agg['DayOfWeek'] / 7)
    agg['Hour_Block_sin'] = np.sin(2 * np.pi * agg['Hour_Block'] / 4)
    agg['Hour_Block_cos'] = np.cos(2 * np.pi * agg['Hour_Block'] / 4)

    # Week of year
    agg['Week']     = pd.to_datetime(agg['Date']).dt.isocalendar().week.astype(int)
    agg['Week_sin'] = np.sin(2 * np.pi * agg['Week'] / 52)
    agg['Week_cos'] = np.cos(2 * np.pi * agg['Week'] / 52)

    # Holiday flag
    cal      = USFederalHolidayCalendar()
    holidays = cal.holidays(start=agg['Date'].min(), end=agg['Date'].max())
    agg['Is_Holiday'] = agg['Date'].isin(holidays).astype(int)

    # Sort chronologically for accurate lags
    agg = agg.sort_values(['Date', 'Hour_Block']).reset_index(drop=True)

    # Lag features
    agg['Enc_Lag_1Week']  = agg['Total_Enc'].shift(7  * BLOCKS_PER_DAY)
    agg['Enc_Lag_1Month'] = agg['Total_Enc'].shift(30 * BLOCKS_PER_DAY)
    agg['Adm_Lag_1Week']  = agg['Total_Admitted'].shift(7  * BLOCKS_PER_DAY)
    agg['Adm_Lag_1Month'] = agg['Total_Admitted'].shift(30 * BLOCKS_PER_DAY)

    # Rolling averages
    agg['Enc_Rolling_7Day']  = agg['Total_Enc'].rolling(window=7  * BLOCKS_PER_DAY).mean()
    agg['Enc_Rolling_30Day'] = agg['Total_Enc'].rolling(window=30 * BLOCKS_PER_DAY).mean()
    agg['Adm_Rolling_7Day']  = agg['Total_Admitted'].rolling(window=7  * BLOCKS_PER_DAY).mean()
    agg['Adm_Rolling_30Day'] = agg['Total_Admitted'].rolling(window=30 * BLOCKS_PER_DAY).mean()

    agg = agg.dropna().reset_index(drop=True)
    return agg


# ── Run Predictions ──────────────────────────────────────────────
def run_predictions(agg, model_enc, model_adm):
    predictions = agg.copy()

    # Stage 1: Predict encounters
    predictions['Predicted_Enc'] = (
        model_enc.predict(predictions[ENC_FEATURES]).clip(0).round().astype(int)
    )

    # Stage 2: Predict admissions using predicted encounters
    adm_input = predictions[ENC_FEATURES].copy()
    adm_input['Total_Enc']         = predictions['Predicted_Enc']
    adm_input['Adm_Lag_1Week']     = predictions['Adm_Lag_1Week']
    adm_input['Adm_Lag_1Month']    = predictions['Adm_Lag_1Month']
    adm_input['Adm_Rolling_7Day']  = predictions['Adm_Rolling_7Day']
    adm_input['Adm_Rolling_30Day'] = predictions['Adm_Rolling_30Day']

    predictions['Predicted_Adm'] = (
        model_adm.predict(adm_input[ADM_FEATURES]).clip(0).round().astype(int)
    )

    # Readable labels
    predictions['Month_Name']      = predictions['Month'].map(MONTH_NAMES)
    predictions['Day_Name']        = predictions['DayOfWeek'].map(DAY_NAMES)
    predictions['Hour_Block_Name'] = predictions['Hour_Block'].map(BLOCK_NAMES)

    return predictions


# ── Summarize ────────────────────────────────────────────────────
def summarize(df, group_col, group_label):
    summary = df.groupby(group_col).agg(
        Predicted_Enc=('Predicted_Enc', 'sum'),
        Predicted_Adm=('Predicted_Adm', 'sum')
    ).reset_index().rename(columns={group_col: group_label})
    return summary


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Load and prepare
    df  = load_and_prepare(INPUT_FILE)
    agg = aggregate(df)
    agg = engineer_features(agg)

    # Load models
    print("Loading models...")
    model_enc = joblib.load("xgb_encounter_forecast.pkl")
    model_adm = joblib.load("xgb_admissions_forecast.pkl")

    # Run predictions
    print("Generating predictions...")
    predictions = run_predictions(agg, model_enc, model_adm)

    # ── Slice into forecast horizons ─────────────────────────────
    min_date = predictions['Date'].min()

    horizon_7  = predictions[predictions['Date'] <= min_date + pd.Timedelta(days=7)]
    horizon_30 = predictions[predictions['Date'] <= min_date + pd.Timedelta(days=30)]
    horizon_60 = predictions[predictions['Date'] <= min_date + pd.Timedelta(days=60)]

    # ── Save summaries for each horizon and grouping ─────────────
    groupings = [
        ('Month_Name',      'Month'),
        ('Day_Name',        'Day_of_Week'),
        ('Hour_Block_Name', 'Hour_Block'),
    ]

    for horizon_label, horizon_df in [('7day', horizon_7), ('30day', horizon_30), ('60day', horizon_60)]:
        print(f"\n══ {horizon_label} Forecast ══")
        for group_col, group_label in groupings:
            summary = summarize(horizon_df, group_col, group_label)
            print(f"\n── By {group_label} ──")
            print(summary.to_string(index=False))
            summary.to_csv(f"predictions_{group_label}_{horizon_label}.csv", index=False)

    # ── Detailed row-level output for each horizon ───────────────
    for horizon_label, horizon_df in [('7day', horizon_7), ('30day', horizon_30), ('60day', horizon_60)]:
        detail = horizon_df[[
            'Date', 'Month_Name', 'Day_Name', 'Hour_Block_Name',
            'Predicted_Enc', 'Predicted_Adm'
        ]]
        detail.to_csv(f"predictions_detail_{horizon_label}.csv", index=False)

    print("\nSaved:")
    print("  predictions_detail_7day.csv / 30day / 60day")
    print("  predictions_Month_7day.csv  / 30day / 60day")
    print("  predictions_Day_of_Week_7day.csv / 30day / 60day")
    print("  predictions_Hour_Block_7day.csv  / 30day / 60day")
    print("\nDone.")