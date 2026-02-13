import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas.tseries.holiday import USFederalHolidayCalendar


# ── Constants ───────────────────────────────────────────────────
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


# ── Load and Prepare Data ───────────────────────────────────────
def load_and_prepare(filepath):
    df = pd.read_parquet(filepath)
    df['Date']      = pd.to_datetime(df['Date'])
    df['Month']     = df['Date'].dt.month
    df['Year']      = df['Date'].dt.year
    df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday

    def hour_to_block(hour):
        if 0 <= hour < 6:     return 0  # Night
        elif 6 <= hour < 12:  return 1  # Morning
        elif 12 <= hour < 18: return 2  # Afternoon
        else:                 return 3  # Evening

    df['Hour_Block'] = df['Hour'].apply(hour_to_block)
    return df


def aggregate(df):
    agg = df.groupby(['Date', 'Year', 'Month', 'DayOfWeek', 'Hour_Block']).agg(
        Total_Enc=('ED Enc', 'sum'),
        Total_Admitted=('ED Enc Admitted', 'sum')
    ).reset_index()
    return agg


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

    # Sort chronologically so lags reflect real calendar time
    agg = agg.sort_values(['Date', 'Hour_Block']).reset_index(drop=True)

    # Lag features (4 blocks/day * 7 days = 28 rows back for 1 week)
    agg['Enc_Lag_1Week']  = agg['Total_Enc'].shift(7  * BLOCKS_PER_DAY)
    agg['Enc_Lag_1Month'] = agg['Total_Enc'].shift(30 * BLOCKS_PER_DAY)
    agg['Adm_Lag_1Week']  = agg['Total_Admitted'].shift(7  * BLOCKS_PER_DAY)
    agg['Adm_Lag_1Month'] = agg['Total_Admitted'].shift(30 * BLOCKS_PER_DAY)

    # Rolling averages
    agg['Enc_Rolling_7Day']  = agg['Total_Enc'].rolling(window=7  * BLOCKS_PER_DAY).mean()
    agg['Enc_Rolling_30Day'] = agg['Total_Enc'].rolling(window=30 * BLOCKS_PER_DAY).mean()
    agg['Adm_Rolling_7Day']  = agg['Total_Admitted'].rolling(window=7  * BLOCKS_PER_DAY).mean()
    agg['Adm_Rolling_30Day'] = agg['Total_Admitted'].rolling(window=30 * BLOCKS_PER_DAY).mean()

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=agg['Date'].min(), end=agg['Date'].max())
    agg['Is_Holiday'] = agg['Date'].isin(holidays).astype(int)

    agg = agg.dropna().reset_index(drop=True)
    return agg


# ── Optuna Hyperparameter Tuning ────────────────────────────────
def tune_model(X_train, y_train, X_test, y_test, n_trials=100):
    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 2000),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth':        trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma':            trial.suggest_float('gamma', 0, 5),
        }
        model = xgb.XGBRegressor(
            **params,
            objective='reg:squarederror',
            early_stopping_rounds=50,
            random_state=42
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        return mean_absolute_error(y_test, model.predict(X_test))

    study = optuna.create_study(direction='minimize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("Loading and preparing data...")
    df  = load_and_prepare("DSU-Dataset.parquet")
    agg = aggregate(df)
    agg = engineer_features(agg)

    # Train on everything before 2025, test on 2025
    train = agg.copy()
    test  = agg[agg['Date'] >= '2025-01-01']

    X_train_enc, y_train_enc = train[ENC_FEATURES], train['Total_Enc']
    X_test_enc,  y_test_enc  = test[ENC_FEATURES],  test['Total_Enc']
    X_train_adm, y_train_adm = train[ADM_FEATURES], train['Total_Admitted']
    X_test_adm,  y_test_adm  = test[ADM_FEATURES],  test['Total_Admitted']

    # ── Stage 1: Encounter Model ─────────────────────────────────
    print("\nTuning encounter model...")
    best_enc_params, enc_study = tune_model(X_train_enc, y_train_enc, X_test_enc, y_test_enc)

    optuna.visualization.plot_optimization_history(enc_study).show()


    model_enc = xgb.XGBRegressor(
        **best_enc_params,
        objective='reg:squarederror',
        early_stopping_rounds=50,
        random_state=42
    )
    model_enc.fit(X_train_enc, y_train_enc, eval_set=[(X_test_enc, y_test_enc)], verbose=False)

    enc_preds = model_enc.predict(X_test_enc)
    print("\n── Encounter Model Performance ──")
    print(f"MAE:  {mean_absolute_error(y_test_enc, enc_preds):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_enc, enc_preds)):.4f}")
    print(f"Encounter stats:\n{y_test_enc.describe()}")

    joblib.dump(model_enc, "xgb_encounter_forecast_final.pkl")
    print("Saved: xgb_encounter_forecast_final.pkl")

    # ── Stage 2: Admissions Model ────────────────────────────────
    print("\nTuning admissions model...")
    best_adm_params, adm_study = tune_model(X_train_adm, y_train_adm, X_test_adm, y_test_adm)

    optuna.visualization.plot_optimization_history(adm_study).show()


    model_adm = xgb.XGBRegressor(
        **best_adm_params,
        objective='reg:squarederror',
        early_stopping_rounds=50,
        random_state=42
    )
    model_adm.fit(X_train_adm, y_train_adm, eval_set=[(X_test_adm, y_test_adm)], verbose=False)

    adm_preds = model_adm.predict(X_test_adm)
    print("\n── Admissions Model Performance ──")
    print(f"MAE:  {mean_absolute_error(y_test_adm, adm_preds):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_adm, adm_preds)):.4f}")
    print(f"\nAdmissions stats:\n{y_test_adm.describe()}")

    joblib.dump(model_adm, "xgb_admissions_forecast_final.pkl")
    print("Saved: xgb_admissions_forecast.pkl")

    # Save aggregated dataset for use in forecast script
    joblib.dump(agg, "agg_data_final.pkl")
    print("Saved: agg_data_final.pkl")

    print("\nTraining complete.")