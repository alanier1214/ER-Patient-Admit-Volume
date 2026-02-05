import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
    
########## Get all ED Encounter information ##########    
df = pd.read_parquet("DSU-Dataset-Expanded.parquet")

########## Start running regressions on encounter data ##########
target = "ED Enc Admitted"

y = pd.to_numeric(df[target], errors="coerce")
X = df.drop(columns=[target])

# Convert all remaining columns to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Drop columns that are fully NaN
X = X.dropna(axis=1, how="all")

# Drop rows with missing target
valid = y.notna()
X = X[valid]
y = y[valid]

# Use imputer to handle NaN values
imputer = SimpleImputer(strategy="mean")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

# Create train and test splits
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

# Run regression
model = RandomForestRegressor(
    n_estimators=1000,
    random_state=42,
    max_depth=None
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Print MAE and R²
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Look at feature importance and permutatin feature importance
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values("importance", ascending=False)
print("Top Features and Importance")
print(importance)

perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
perm_df = pd.DataFrame({
    "feature": X.columns,
    "importance": perm_importance.importances_mean
}).sort_values("importance", ascending=False)
print("Top Features and Importance (Permutation)")
print(perm_df)