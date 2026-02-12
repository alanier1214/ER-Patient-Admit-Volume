from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score

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

# Assign data to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""# Build and train
model = GradientBoostingClassifier(
    n_estimators=100,      # number of trees
    learning_rate=0.1,     # shrinks contribution of each tree
    max_depth=3,           # depth of each tree
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")"""

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'objective': 'multi:softmax',
        'num_class': 6,
        'random_state': 42
    }

    model = xgb.XGBClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best params:", study.best_params)
print("Best score:", study.best_value)



"""model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.5,
    early_stopping_rounds=50,
    max_depth=4,
    objective='multi:softmax',   # for multi-class
    num_class=6,                  # set this to your actual number of classes
    eval_metric='mlogloss',       # multi-class log loss
    random_state=42
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")"""

best_params = study.best_params

final_model = xgb.XGBClassifier(
    **best_params,
    n_estimators=2000,          # set high, early stopping will cut it off
    early_stopping_rounds=50,
    objective='multi:softmax',
    num_class=6,
    eval_metric='mlogloss',
    random_state=42
)

final_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)

print("Optimal n_estimators:", final_model.best_iteration)

