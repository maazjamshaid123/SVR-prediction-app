import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
import joblib

data = pd.read_csv('data/Data_for_ML.csv')
X = data[['A', 'B', 'C', 'D']]
y = data['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_kf_splits = 5
n_ss_splits = 20

shuffle_split = ShuffleSplit(n_splits=n_ss_splits, test_size=0.2, random_state=42)

model = SVR()

param_grid = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.1, 0.2, 0.5, 0.8, 1, 2, 3, 5, 8, 10],
    'epsilon': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
}

best_models = []

for ss_idx, (train_index, test_index) in enumerate(shuffle_split.split(X_scaled)):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    grid_search = GridSearchCV(model, param_grid, cv=n_kf_splits, refit=True)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_models.append(best_model)

for i, best_model in enumerate(best_models):
    filename = f"best_model_svr_{i}.joblib"  # Adjust the filename format as needed
    joblib.dump(best_model, filename)
    print(f"Saved best model {i} to {filename}")
