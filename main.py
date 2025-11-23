import pandas as pd
import numpy as np
import math
import time
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# --- 1. Generate Synthetic Data ---
np.random.seed(42)
n_samples = 4412

data = {
    'HAEMATOCRIT': np.random.normal(38.2, 6.0, n_samples),
    'HAEMOGLOBINS': np.random.normal(12.7, 2.1, n_samples),
    'ERYTHROCYTE': np.random.normal(4.54, 0.78, n_samples),
    'LEUCOCYTE': np.random.normal(8.7, 5.0, n_samples),
    'THROMBOCYTE': np.random.normal(257, 114, n_samples),
    'MCH': np.random.normal(28.2, 2.7, n_samples),
    'MCHC': np.random.normal(33.3, 1.2, n_samples),
    'MCV': np.random.normal(84.6, 6.9, n_samples),
    'AGE': np.random.normal(46.6, 21.7, n_samples).astype(int),
    'SEX': np.random.choice(['M', 'F'], n_samples, p=[2290/4412, 2122/4412]),
    'SOURCE': np.random.choice(['out', 'in'], n_samples, p=[2628/4412, 1784/4412])
}
df = pd.DataFrame(data)
# Ensure non-negative values where appropriate
for col in ['HAEMATOCRIT', 'HAEMOGLOBINS', 'ERYTHROCYTE', 'LEUCOCYTE', 'THROMBOCYTE', 'MCV', 'AGE']:
    df[col] = df[col].clip(lower=0)

# --- 2. Preprocessing ---
target_col = "SOURCE"
def encode_target(series):
    mapping = {'in': 0, 'out': 1}
    return series.map(mapping).astype(int), mapping

y, target_mapping = encode_target(df[target_col])
X = df.drop(columns=[target_col])

# One-hot encoding for SEX
X = pd.get_dummies(X, columns=['SEX'], drop_first=True)

# Split Data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Data shapes:", X_train.shape, X_valid.shape, X_test.shape)

# --- 3. Custom XGBoost Implementation (from provided code) ---
class TreeBooster():
    def __init__(self, X, g, h, params, max_depth, idxs=None):
        self.params = params
        self.max_depth = max_depth
        self.min_child_weight = params['min_child_weight'] if params['min_child_weight'] else 1.0
        self.reg_lambda = params['reg_lambda'] if params['reg_lambda'] else 1.0
        self.gamma = params['gamma'] if params['gamma'] else 1.0
        self.colsample_bynode = params['colsample_bynode'] if params['colsample_bynode'] else 1.0
        if isinstance(g, pd.Series): g = g.values
        if isinstance(h, pd.Series): h = h.values
        if idxs is None: idxs = np.arange(len(g))
        self.X, self.g, self.h, self.idxs = X, g, h, idxs
        self.n, self.c = len(idxs), X.shape[1]
        self.value = -g[idxs].sum() /  (h[idxs].sum() + self.reg_lambda)
        self.best_score_so_far = 0
        if self.max_depth > 0:
            self._maybe_insert_child_nodes()
    def _maybe_insert_child_nodes(self):
        for i in range(self.c): self._find_better_split(i)
        if self.is_leaf: return
        x = self.X.values[self.idxs,self.split_feature_idx]
        left_idx = np.nonzero(x <= self.threshold)[0]
        right_idx = np.nonzero(x > self.threshold)[0]
        self.left = TreeBooster(self.X, self.g, self.h, self.params, self.max_depth - 1, self.idxs[left_idx])
        self.right = TreeBooster(self.X, self.g, self.h, self.params, self.max_depth - 1, self.idxs[right_idx])
    @property
    def is_leaf(self): return self.best_score_so_far == 0
    def _find_better_split(self, feature_index):
        x = self.X.values[self.idxs,feature_index]
        g,h = self.g[self.idxs],self.h[self.idxs]
        sort_idx = np.argsort(x)
        sort_g, sort_h, sort_x = g[sort_idx], h[sort_idx], x[sort_idx]
        sum_g, sum_h = g.sum(), h.sum()
        sum_g_right, sum_h_right = sum_g, sum_h
        sum_g_left, sum_h_left = 0.,0.
        for i in range(0, self.n-1):
            g_i, h_i, x_i, x_i_next = sort_g[i], sort_h[i], sort_x[i], sort_x[i+1]
            sum_g_left += g_i; sum_g_right -= g_i
            sum_h_left += h_i; sum_h_right -= h_i
            if sum_h_left < self.min_child_weight or x_i == x_i_next: continue
            if sum_h_right < self.min_child_weight: break
            gain = 0.5 * ((sum_g_left**2 / (sum_h_left + self.reg_lambda))
                            + (sum_g_right**2 / (sum_h_right + self.reg_lambda))
                            - (sum_g**2 / (sum_h + self.reg_lambda))
                            ) - self.gamma/2
            if gain > self.best_score_so_far:
                self.split_feature_idx = feature_index
                self.best_score_so_far = gain
                self.threshold = (x_i + x_i_next) / 2
    def predict(self, X):
        return np.array([self._predict_row(row) for i, row in X.iterrows()])
    def _predict_row(self, row):
        if self.is_leaf: return self.value
        child = (self.left if row.iloc[self.split_feature_idx] <= self.threshold else self.right)
        return child._predict_row(row)

class XGBoostModel():
    def __init__(self, params, random_seed = None):
        self.params = defaultdict(lambda: None, params)
        self.subsample = self.params['subsample'] if self.params['subsample'] else 1.0
        self.learning_rate = self.params['learning_rate'] if self.params['learning_rate'] else 0.3
        self.base_prediction = self.params['base_score'] if self.params['base_score'] else 0.5
        self.max_depth = self.params['max_depth'] if self.params['max_depth'] else 5
        self.rng = np.random.default_rng(seed=random_seed)
    def fit(self, X, y, objective, num_boost_round, verbose=False, x_valid = None, y_valid = None, early_stopping_rounds=None):
        current_predictions = self.base_prediction * np.ones(shape = y.shape)
        self.boosters = []
        for i in range(num_boost_round):
            gradients = objective.gradient(y, current_predictions)
            hessian = objective.hessian(y, current_predictions)
            sample_idxs = None if self.subsample == 1.0 else self.rng.choice(len(y), size=math.floor(self.subsample * len(y)), replace=False)
            booster = TreeBooster(X, gradients, hessian, self.params, self.max_depth, sample_idxs)
            current_predictions += self.learning_rate * booster.predict(X)
            self.boosters.append(booster)
    def predict(self,X):
        return (self.base_prediction + self.learning_rate * np.sum([booster.predict(X) for booster in self.boosters],axis=0))

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

class LogisticObjective:
    @staticmethod
    def loss(y,y_raw):
        p = sigmoid(y_raw)
        eps = 1e-12
        return -np.mean(y*np.log(p+eps) + (1-y)*np.log(1-p+eps))
    @staticmethod
    def gradient(y, y_raw):
        p = sigmoid(y_raw)
        return (p - y)
    @staticmethod
    def hessian(y, y_raw):
        p = sigmoid(y_raw)
        return p * (1.0 - p)

# Train Custom Model
pos_rate = float(y_train.mean())
base_score = math.log(pos_rate / (1 - pos_rate))
params = {
    "learning_rate": 0.1,
    "max_depth": 3,
    "subsample": 0.8,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "base_score": base_score,
    "colsample_bynode": 1.0,
}
objective = LogisticObjective()
custom_model = XGBoostModel(params=params, random_seed=42)

start_time = time.time()
custom_model.fit(X_train, y_train, objective, num_boost_round=10) # Reduced rounds for speed in demo
custom_time = time.time() - start_time

# Predict Custom
y_pred_raw = custom_model.predict(X_test)
y_pred_custom = (sigmoid(y_pred_raw) >= 0.5).astype(int)
acc_custom = accuracy_score(y_test, y_pred_custom)

# --- 4. Standard XGBoost ---
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

xgb_params = {
    'eta': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'min_child_weight': 1.0,
    'lambda': 1.0,
    'gamma': 0.0,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'seed': 42
}

start_time = time.time()
bst = xgb.train(xgb_params, dtrain, num_boost_round=10)
std_time = time.time() - start_time

# Predict Standard
y_pred_prob_std = bst.predict(dtest)
y_pred_std = (y_pred_prob_std >= 0.5).astype(int)
acc_std = accuracy_score(y_test, y_pred_std)

print(f"Custom Model Accuracy: {acc_custom:.4f}, Time: {custom_time:.4f}s")
print(f"Standard Library Accuracy: {acc_std:.4f}, Time: {std_time:.4f}s")