import pandas as pd
import optuna
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
trial_number=100

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import optuna
import joblib

def linear_regression(X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val):
    trial_number = 100  # Optunaの試行回数を設定
    
    # ハイパーパラメータ探索の定義（length）
    def objective_length(trial):
        params = {
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'copy_X': trial.suggest_categorical('copy_X', [True, False]),
            'n_jobs': trial.suggest_categorical('n_jobs', [-1, 1, 2]),
            'positive': trial.suggest_categorical('positive', [True, False])
        }
        model = LinearRegression(**params)
        model.fit(X_train, y_length_train)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_length_val, preds, squared=False)
        return rmse
    
    # ハイパーパラメータ探索の定義（size）
    def objective_size(trial):
        params = {
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'copy_X': trial.suggest_categorical('copy_X', [True, False]),
            'n_jobs': trial.suggest_categorical('n_jobs', [-1, 1, 2]),
            'positive': trial.suggest_categorical('positive', [True, False])
        }
        model = LinearRegression(**params)
        model.fit(X_train, y_size_train)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_size_val, preds, squared=False)
        return rmse

    # Optunaによるハイパーパラメータ探索の実行（長さ）
    study_length = optuna.create_study(direction='minimize')
    study_length.optimize(objective_length, n_trials=trial_number)

    # Optunaによるハイパーパラメータ探索の実行（サイズ）
    study_size = optuna.create_study(direction='minimize')
    study_size.optimize(objective_size, n_trials=trial_number)

    # 最適なハイパーパラメータの取得（長さ）
    best_params_length = study_length.best_params

    # 最適なハイパーパラメータの取得（サイズ）
    best_params_size = study_size.best_params

    # 最適なモデルの学習（長さ）
    best_model_length = LinearRegression(**best_params_length)
    best_model_length.fit(X_train, y_length_train)

    # 最適なモデルの学習（サイズ）
    best_model_size = LinearRegression(**best_params_size)
    best_model_size.fit(X_train, y_size_train)

    print("\nTrainRSME:")
    print(study_length.best_value)
    print(study_size.best_value)

    # モデルの保存
    joblib.dump(best_model_length, './model/linear_model_length.pkl')
    joblib.dump(best_model_size, './model/linear_model_size.pkl')

    # バリデーションデータでの予測と評価
    y_length_val_pred = best_model_length.predict(X_val)
    y_size_val_pred = best_model_size.predict(X_val)

    length_mae = mean_absolute_error(y_length_val, y_length_val_pred)
    length_r2 = r2_score(y_length_val, y_length_val_pred)
    length_acc = accuracy_score((y_length_val - y_length_val_pred).abs() < 5, [True]*len(y_length_val))

    size_mae = mean_absolute_error(y_size_val, y_size_val_pred)
    size_r2 = r2_score(y_size_val, y_size_val_pred)
    size_acc = accuracy_score((y_size_val - y_size_val_pred).abs() < 1, [True]*len(y_size_val))

    return './model/linear_model_length.pkl', './model/linear_model_size.pkl', study_length.best_value, study_size.best_value, length_mae, size_mae, length_r2, size_r2, length_acc, size_acc


def svr_regression(X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val):
    trial_number = 100  # Optunaの試行回数を設定
    
    # OptunaのObjective関数の定義（length）
    def objective_length(trial):
        params = {
            'C': trial.suggest_float('C', 0.1, 10.0, log=True),
            'epsilon': trial.suggest_float('epsilon', 0.01, 0.5),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'max_iter': trial.suggest_categorical('max_iter', [-1, 1000, 10000])
        }
        model = SVR(**params)
        model.fit(X_train, y_length_train)
        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_length_val, y_pred, squared=False)
        return rmse

    # OptunaのObjective関数の定義（size）
    def objective_size(trial):
        params = {
            'C': trial.suggest_float('C', 0.1, 10.0, log=True),
            'epsilon': trial.suggest_float('epsilon', 0.01, 0.5),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'max_iter': trial.suggest_categorical('max_iter', [-1, 1000, 10000])
        }
        model = SVR(**params)
        model.fit(X_train, y_size_train)
        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_size_val, y_pred, squared=False)
        return rmse

    # Optunaのスタディの作成と最適化の実行
    study_length = optuna.create_study(direction='minimize')
    study_size = optuna.create_study(direction='minimize')
    study_length.optimize(objective_length, n_trials=trial_number)
    study_size.optimize(objective_size, n_trials=trial_number)

    # 最適なハイパーパラメータの表示
    print("Best hyperparameters for length: ", study_length.best_params)
    print("Best hyperparameters for size: ", study_size.best_params)

    # 最適なモデルの定義
    best_model_length = SVR(**study_length.best_params)
    best_model_size = SVR(**study_size.best_params)

    # 最適なモデルの訓練
    best_model_length.fit(X_train, y_length_train)
    best_model_size.fit(X_train, y_size_train)

    # バリデーションデータでの予測と評価
    y_length_val_pred = best_model_length.predict(X_val)
    y_size_val_pred = best_model_size.predict(X_val)

    length_mae = mean_absolute_error(y_length_val, y_length_val_pred)
    length_r2 = r2_score(y_length_val, y_length_val_pred)
    length_acc = accuracy_score((y_length_val - y_length_val_pred).abs() < 5, [True]*len(y_length_val))

    size_mae = mean_absolute_error(y_size_val, y_size_val_pred)
    size_r2 = r2_score(y_size_val, y_size_val_pred)
    size_acc = accuracy_score((y_size_val - y_size_val_pred).abs() < 1, [True]*len(y_size_val))

    print("\nTrainRSME:")
    print(study_length.best_value)
    print(study_size.best_value)

    # モデルの保存
    length_model_path = './model/svr_model_length.pkl'
    size_model_path = './model/svr_model_size.pkl'
    joblib.dump(best_model_length, length_model_path)
    joblib.dump(best_model_size, size_model_path)

    return length_model_path, size_model_path, study_length.best_value, study_size.best_value, length_mae, size_mae, length_r2, size_r2, length_acc, size_acc



def lgbm_regression(X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val):
    trial_number = 100  # Optunaの試行回数を設定
    
    # ハイパーパラメータ探索の定義（length）
    def objective_length(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 31, 100),
            'max_depth': trial.suggest_int('max_depth', 1, 30),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 1, 500),
            'reg_alpha': trial.suggest_categorical('reg_alpha', [0.0, 0.1, 0.5]),
            'reg_lambda': trial.suggest_categorical('reg_lambda', [0.0, 0.1, 0.5]),
            'random_state': 42
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_length_train)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_length_val, preds, squared=False)
        return rmse

    # ハイパーパラメータ探索の定義（size）
    def objective_size(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 31, 100),
            'max_depth': trial.suggest_int('max_depth', 1, 30),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 1, 500),
            'reg_alpha': trial.suggest_categorical('reg_alpha', [0.0, 0.1, 0.5]),
            'reg_lambda': trial.suggest_categorical('reg_lambda', [0.0, 0.1, 0.5]),
            'random_state': 42
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_size_train)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_size_val, preds, squared=False)
        return rmse

    # Optunaによるハイパーパラメータ探索の実行
    study_length = optuna.create_study(direction='minimize')
    study_size = optuna.create_study(direction='minimize')
    study_length.optimize(objective_length, n_trials=trial_number)
    study_size.optimize(objective_size, n_trials=trial_number)

    # 最適なハイパーパラメータの取得
    best_params_length = study_length.best_params
    best_params_size = study_size.best_params

    # 最適なモデルの学習
    best_model_length = lgb.LGBMRegressor(**best_params_length)
    best_model_length.fit(X_train, y_length_train)

    best_model_size = lgb.LGBMRegressor(**best_params_size)
    best_model_size.fit(X_train, y_size_train)

    # バリデーションデータでの予測と評価
    y_length_val_pred = best_model_length.predict(X_val)
    y_size_val_pred = best_model_size.predict(X_val)

    length_mae = mean_absolute_error(y_length_val, y_length_val_pred)
    length_r2 = r2_score(y_length_val, y_length_val_pred)
    length_acc = accuracy_score((y_length_val - y_length_val_pred).abs() < 5, [True]*len(y_length_val))

    size_mae = mean_absolute_error(y_size_val, y_size_val_pred)
    size_r2 = r2_score(y_size_val, y_size_val_pred)
    size_acc = accuracy_score((y_size_val - y_size_val_pred).abs() < 1, [True]*len(y_size_val))

    print("\nTrainRSME:")
    print(study_length.best_value)
    print(study_size.best_value)

    # モデルの保存
    length_model_path = './model/lgbm_model_length.pkl'
    size_model_path = './model/lgbm_model_size.pkl'
    joblib.dump(best_model_length, length_model_path)
    joblib.dump(best_model_size, size_model_path)
    
    return length_model_path, size_model_path, study_length.best_value, study_size.best_value, length_mae, size_mae, length_r2, size_r2, length_acc, size_acc


def rf_regression(X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val):
    trial_number = 100  # Optunaの試行回数を設定
    
    # Optunaによるハイパーパラメータ探索の定義（length）
    def objective_length(trial):
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 1, 500),
            'criterion': trial.suggest_categorical('criterion', ['squared_error', 'absolute_error']),
            'max_depth': trial.suggest_int('max_depth', 1, 30),
            'n_jobs': -1,
            'random_state': 42
        }
        if bootstrap:
            params['max_samples'] = trial.suggest_float('max_samples', 0.5, 1.0)

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_length_train)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_length_val, preds, squared=False)
        return rmse

    # Optunaによるハイパーパラメータ探索の定義（size）
    def objective_size(trial):
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 1, 500),
            'criterion': trial.suggest_categorical('criterion', ['squared_error', 'absolute_error']),
            'max_depth': trial.suggest_int('max_depth', 1, 30),
            'n_jobs': -1,
            'random_state': 42
        }
        if bootstrap:
            params['max_samples'] = trial.suggest_float('max_samples', 0.5, 1.0)

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_size_train)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_size_val, preds, squared=False)
        return rmse

    # スタディの設定
    study_length = optuna.create_study(direction='minimize')
    study_size = optuna.create_study(direction='minimize')

    # ハイパーパラメータ探索の実行
    study_length.optimize(objective_length, n_trials=trial_number, timeout=600)
    study_size.optimize(objective_size, n_trials=trial_number, timeout=600)

    # 最適なハイパーパラメータの取得
    best_params_length = study_length.best_params
    best_params_size = study_size.best_params

    # 最適なモデルの学習
    best_model_length = RandomForestRegressor(**best_params_length)
    best_model_size = RandomForestRegressor(**best_params_size)

    best_model_length.fit(X_train, y_length_train)
    best_model_size.fit(X_train, y_size_train)

    # バリデーションデータでの予測と評価
    y_length_val_pred = best_model_length.predict(X_val)
    y_size_val_pred = best_model_size.predict(X_val)

    length_mae = mean_absolute_error(y_length_val, y_length_val_pred)
    length_r2 = r2_score(y_length_val, y_length_val_pred)
    length_acc = accuracy_score((y_length_val - y_length_val_pred).abs() < 5, [True]*len(y_length_val))

    size_mae = mean_absolute_error(y_size_val, y_size_val_pred)
    size_r2 = r2_score(y_size_val, y_size_val_pred)
    size_acc = accuracy_score((y_size_val - y_size_val_pred).abs() < 1, [True]*len(y_size_val))

    print("\nTrainRMSE:")
    print(study_length.best_value)
    print(study_size.best_value)

    # モデルの保存
    length_model_path = './model/rf_model_length.pkl'
    size_model_path = './model/rf_model_size.pkl'
    joblib.dump(best_model_length, length_model_path)
    joblib.dump(best_model_size, size_model_path)

    return length_model_path, size_model_path, study_length.best_value, study_size.best_value, length_mae, size_mae, length_r2, size_r2, length_acc, size_acc



def gb_regression(X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val):
    trial_number = 100  # Optunaの試行回数を設定
    
    # Optunaによるハイパーパラメータ探索の定義（length）
    def objective_length(trial):
        params = {
            'loss': trial.suggest_categorical('loss', ['squared_error', 'absolute_error', 'huber', 'quantile']),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 1, 500),
            'max_depth': trial.suggest_int('max_depth', 1, 30),
            'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2']),
            'alpha': trial.suggest_float('alpha', 0.9, 0.99),
            'random_state': 42,
        }
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_length_train)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_length_val, preds, squared=False)
        return rmse

    # Optunaによるハイパーパラメータ探索の定義（size）
    def objective_size(trial):
        params = {
            'loss': trial.suggest_categorical('loss', ['squared_error', 'absolute_error', 'huber', 'quantile']),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 1, 500),
            'max_depth': trial.suggest_int('max_depth', 1, 30),
            'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2']),
            'alpha': trial.suggest_float('alpha', 0.9, 0.99),
            'random_state': 42,
        }
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_size_train)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_size_val, preds, squared=False)
        return rmse

    # スタディの設定
    study_length = optuna.create_study(direction='minimize')
    study_size = optuna.create_study(direction='minimize')

    # ハイパーパラメータ探索の実行
    study_length.optimize(objective_length, n_trials=trial_number, timeout=600)
    study_size.optimize(objective_size, n_trials=trial_number, timeout=600)

    # 最適なハイパーパラメータの取得
    best_params_length = study_length.best_params
    best_params_size = study_size.best_params

    # 最適なモデルの学習
    best_model_length = GradientBoostingRegressor(**best_params_length)
    best_model_size = GradientBoostingRegressor(**best_params_size)

    best_model_length.fit(X_train, y_length_train)
    best_model_size.fit(X_train, y_size_train)

    # バリデーションデータでの予測と評価
    y_length_val_pred = best_model_length.predict(X_val)
    y_size_val_pred = best_model_size.predict(X_val)

    length_mae = mean_absolute_error(y_length_val, y_length_val_pred)
    length_r2 = r2_score(y_length_val, y_length_val_pred)
    length_acc = accuracy_score((y_length_val - y_length_val_pred).abs() < 5, [True]*len(y_length_val))

    size_mae = mean_absolute_error(y_size_val, y_size_val_pred)
    size_r2 = r2_score(y_size_val, y_size_val_pred)
    size_acc = accuracy_score((y_size_val - y_size_val_pred).abs() < 1, [True]*len(y_size_val))

    print("\nTrainRSME:")
    print(study_length.best_value)
    print(study_size.best_value)
    
    # モデルの保存
    length_model_path = './model/gb_model_length.pkl'
    size_model_path = './model/gb_model_size.pkl'
    joblib.dump(best_model_length, length_model_path)
    joblib.dump(best_model_size, size_model_path)

    return length_model_path, size_model_path, study_length.best_value, study_size.best_value, length_mae, size_mae, length_r2, size_r2, length_acc, size_acc



def mlp_regression(X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val):
    trial_number = 100  # Optunaの試行回数を設定
    
    def objective_length(trial):
        # ハイパーパラメータの提案
        params = {
            'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50)]),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive'])
        }

        model = MLPRegressor(**params, random_state=42)
        model.fit(X_train, y_length_train)
        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_length_val, y_pred, squared=False)
        
        return rmse

    def objective_size(trial):
        # ハイパーパラメータの提案
        params = {
            'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50)]),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive'])
        }

        model = MLPRegressor(**params, random_state=42)
        model.fit(X_train, y_size_train)
        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_size_val, y_pred, squared=False)
        
        return rmse

    # Optunaのスタディの作成と最適化の実行
    study_length = optuna.create_study(direction='minimize')
    study_size = optuna.create_study(direction='minimize')
    study_length.optimize(objective_length, n_trials=trial_number)
    study_size.optimize(objective_size, n_trials=trial_number)

    # 最適なハイパーパラメータの取得
    best_params_length = study_length.best_params
    best_params_size = study_size.best_params

    # 最適なモデルの定義
    best_model_length = MLPRegressor(**best_params_length, random_state=42)
    best_model_size = MLPRegressor(**best_params_size, random_state=42)

    # 最適なモデルの訓練
    best_model_length.fit(X_train, y_length_train)
    best_model_size.fit(X_train, y_size_train)
    
    # バリデーションデータでの予測と評価
    y_length_val_pred = best_model_length.predict(X_val)
    y_size_val_pred = best_model_size.predict(X_val)

    length_mae = mean_absolute_error(y_length_val, y_length_val_pred)
    length_r2 = r2_score(y_length_val, y_length_val_pred)
    length_acc = accuracy_score((y_length_val - y_length_val_pred).abs() < 5, [True]*len(y_length_val))

    size_mae = mean_absolute_error(y_size_val, y_size_val_pred)
    size_r2 = r2_score(y_size_val, y_size_val_pred)
    size_acc = accuracy_score((y_size_val - y_size_val_pred).abs() < 1, [True]*len(y_size_val))

    # モデルの保存
    length_model_path = './model/mlp_model_length.pkl'
    size_model_path = './model/mlp_model_size.pkl'
    joblib.dump(best_model_length, length_model_path)
    joblib.dump(best_model_size, size_model_path)

    return length_model_path, size_model_path, study_length.best_value, study_size.best_value, length_mae, size_mae, length_r2, size_r2, length_acc, size_acc