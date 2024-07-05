from src.regression import linear_regression
from src.regression import svr_regression
from src.regression import lgbm_regression
from src.regression import rf_regression
from src.regression import gb_regression
from src.regression import mlp_regression

import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

def main():
    # データの読み込み
    train_data = pd.read_csv('./dataset/exp_dataset/medid_1_v8/train.csv')
    val_data = pd.read_csv('./dataset/exp_dataset/medid_1_v8/val.csv')
    # test_data = pd.read_csv('./dataset/exp_dataset/medid_1_v8/test.csv')

    # 説明変数と目的変数の設定
    X_train = train_data[['Sex', 'Age', 'Aneu_neck', 'Aneu_width', 'Aneu_height', 'Aneu_volume', 'Aneu_location', 'Adj_tech', 'Is_bleb']]
    y_length_train = train_data['coil_length1']
    y_size_train = train_data['coil_size1']

    X_val = val_data[['Sex', 'Age', 'Aneu_neck', 'Aneu_width', 'Aneu_height', 'Aneu_volume', 'Aneu_location', 'Adj_tech', 'Is_bleb']]
    y_length_val = val_data['coil_length1']
    y_size_val = val_data['coil_size1']

    # X_test = test_data[['Sex', 'Age', 'Aneu_neck', 'Aneu_width', 'Aneu_height', 'Aneu_volume', 'Aneu_location', 'Adj_tech', 'Is_bleb']]
    # y_length_test = test_data['coil_length1']
    # y_size_test = test_data['coil_size1']

    # カテゴリ変数をダミー変数に変換
    print(X_train.shape)
    X_train = pd.get_dummies(X_train)
    X_val = pd.get_dummies(X_val)
    # X_test = pd.get_dummies(X_test)
    # print(X_train.shape)
    
    # 訓練データとバリデーションデータのカラムを一致させる
    X_train, X_val = X_train.align(X_val, join='left', axis=1, fill_value=0)
    # X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # 標準化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    # X_test = scaler.transform(X_test)
    
    # print(X_test.shape)
    
    ''''''
    
    # 関数の呼び出し
    linear_results = linear_regression(X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val)
    svr_results = svr_regression(X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val)
    lgbm_results = lgbm_regression(X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val)
    rf_results = rf_regression(X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val)
    gb_results = gb_regression(X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val)
    mlp_results = mlp_regression(X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val)

    # Length RMSEの結果をまとめる
    length_results = {
        'Model': ['Linear', 'SVR', 'LGBM', 'Random Forest', 'Gradient Boosting', 'MLP'],
        'Length RMSE': [linear_results[2], svr_results[2], lgbm_results[2], rf_results[2], gb_results[2], mlp_results[2]],
        'Length MAE': [linear_results[4], svr_results[4], lgbm_results[4], rf_results[4], gb_results[4], mlp_results[4]],
        'Length R2': [linear_results[6], svr_results[6], lgbm_results[6], rf_results[6], gb_results[6], mlp_results[6]],
        'Length Accuracy': [linear_results[8], svr_results[8], lgbm_results[8], rf_results[8], gb_results[8], mlp_results[8]]
    }
    length_results_df = pd.DataFrame(length_results)
    length_results_df.to_csv('./result_1/length_train_results.csv', index=False)

    # Size RMSEの結果をまとめる
    size_results = {
        'Model': ['Linear', 'SVR', 'LGBM', 'Random Forest', 'Gradient Boosting', 'MLP'],
        'Size RMSE': [linear_results[3], svr_results[3], lgbm_results[3], rf_results[3], gb_results[3], mlp_results[3]],
        'Size MAE': [linear_results[5], svr_results[5], lgbm_results[5], rf_results[5], gb_results[5], mlp_results[5]],
        'Size R2': [linear_results[7], svr_results[7], lgbm_results[7], rf_results[7], gb_results[7], mlp_results[7]],
        'Size Accuracy': [linear_results[9], svr_results[9], lgbm_results[9], rf_results[9], gb_results[9], mlp_results[9]]
    }
    size_results_df = pd.DataFrame(size_results)
    size_results_df.to_csv('./result_1/size_train_results.csv', index=False)

    # 上位3モデルの選定
    model_paths_and_rmse = [
        (linear_results[0], linear_results[1], linear_results[2], linear_results[3], 'linear_regression'),
        (svr_results[0], svr_results[1], svr_results[2], svr_results[3], 'svr_regression'),
        (lgbm_results[0], lgbm_results[1], lgbm_results[2], lgbm_results[3], 'lgbm_regression'),
        (rf_results[0], rf_results[1], rf_results[2], rf_results[3], 'rf_regression'),
        (gb_results[0], gb_results[1], gb_results[2], gb_results[3], 'gb_regression'),
        (mlp_results[0], mlp_results[1], mlp_results[2], mlp_results[3], 'mlp_regression')
    ]

    # Length RMSEで上位3モデルを選択
    model_paths_and_rmse.sort(key=lambda x: x[2])  # Length RMSEでソート
    top_3_length_models = model_paths_and_rmse[:3]

    # Size RMSEで上位3モデルを選択
    model_paths_and_rmse.sort(key=lambda x: x[3])  # Size RMSEでソート
    top_3_size_models = model_paths_and_rmse[:3]

    # 上位3モデルの結果をCSVファイルとして保存
    top_3_length_df = pd.DataFrame([(x[4], x[0], x[2]) for x in top_3_length_models], columns=['Function', 'Length Model Path', 'Length RMSE'])
    top_3_length_df.to_csv('./result_1/top_3_length_models.csv', index=False)

    top_3_size_df = pd.DataFrame([(x[4], x[1], x[3]) for x in top_3_size_models], columns=['Function', 'Size Model Path', 'Size RMSE'])
    top_3_size_df.to_csv('./result_1/top_3_size_models.csv', index=False)

if __name__ == "__main__":
    main()