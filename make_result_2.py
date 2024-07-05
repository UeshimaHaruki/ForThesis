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
    train_A_data = pd.read_csv('./dataset/Ueshima_TrainingData/medid_1_v8_A_train.csv')
    train_B_data = pd.read_csv('./dataset/Ueshima_TrainingData/medid_1_v8_B_train.csv')
    train_C_data = pd.read_csv('./dataset/Ueshima_TrainingData/medid_1_v8_C_train.csv')
    train_D_data = pd.read_csv('./dataset/Ueshima_TrainingData/medid_1_v8_D_train.csv')
    train_E_data = pd.read_csv('./dataset/Ueshima_TrainingData/medid_1_v8_E_train.csv')
    train_F_data = pd.read_csv('./dataset/Ueshima_TrainingData/medid_1_v8_F_train.csv')
    train_G_data = pd.read_csv('./dataset/Ueshima_TrainingData/medid_1_v8_G_train.csv')
    best_length_data = pd.read_csv('./result_1/top_3_length_models.csv')
    best_size_data = pd.read_csv('./result_1/top_3_size_models.csv')
    
    val_data = pd.read_csv('./dataset/exp_dataset/medid_1_v8/val.csv')
    # test_data = pd.read_csv('./dataset/exp_dataset/medid_1_v8/test.csv')

    pre_colomuns=['Sex', 'Age', 'Aneu_neck', 'Aneu_width', 'Aneu_height', 'Aneu_volume', 'Aneu_location', 'Adj_tech', 'Is_bleb', 'coil_length1', 'coil_size1']
    
    train_data = train_data[pre_colomuns].dropna()
    train_A_data = train_A_data[pre_colomuns].dropna()
    train_B_data = train_B_data[pre_colomuns].dropna()
    train_C_data = train_C_data[pre_colomuns].dropna()
    train_D_data = train_D_data[pre_colomuns].dropna()
    train_E_data = train_E_data[pre_colomuns].dropna()
    train_F_data = train_F_data[pre_colomuns].dropna()
    train_G_data = train_G_data[pre_colomuns].dropna()
    val_data = val_data[pre_colomuns].dropna()
    
    
    colomuns=['Sex', 'Age', 'Aneu_neck', 'Aneu_width', 'Aneu_height', 'Aneu_volume', 'Aneu_location', 'Adj_tech', 'Is_bleb']
    
    # 説明変数と目的変数の設定
    X_train = train_data[colomuns]
    y_length_train = train_data['coil_length1']
    y_size_train = train_data['coil_size1']
    
    X_train_A = train_A_data[colomuns]
    y_length_train_A = train_A_data['coil_length1']
    y_size_train_A = train_A_data['coil_size1']
    
    X_train_B = train_B_data[colomuns]
    y_length_train_B = train_B_data['coil_length1']
    y_size_train_B = train_B_data['coil_size1']
    
    X_train_C = train_C_data[colomuns]
    y_length_train_C = train_C_data['coil_length1']
    y_size_train_C= train_C_data['coil_size1']
    
    X_train_D = train_D_data[colomuns]
    y_length_train_D = train_D_data['coil_length1']
    y_size_train_D = train_D_data['coil_size1']
    
    X_train_E = train_E_data[colomuns]
    y_length_train_E = train_E_data['coil_length1']
    y_size_train_E = train_E_data['coil_size1']
    
    X_train_F = train_F_data[colomuns]
    y_length_train_F = train_F_data['coil_length1']
    y_size_train_F = train_F_data['coil_size1']
    
    X_train_G = train_G_data[colomuns]
    y_length_train_G = train_G_data['coil_length1']
    y_size_train_G = train_G_data['coil_size1']
    
    X_val = val_data[colomuns]
    y_length_val = val_data['coil_length1']
    y_size_val = val_data['coil_size1']
    
    
    
    
    # カテゴリ変数をダミー変数に変換
    X_train = pd.get_dummies(X_train)
    X_train_A = pd.get_dummies(X_train_A)
    X_train_B = pd.get_dummies(X_train_B)
    X_train_C = pd.get_dummies(X_train_C)   
    X_train_D = pd.get_dummies(X_train_D)
    X_train_E = pd.get_dummies(X_train_E)
    X_train_F = pd.get_dummies(X_train_F)
    X_train_G = pd.get_dummies(X_train_G)
    X_val = pd.get_dummies(X_val)
    # X_test = pd.get_dummies(X_test)
    
    print(X_train.shape)
    print(X_train_A.shape)
    print(X_train_B.shape)
    print(X_train_C.shape)
    print(X_train_D.shape)
    print(X_train_E.shape)
    print(X_train_F.shape)
    print(X_train_G.shape)
    print(X_val.shape)
    
    
    # print(X_train.isnull().sum())
    # print(X_train_A.isnull().sum())
    # print(X_train_B.isnull().sum())
    # print(X_train_C.isnull().sum())
    # print(X_train_D.isnull().sum())
    # print(X_train_E.isnull().sum())
    # print(X_train_F.isnull().sum())
    # print(X_train_G.isnull().sum())
    # print(X_val.isnull().sum())
    
    # 訓練データとバリデーションデータのカラムを一致させる
    X_train, _ = X_train.align(X_train_G, join='right', axis=1, fill_value=0)
    X_train_A, _ = X_train_A.align(X_train_G, join='right', axis=1, fill_value=0)
    X_train_B, _ = X_train_B.align(X_train_G, join='right', axis=1, fill_value=0)
    X_train_C, _ = X_train_C.align(X_train_G, join='right', axis=1, fill_value=0)
    X_train_D, _ = X_train_D.align(X_train_G, join='right', axis=1, fill_value=0)
    X_train_E, _ = X_train_E.align(X_train_G, join='right', axis=1, fill_value=0)
    X_train_F, _ = X_train_F.align(X_train_G, join='right', axis=1, fill_value=0)
    X_val, _ = X_val.align(X_train_G, join='right', axis=1, fill_value=0)
    
    print(X_train.shape)
    print(X_train_A.shape)
    print(X_train_B.shape)
    print(X_train_C.shape)
    print(X_train_D.shape)
    print(X_train_E.shape)
    print(X_train_F.shape)
    print(X_train_G.shape)
    print(X_val.shape)
    
    # print(X_train.isnull().sum())
    # print(X_train_A.isnull().sum())
    # print(X_train_B.isnull().sum())
    # print(X_train_C.isnull().sum())
    # print(X_train_D.isnull().sum())
    # print(X_train_E.isnull().sum())
    # print(X_train_F.isnull().sum())
    # print(X_train_G.isnull().sum())
    # print(X_val.isnull().sum())
    
    
    # 標準化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train_A = scaler.transform(X_train_A)
    X_train_B = scaler.transform(X_train_B)
    X_train_C = scaler.transform(X_train_C)
    X_train_D = scaler.transform(X_train_D)
    X_train_E = scaler.transform(X_train_E)
    X_train_F = scaler.transform(X_train_F)
    X_train_G = scaler.transform(X_train_G)
    X_val = scaler.transform(X_val)
    # X_test = scaler.transform(X_test)
    
    
    
    
    
    best_length_function = best_length_data['Function'][0]  
    best_size_function = best_size_data['Function'][0]    
    
    print(best_length_function)
    print(best_size_function)
    
    # 関数の呼び出し
    train_length_path, _, train_length_rmse, _, train_length_mae, _, train_length_r2, _, train_length_acc, _ = globals()[best_length_function](X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val)
    train_A_length_path, _, train_A_length_rmse, _, train_A_length_mae, _, train_A_length_r2, _, train_A_length_acc, _ = globals()[best_length_function](X_train_A, X_val, y_length_train_A, y_length_val, y_size_train_A, y_size_val)
    train_B_length_path, _, train_B_length_rmse, _, train_B_length_mae, _, train_B_length_r2, _, train_B_length_acc, _ = globals()[best_length_function](X_train_B, X_val, y_length_train_B, y_length_val, y_size_train_B, y_size_val)
    train_C_length_path, _, train_C_length_rmse, _, train_C_length_mae, _, train_C_length_r2, _, train_C_length_acc, _ = globals()[best_length_function](X_train_C, X_val, y_length_train_C, y_length_val, y_size_train_C, y_size_val)
    train_D_length_path, _, train_D_length_rmse, _, train_D_length_mae, _, train_D_length_r2, _, train_D_length_acc, _ = globals()[best_length_function](X_train_D, X_val, y_length_train_D, y_length_val, y_size_train_D, y_size_val)
    train_E_length_path, _, train_E_length_rmse, _, train_E_length_mae, _, train_E_length_r2, _, train_E_length_acc, _ = globals()[best_length_function](X_train_E, X_val, y_length_train_E, y_length_val, y_size_train_E, y_size_val)
    train_F_length_path, _, train_F_length_rmse, _, train_F_length_mae, _, train_F_length_r2, _, train_F_length_acc, _ = globals()[best_length_function](X_train_F, X_val, y_length_train_F, y_length_val, y_size_train_F, y_size_val)
    train_G_length_path, _, train_G_length_rmse, _, train_G_length_mae, _, train_G_length_r2, _, train_G_length_acc, _ = globals()[best_length_function](X_train_G, X_val, y_length_train_G, y_length_val, y_size_train_G, y_size_val)
    
    
    _, train_size_path, _, train_size_rmse, _, train_size_mae, _, train_size_r2, _, train_size_acc = globals()[best_size_function](X_train, X_val, y_length_train, y_length_val, y_size_train, y_size_val)
    _, train_A_size_path, _, train_A_size_rmse, _, train_A_size_mae, _, train_A_size_r2, _, train_A_size_acc = globals()[best_size_function](X_train_A, X_val, y_length_train_A, y_length_val, y_size_train_A, y_size_val)
    _, train_B_size_path, _, train_B_size_rmse, _, train_B_size_mae, _, train_B_size_r2, _, train_B_size_acc = globals()[best_size_function](X_train_B, X_val, y_length_train_B, y_length_val, y_size_train_B, y_size_val)
    _, train_C_size_path, _, train_C_size_rmse, _, train_C_size_mae, _, train_C_size_r2, _, train_C_size_acc = globals()[best_size_function](X_train_C, X_val, y_length_train_C, y_length_val, y_size_train_C, y_size_val)
    _, train_D_size_path, _, train_D_size_rmse, _, train_D_size_mae, _, train_D_size_r2, _, train_D_size_acc = globals()[best_size_function](X_train_D, X_val, y_length_train_D, y_length_val, y_size_train_D, y_size_val)
    _, train_E_size_path, _, train_E_size_rmse, _, train_E_size_mae, _, train_E_size_r2, _, train_E_size_acc = globals()[best_size_function](X_train_E, X_val, y_length_train_E, y_length_val, y_size_train_E, y_size_val)
    _, train_F_size_path, _, train_F_size_rmse, _, train_F_size_mae, _, train_F_size_r2, _, train_F_size_acc = globals()[best_size_function](X_train_F, X_val, y_length_train_F, y_length_val, y_size_train_F, y_size_val)
    _, train_G_size_path, _, train_G_size_rmse, _, train_G_size_mae, _, train_G_size_r2, _, train_G_size_acc = globals()[best_size_function](X_train_G, X_val, y_length_train_G, y_length_val, y_size_train_G, y_size_val)
    
    
    
    # Length RMSEの結果をまとめる
    length_results = {
        'Criteria': ['original', 'A', 'B', 'C', 'D', 'E', 'F', 'G'],
        'Length RMSE': [train_length_rmse, train_A_length_rmse, train_B_length_rmse, train_C_length_rmse, train_D_length_rmse, train_E_length_rmse, train_F_length_rmse, train_G_length_rmse],
        'Length MAE': [train_length_mae, train_A_length_mae, train_B_length_mae, train_C_length_mae, train_D_length_mae, train_E_length_mae, train_F_length_mae, train_G_length_mae],
        'Length R2': [train_length_r2, train_A_length_r2, train_B_length_r2, train_C_length_r2, train_D_length_r2, train_E_length_r2, train_F_length_r2, train_G_length_r2],
        'Length Accuracy': [train_length_acc, train_A_length_acc, train_B_length_acc, train_C_length_acc, train_D_length_acc, train_E_length_acc, train_F_length_acc, train_G_length_acc]
    }
    length_results_df = pd.DataFrame(length_results)
    length_results_df.to_csv('./result_2/length_train_results.csv', index=False)

    # Size RMSEの結果をまとめる
    size_results = {
        'Criteria': ['original', 'A', 'B', 'C', 'D', 'E', 'F', 'G'],
        'Size RMSE': [train_size_rmse, train_A_size_rmse, train_B_size_rmse, train_C_size_rmse, train_D_size_rmse, train_E_size_rmse, train_F_size_rmse, train_G_size_rmse],
        'Size MAE': [train_size_mae, train_A_size_mae, train_B_size_mae, train_C_size_mae, train_D_size_mae, train_E_size_mae, train_F_size_mae, train_G_size_mae],
        'Size R2': [train_size_r2, train_A_size_r2, train_B_size_r2, train_C_size_r2, train_D_size_r2, train_E_size_r2, train_F_size_r2, train_G_size_r2],
        'Size Accuracy': [train_size_acc, train_A_size_acc, train_B_size_acc, train_C_size_acc, train_D_size_acc, train_E_size_acc, train_F_size_acc, train_G_size_acc]
    }
    size_results_df = pd.DataFrame(size_results)
    size_results_df.to_csv('./result_2/size_train_results.csv', index=False)


    # 上位3モデルの選定
    model_paths_and_rmse = [
        ('original', train_length_path, train_length_rmse, train_length_mae, train_length_r2, train_length_acc, train_size_path, train_size_rmse, train_size_mae, train_size_r2, train_size_acc ),
        ('A', train_A_length_path, train_A_length_rmse, train_A_length_mae, train_A_length_r2, train_A_length_acc, train_A_size_path, train_A_size_rmse, train_A_size_mae, train_A_size_r2, train_A_size_acc),
        ('B', train_B_length_path, train_B_length_rmse, train_B_length_mae, train_B_length_r2, train_B_length_acc, train_B_size_path, train_B_size_rmse, train_B_size_mae, train_B_size_r2, train_B_size_acc),
        ('C', train_C_length_path, train_C_length_rmse, train_C_length_mae, train_C_length_r2, train_C_length_acc, train_C_size_path, train_C_size_rmse, train_C_size_mae, train_C_size_r2, train_C_size_acc),
        ('D', train_D_length_path, train_D_length_rmse, train_D_length_mae, train_D_length_r2, train_D_length_acc, train_D_size_path, train_D_size_rmse, train_D_size_mae, train_D_size_r2, train_D_size_acc),
        ('E', train_E_length_path, train_E_length_rmse, train_E_length_mae, train_E_length_r2, train_E_length_acc, train_E_size_path, train_E_size_rmse, train_E_size_mae, train_E_size_r2, train_E_size_acc),
        ('F', train_F_length_path, train_F_length_rmse, train_F_length_mae, train_F_length_r2, train_F_length_acc, train_F_size_path, train_F_size_rmse, train_F_size_mae, train_F_size_r2, train_F_size_acc),
        ('G', train_G_length_path, train_G_length_rmse, train_G_length_mae, train_G_length_r2, train_G_length_acc, train_G_size_path, train_G_size_rmse, train_G_size_mae, train_G_size_r2, train_G_size_acc)
    ]
    

    # Length RMSEで上位3モデルを選択
    model_paths_and_rmse.sort(key=lambda x: x[2])  # Length RMSEでソート
    top_1_length_criteria = model_paths_and_rmse[:1]

    # Size RMSEで上位3モデルを選択
    model_paths_and_rmse.sort(key=lambda x: x[7])  # Size RMSEでソート
    top_1_size_criteria = model_paths_and_rmse[:1]
    
    # 上位3モデルの結果をCSVファイルとして保存
    top_1_length_df = pd.DataFrame([(x[4], x[0], x[2]) for x in top_1_length_criteria], columns=['Function', 'Length Model Path', 'Length RMSE'])
    top_1_length_df.to_csv('./result_2/top_3_length_models.csv', index=False)

    top_1_size_df = pd.DataFrame([(x[4], x[1], x[3]) for x in top_1_size_criteria], columns=['Function', 'Size Model Path', 'Size RMSE'])
    top_1_size_df.to_csv('./result_2/top_3_size_models.csv', index=False)

        
if __name__ == "__main__":
    main()