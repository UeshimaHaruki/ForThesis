import pandas as pd
import numpy as np

# CSVファイルのパス
file_path = 'C:/Users/ohwada/ue-remo/J_Bridge_datasets/test.csv'
output_file_path_1 = './data_rows_processed.csv'
output_file_path = './b_matrix.csv'

# CSVファイルを読み込む
df = pd.read_csv(file_path, encoding='shift-jis', header=0)

center_value = [90, 40, 113, 131, 135, 95, 124, 125, 90, 297 ,181 ,216, 95, 162, 186, 340, 233, 349, 132, 117, 42, 143, 97, 61, 149, 165, 101, 47, 127]
center_column = []

# 各行をfor文で回して特定の値がどの列に入っているかを見つける
for row_index, row in df.iterrows():
    if row_index % 2 == 1:
        continue
    num = int(row_index / 2)
    value_to_find = center_value[num]
    for i, j in row.items():
        if j == value_to_find:
            col_index = df.columns.get_loc(i)
            center_column.append(col_index)


# dfを奇数番号の行だけにする
df_skip = df.iloc[::2]
# -------------------------------------------------------------
j = df_skip.iloc[:,2] #初期値
j = j.reset_index(drop=True)

df_center_column = pd.DataFrame(center_column)
df_center_column = df_center_column - 1

count = 0

# データフレームの作成と更新
k = pd.DataFrame(index=range(len(df_center_column)), columns=[1], dtype=float)

for idx, value in enumerate(df_center_column[0]):
    k.loc[idx, 1] = df_skip.iloc[idx, value]

k = np.array(k)
j = np.array(j)
# k を j の形状に再整形
k = k.reshape(29)
i = (k - j) + 1



# すべての操作を i の各要素に対して実行するための関数
def process_element(i_element, j_element):
    data = np.arange(i_element)
    indices = np.round(np.linspace(0, len(data) - 1, 25)).astype(int)
    a = list(set(data) ^ set(indices))

    for n in range(len(a)):
        a[n] = int(a[n])

    for m in range(len(a)):
        a[m] += j_element

    sampled_data = data[indices]
    return data, indices, a, sampled_data

# i のすべての要素に対して処理を実行し、a を収集
all_a = [process_element(i_element, j_element)[2] for i_element, j_element in zip(i, j)]

# 最長の a の長さを取得
max_len = max(len(a) for a in all_a)

# すべての a を同じ長さにパディング
all_a_padded = [a + [0] * (max_len - len(a)) for a in all_a]

# 行列に変換
a_matrix = np.array(all_a_padded)

# 0 を nan に変更
a_matrix[a_matrix == 0] = np.nan

# 'number'列がNaNである行を説明的な行として分離
descriptive_rows = df[df['number'].isna()]

# 'number'列がNaNでない行をデータ行として分離
data_rows = df[df['number'].notna()]

# インデックスをリセット
descriptive_rows.reset_index(drop=True, inplace=True)
data_rows.reset_index(drop=True, inplace=True)

# Replace numbers in descriptive_rows with NaN based on a_matrix
for row_idx, replace_values in enumerate(a_matrix):
    print(row_idx)
    for value in replace_values:
        if not np.isnan(value):
            for col in descriptive_rows.columns:
                if descriptive_rows.loc[row_idx, col] == value:
                    print(value)
                    data_rows.loc[row_idx, col] = np.nan
                    break
# print(descriptive_rows.head())
# # NaN値を左詰めにする関数
def left_justify_na(df):
    # 各行に対して処理を行う
    for index, row in df.iterrows():
        # NaNでない値を取得
        non_na_values = row.dropna().values
        # NaNの個数を数える
        num_na = len(row) - len(non_na_values)
        # 左詰めにして再割り当て
        new_row = np.concatenate([non_na_values, [np.nan] * num_na])
        df.loc[index] = new_row
    return df

# # NaN値を左詰めにする
# descriptive_rows = left_justify_na(descriptive_rows)
# -------------------------------------------------------------
# # j = df_skip.iloc[:,-2] #初期値
# # nan以外の最後の値を格納するリストを作成
j_1 = []

for index, row in df_skip.iterrows():
    # NaN以外の値のインデックスを取得
    non_nan_values = row.dropna().values
    if len(non_nan_values) > 0:
        # 最後の値をリストに追加
        j_1.append(non_nan_values[-2])
    else:
        # 全てNaNの場合はNaNを追加
        j_1.append(np.nan)


# j = j.reset_index(drop=True)
df_center_column = pd.DataFrame(center_column)
df_center_column = df_center_column +1

count = 0

# データフレームの作成と更新
k = pd.DataFrame(index=range(len(df_center_column)), columns=[1], dtype=float)

for idx, value in enumerate(df_center_column[0]):
    k.loc[idx, 1] = df_skip.iloc[idx, value]

k = np.array(k)
j = np.array(j)

# k を j の形状に再整形
k = k.reshape(29)

j=k
i = (j_1 - k) + 1
# すべての操作を i の各要素に対して実行するための関数
def process_element(i_element, j_element):
    data = np.arange(i_element)
    indices = np.round(np.linspace(0, len(data) - 1, 25)).astype(int)
    b = list(set(data) ^ set(indices))
    if i_element == 25:
        sample = [99999]
        #sampleにnp.nanを24回いれる
        for i in range(24):
            sample.append(np.nan)
        return data, indices, sample, sample

    for n in range(len(b)):
        b[n] = int(b[n])

    for m in range(len(b)):
        b[m] += j_element

    sampled_data = data[indices]
    return data, indices, b, sampled_data

# i のすべての要素に対して処理を実行し、a を収集
all_b = [process_element(i_element, j_element)[2] for i_element, j_element in zip(i, j)]

# 最長の b の長さを取得
max_len = max(len(b) for b in all_b)

# すべての b を同じ長さにパディング
all_b_padded = [b + [0] * (max_len - len(b)) for b in all_b]

# 行列に変換
b_matrix = np.array(all_b_padded)

# 0 を nan に変更
b_matrix[b_matrix == 0] = np.nan
# # 'number'列がNaNである行を説明的な行として分離
# descriptive_rows = df[df['number'].isna()]

# # 'number'列がNaNでない行をデータ行として分離
# data_rows = df[df['number'].notna()]

# # インデックスをリセット
# descriptive_rows.reset_index(drop=True, inplace=True)
# data_rows.reset_index(drop=True, inplace=True)

# Replace numbers in descriptive_rows with NaN based on b_matrix

for row_idx, replace_values in enumerate(b_matrix):
    if replace_values[0] != 99999:
        # print(replace_values)
        for value in replace_values:
            # if row_idx == 0:
                # print(value)
            if not np.isnan(value):
                for col in descriptive_rows.columns:
                    if descriptive_rows.loc[row_idx, col] == value:
                        data_rows.loc[row_idx, col] = np.nan
                        break
    else:
        continue
# print(descriptive_rows.head())

# # # NaN値を左詰めにする関数
# def left_justify_na(df):
#     # 各行に対して処理を行う
#     for index, row in df.iterrows():
#         # NaNでない値を取得
#         non_na_values = row.dropna().values
#         # NaNの個数を数える
#         num_na = len(row) - len(non_na_values)
#         # 左詰めにして再割り当て
#         new_row = np.concatenate([non_na_values, [np.nan] * num_na])
#         df.loc[index] = new_row
#     return df










# # NaN値を左詰めにする
data_rows = left_justify_na(data_rows)

# descriptive_rowsをCSVファイルとして出力

data_rows.to_csv(output_file_path_1, index=False)
b_matrix=pd.DataFrame(b_matrix)
b_matrix.to_csv(output_file_path, index=False)
