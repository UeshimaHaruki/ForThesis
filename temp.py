


import pandas as pd
import numpy as np



j = 90 #初期値  40
i = 55 #要素数 89-40+1=50 143-89+1 = 55



# 元のデータ（例えば0から始まる整数のリスト）
data = np.arange(i)

# 25個のデータを等間隔で選ぶためのインデックスを計算
indices = np.round(np.linspace(0, len(data) - 1, 25)).astype(int)
a = []
a = set(data) ^ set(indices)

a = list(a)

for n in range(len(a)):
    a[n] = int(a[n])
    
for m in range(len(a)):
    a[m] += j

# 選択されたデータ
sampled_data = data[indices]
print(data)
print(indices)
print(a)


#print(indices + j)
# l1 = ['a', 'b', 'c']
# l2 = ['b', 'c', 'd']
# l3 = ['c', 'd', 'e']
# print(set(l1) ^ set(l2))
# # {'d', 'a'}