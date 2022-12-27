# ---------- テスト用データ作成 ---------- #
import numpy as np
import matplotlib.pyplot as plt
import math

N1 = 1000
N2 = 1000
D = 2         # クラス数
N = N1 + N2   # データ総数

Z = np.linspace(-1, 1, N)
Z_theta = 2 * math.pi * ((Z + 2.0) / 2.0)

W = np.zeros([N, D])
W[:, 0] = np.sin(Z_theta)
W[:, 1] = np.cos(Z_theta)

x1 = np.zeros([N1, D])
x2 = np.zeros([N2, D])

x1[:, :] = W[0:N1, :]
x2[:, :] = W[N1:N, :]

x1[:, 0] += 0
x1[:, 1] += -0.5
x2[:, 0] += 0
x2[:, 1] += 0.5

val = 0.2
x1[:, 0] += np.random.normal(0, val, N1)
x1[:, 1] += np.random.normal(0, val, N1)
x2[:, 0] += np.random.normal(0, val, N2)
x2[:, 1] += np.random.normal(0, val, N2)

X_test = np.zeros([N, D])
X_test[0:N1, :] = x1[:, :]
X_test[N1:N, :] = x2[:, :]
print("データ数:", X_test.shape[0])

Y_test = np.zeros([N, 1])
Y_test[N1:N, 0] = 1
print("ラベル数:", Y_test.shape[0])

# 全データの描画
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x1[:, 0], x1[:, 1], c='red', s=4)
ax.scatter(x2[:, 0], x2[:, 1], c='blue', s=4)

margin =1.1
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
plt.title("Test Data")
plt.show()

# ファイルの保存
np.savetxt("X_test.csv", X=X_test, delimiter=",")
np.savetxt("Y_test.csv", X=Y_test, delimiter=",")
