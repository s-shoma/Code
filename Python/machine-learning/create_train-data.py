# ---------- 訓練用データ作成 ---------- #
import numpy as np
import matplotlib.pyplot as plt
import math

N1 = 1500
N2 = 1500
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

X = np.zeros([N, D])
X[0:N1, :] = x1[:, :]
X[N1:N, :] = x2[:, :]
print("総データ数:", X.shape[0])

Y = np.zeros([N, 1])
Y[N1:N, 0] = 1
#print("ラベル数:", Y.shape[0])

# 全データの描画
all_data_fig = plt.figure(figsize=(6, 6))
ax = all_data_fig.add_subplot(1, 1, 1)
ax.scatter(x1[:, 0], x1[:, 1], c='red', s=4)
ax.scatter(x2[:, 0], x2[:, 1], c='blue', s=4)

margin = 1.1
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
plt.title("Total Data")
plt.show()

# ----- ラベルなしデータの作成 ----- #
Num = np.arange(X.shape[0])
#print(Num.shape[0])

rand = np.random.choice(Num, size=500, replace=False)
#print("rand:", rand)

X_train = np.zeros([rand.shape[0], 2])
Y_train = np.zeros([rand.shape[0], 1])
for i in range(rand.shape[0]):
  X_train[i] = X[rand[i]]
  Y_train[i] = Y[rand[i]]

print("ラベルありデータ数:", X_train.shape[0])
#print(Y_train.shape[0])

X_unlabel = np.delete(X, rand, axis=0)
Y_unlabel = np.delete(Y, rand, axis=0)
print("ラベルなしデータ数:", X_unlabel.shape[0])
#print(Y_unlabel.shape[0])

# ラベルありデータの描画
train_data_fig = plt.figure(figsize=(6, 6))
ax = train_data_fig.add_subplot(1, 1, 1)
for i in range(X_train.shape[0]):
  if Y_train[i] == 0:
    ax.scatter(X_train[i, 0], X_train[i, 1], c='red', s=4)
  elif Y_train[i] == 1:
    ax.scatter(X_train[i, 0], X_train[i, 1], c='blue', s=4)
margin = 1.1
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
plt.title("Train Data")
plt.show()

# ラベルなしデータの描画
unlabel_data_fig = plt.figure(figsize=(6, 6))
ax = unlabel_data_fig.add_subplot(1, 1, 1)
for i in range(X_unlabel.shape[0]):
  if Y_unlabel[i] == 0:
    ax.scatter(X_unlabel[i, 0], X_unlabel[i, 1], c='red', s=4)
  elif Y_unlabel[i] == 1:
    ax.scatter(X_unlabel[i, 0], X_unlabel[i, 1], c='blue', s=4)
margin = 1.1
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
plt.title("Unlabel Data")
plt.show()

# ファイルの保存
np.savetxt("X_train.csv", X=X_train, delimiter=",")
np.savetxt("Y_train.csv", X=Y_train, delimiter=",")
np.savetxt("X_unlabel.csv", X=X_unlabel, delimiter=",")
np.savetxt("Y_unlabel.csv", X=Y_unlabel, delimiter=",")
