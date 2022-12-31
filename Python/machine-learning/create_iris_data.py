import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# アヤメデータの取得
iris = datasets.load_iris()
X = iris.data
Y = iris.target
#print(Y)

# 標準化
sc = StandardScaler()

# 標準化させる値
X = sc.fit_transform(X)

# 主成分分析の実行
pca = PCA()
pca.fit(X)

# データを主成分に変換する
X = pca.transform(X)

# 全データの描画
iris_fig = plt.figure(figsize=(7, 4))
ax = iris_fig.add_subplot(1, 1, 1)
for i in range(X.shape[0]):
  if Y[i] == 0:
    ax.scatter(X[i, 0], X[i, 1], c='red', s=4)
  elif Y[i] == 1:
    ax.scatter(X[i, 0], X[i, 1], c='blue', s=4)
  elif Y[i] == 2:
    ax.scatter(X[i, 0], X[i, 1], c='green', s=4)
plt.title('Total iris data')
plt.savefig('iris_data_fig')
plt.show()

class0_data = X[0:50]
class0_label = Y[0:50]

class1_data = X[50:100]
class1_label = Y[50:100]

class2_data = X[100:150]
class2_label = Y[100:150]

num0 = np.arange(class0_data.shape[0])
num1 = np.arange(class1_data.shape[0])
num2 = np.arange(class2_data.shape[0])

rand0 = np.random.choice(num0, size=15, replace=False)
rand1 = np.random.choice(num1, size=15, replace=False)
rand2 = np.random.choice(num2, size=15, replace=False)

X_test0 = class0_data[rand0]
Y_test0 = class0_label[rand0]
X_test1 = class1_data[rand1]
Y_test1 = class1_label[rand1]
X_test2 = class2_data[rand2]
Y_test2 = class2_label[rand2]

X_test = np.concatenate((X_test0, X_test1, X_test2), axis=0)
Y_test = np.concatenate((Y_test0, Y_test1, Y_test2), axis=0)

class0_data = np.delete(class0_data, rand0, axis=0)
class0_label = np.delete(class0_label, rand0, axis=0)
class1_data = np.delete(class1_data, rand1, axis=0)
class1_label = np.delete(class1_label, rand1, axis=0)
class2_data = np.delete(class2_data, rand2, axis=0)
class2_label = np.delete(class2_label, rand2, axis=0)

num0 = np.arange(class0_data.shape[0])
num1 = np.arange(class1_data.shape[0])
num2 = np.arange(class2_data.shape[0])

rand0 = np.random.choice(num0, size=5, replace=False)
rand1 = np.random.choice(num1, size=5, replace=False)
rand2 = np.random.choice(num2, size=5, replace=False)

X_train0 = class0_data[rand0]
Y_train0 = class0_label[rand0]
X_train1 = class1_data[rand1]
Y_train1 = class1_label[rand1]
X_train2 = class2_data[rand2]
Y_train2 = class2_label[rand2]

X_train = np.concatenate((X_train0, X_train1, X_train2), axis=0)
Y_train = np.concatenate((Y_train0, Y_train1, Y_train2), axis=0)

class0_data = np.delete(class0_data, rand0, axis=0)
class0_label = np.delete(class0_label, rand0, axis=0)
class1_data = np.delete(class1_data, rand1, axis=0)
class1_label = np.delete(class1_label, rand1, axis=0)
class2_data = np.delete(class2_data, rand2, axis=0)
class2_label = np.delete(class2_label, rand2, axis=0)

X_unlabel = np.concatenate((class0_data, class1_data, class2_data), axis=0)
Y_unlabel = np.concatenate((class0_label, class1_label, class2_label), axis=0)

# ラベルありデータの描画
iris_train_fig = plt.figure(figsize=(7, 4))
ax = iris_train_fig.add_subplot(1, 1, 1)
for i in range(X_train.shape[0]):
  if Y_train[i] == 0:
    ax.scatter(X_train[i, 0], X_train[i, 1], c='red', s=4)
  elif Y_train[i] == 1:
    ax.scatter(X_train[i, 0], X_train[i, 1], c='blue', s=4)
  elif Y_train[i] == 2:
    ax.scatter(X_train[i, 0], X_train[i, 1], c='green', s=4)
plt.title('iris train data')
plt.show()

# ラベルなしデータの描画
iris_unlabel_fig = plt.figure(figsize=(7, 4))
ax = iris_unlabel_fig.add_subplot(1, 1, 1)
for i in range(X_unlabel.shape[0]):
  if Y_unlabel[i] == 0:
    ax.scatter(X_unlabel[i, 0], X_unlabel[i, 1], c='red', s=4)
  elif Y_unlabel[i] == 1:
    ax.scatter(X_unlabel[i, 0], X_unlabel[i, 1], c='blue', s=4)
  elif Y_unlabel[i] == 2:
    ax.scatter(X_unlabel[i, 0], X_unlabel[i, 1], c='green', s=4)
plt.title('iris unlabel data')
plt.show()

# テストデータの描画
iris_test_fig = plt.figure(figsize=(7, 4))
ax = iris_test_fig.add_subplot(1, 1, 1)
for i in range(X_test.shape[0]):
  if Y_test[i] == 0:
    ax.scatter(X_test[i, 0], X_test[i, 1], c='red', s=4)
  elif Y_test[i] == 1:
    ax.scatter(X_test[i, 0], X_test[i, 1], c='blue', s=4)
  elif Y_test[i] == 2:
    ax.scatter(X_test[i, 0], X_test[i, 1], c='green', s=4)
plt.title('iris test data')
plt.show()

# ファイルの保存
np.savetxt("X_train.csv", X=X_train, delimiter=",")
np.savetxt("Y_train.csv", X=Y_train, delimiter=",")
np.savetxt("X_unlabel.csv", X=X_unlabel, delimiter=",")
np.savetxt("Y_unlabel.csv", X=Y_unlabel, delimiter=",")
np.savetxt("X_test.csv", X=X_test, delimiter=",")
np.savetxt("Y_test.csv", X=Y_test, delimiter=",")
