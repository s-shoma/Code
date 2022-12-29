# ---------- svmを使った通常の半教師あり学習 ---------- #
# 正の値 : 1
# 負の値 : 0
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import svm
from tensorflow import keras

# ファイルの読み込み
X_train = np.loadtxt("X_train.csv", delimiter=",", dtype="float")
Y_train = np.loadtxt("Y_train.csv", delimiter=",", dtype="float")
X_unlabel = np.loadtxt("X_unlabel.csv", delimiter=",", dtype="float")
X_test = np.loadtxt("X_test.csv", delimiter=",", dtype="float")
Y_test = np.loadtxt("Y_test.csv", delimiter=",", dtype="float")

Loop_count = 1
Loop_final = 100
startTime = time.time()

# ループ
for Loop in range(Loop_final):
  print("ループ回数:", Loop_count)

  if Loop_count == 1:
    # 学習
    model = svm.SVC(kernel="rbf", gamma="scale")
    model.fit(X_train, Y_train)

  # 仮ラベルの予測・付与
  Y_predict = model.decision_function(X_unlabel)
  Y_predict_label = model.predict(X_unlabel)
  np.savetxt("Y_predict.csv", X=Y_predict, delimiter=",")
  np.savetxt("Y_predict_label.csv", X=Y_predict_label, delimiter=",")

  # 信頼度上位10点を訓練データとして追加
  X_add = np.zeros([20, 2])
  Y_add = np.zeros([20, 1])

  for i in range(X_add.shape[0]):
    for t in range(Y_predict.shape[0]):
      abs_max = 0
      abx_max_index = 0

      if abs_max < abs(Y_predict[t]):
        abs_max = abs(Y_predict[t])
        abs_max_index = t

    X_add[i] = X_unlabel[abs_max_index]
    Y_add[i] = Y_predict_label[abs_max_index]

    X_unlabel = np.delete(X_unlabel, abs_max_index, axis=0)
    Y_predict = np.delete(Y_predict, abs_max_index, axis=0)
    Y_predict_label = np.delete(Y_predict_label, abs_max_index, axis=0)

  X_train = np.concatenate((X_train, X_add), axis=0)
  Y_train = np.append(Y_train, Y_add)

  # データ追加後の学習
  model.fit(X_train, Y_train)

  if Loop_count < Loop_final:
    # ループ毎の評価
    evaluation = model.predict(X_test)
    model_accuracy = np.count_nonzero(evaluation == Y_test) / Y_test.shape[0]
    print("%dループ目の精度"%Loop_count)
    print("Accuracy:", model_accuracy)

  elif Loop_count == Loop_final:
    # 最終評価
    evaluation = model.predict(X_test)
    model_accuracy = np.count_nonzero(evaluation == Y_test) / Y_test.shape[0]
    print("最終精度")
    print("Accuracy:", model_accuracy)

  Loop_count += 1

# 追加されたデータの描画
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
for i in range(X_train.shape[0]):
  if Y_train[i] == 0:
    ax.scatter(X_train[i, 0], X_train[i, 1], c='red', s=4)
  elif Y_train[i] == 1:
    ax.scatter(X_train[i, 0], X_train[i, 1], c='blue', s=4)

calculation_time = time.time() - startTime
print("Calculation time:{0:.3f}sec".format(calculation_time))

print("X_unlabelの要素数:", X_unlabel.shape[0])
print("X_trainの要素数:", X_train.shape[0])
