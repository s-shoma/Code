# ---------- 通常の半教師あり学習(アヤメデータ) ---------- #

import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer

# ファイルの読み込み
X_train = np.loadtxt("X_train.csv", delimiter=",", dtype="float")
Y_train = np.loadtxt("Y_train.csv", delimiter=",", dtype="float")
X_unlabel = np.loadtxt("X_unlabel.csv", delimiter=",", dtype="float")
X_test = np.loadtxt("X_test.csv", delimiter=",", dtype="float")
Y_test = np.loadtxt("Y_test.csv", delimiter=",", dtype="float")

# モデルの構築
model = Sequential()
model.add(InputLayer(input_shape=(4,)))
model.add(Dense(500, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

Loop_count = 1
Loop_final = 1
startTime = time.time()

# ループ
for Loop in range(Loop_final):
  print("ループ回数:", Loop_count)

  if Loop_count == 1:
    Y_train_categorical = keras.utils.to_categorical(Y_train, 3)
    print(Y_train_categorical)

    epochs = 300
    batch_size = 256
    model.fit(X_train, Y_train_categorical, batch_size=batch_size, epochs=epochs) #学習

  # 仮ラベルの予測
  Y_predict = model.predict(X_unlabel)

  # ラベル付け
  Y_predict_label = np.zeros([Y_predict.shape[0], 1])
  for i in range(Y_predict.shape[0]):
    if np.argmax(Y_predict[i]) == 0:
      Y_predict_label[i] = 0
    elif np.argmax(Y_predict[i]) == 1:
      Y_predict_label[i] = 1
    elif np.argmax(Y_predict[i]) == 2:
        Y_predict_label[i] = 2

  # 予測データの描画
  #predict_data_fig = plt.figure(figsize=(6, 6))
  #ax = predict_data_fig.add_subplot(1, 1, 1)
  #for i in range(X_unlabel.shape[0]):
    #if Y_predict_label[i] == 0:
      #ax.scatter(X_unlabel[i, 0], X_unlabel[i, 1], c='red', s=4)
    #elif Y_predict_label[i] == 1:
      #ax.scatter(X_unlabel[i, 0], X_unlabel[i, 1], c='blue', s=4)

  #margin = 1.1
  #ax.set_xlim(-2, 2)
  #ax.set_ylim(-2, 2)
  #plt.title("predict result")
  #plt.show()

  # 信頼度上位5点を抽出
  X_add = np.zeros([5, 4])
  Y_add = np.zeros([5, 1])

  for i in range(X_add.shape[0]):
    max_data = 0
    max_data_index = 0

    for t in range(Y_predict.shape[0]):
      if max_data < np.amax(Y_predict[t]):
        max_data = np.amax(Y_predict[t], axis=0)
        max_data_index = t

    X_add[i] = X_unlabel[max_data_index]
    Y_add[i] = Y_predict_label[max_data_index]

    Y_predict = np.delete(Y_predict, max_data_index, axis=0)
    Y_predict_label = np.delete(Y_predict_label, max_data_index, axis=0)
    X_unlabel = np.delete(X_unlabel, max_data_index, axis=0)

  X_train = np.concatenate((X_train, X_add), axis=0)
  Y_train = np.append(Y_train, Y_add)

  # データ追加後の学習
  Y_train_categorical = keras.utils.to_categorical(Y_train-1, 4)
  #print(Y_train_categorical)

  model.fit(X_train, Y_train_categorical, batch_size=batch_size, epochs=epochs)

  if Loop_count < Loop_final:
    # ループ毎の評価
    Y_test_categorical = keras.utils.to_categorical(Y_test-1, 4)
    classifier = model.evaluate(X_test, Y_test_categorical, verbose=0)
    print("%dループ目の精度" %Loop_count)
    print('Cross entropy:{0:.3f}, Accuracy:{1:.3f}'.format(classifier[0], classifier[1]))

  elif Loop_count == Loop_final:
    # 最終評価
    Y_test_categorical = keras.utils.to_categorical(Y_test, 3)
    classifier = model.evaluate(X_test, Y_test_categorical, verbose=0)
    print("最終精度")
    print('Cross entropy:{0:.3f}, Accuracy:{1:.3f}'.format(classifier[0], classifier[1]))

  Loop_count += 1


calculation_time = time.time() - startTime
print("Calculation time:{0:.3f}sec".format(calculation_time))

#print("X_unlabelの要素数:", X_unlabel.shape[0])
#print("X_trainの要素数:", X_train.shape[0])
