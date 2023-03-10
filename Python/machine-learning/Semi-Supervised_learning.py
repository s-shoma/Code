# ---------- 通常の半教師あり学習 ---------- #

import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer
import winsound

# ファイルの読み込み
X_train = np.loadtxt("X_train.csv", delimiter=",", dtype="float")
Y_train = np.loadtxt("Y_train.csv", delimiter=",", dtype="float")
X_unlabel = np.loadtxt("X_unlabel.csv", delimiter=",", dtype="float")
X_test = np.loadtxt("X_test.csv", delimiter=",", dtype="float")
Y_test = np.loadtxt("Y_test.csv", delimiter=",", dtype="float")

# モデルの構築
main_model = Sequential()
main_model.add(InputLayer(input_shape=(2,)))
main_model.add(Dense(500, activation='relu'))
main_model.add(Dense(250, activation='relu'))
main_model.add(Dense(100, activation='relu'))
main_model.add(Dense(50, activation='relu'))
main_model.add(Dense(2, activation='softmax'))
main_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

Loop_count = 1
Loop_final = 120
startTime = time.time()

# ループ
for Loop in range(Loop_final):
  print("ループ回数:", Loop_count)

  if Loop_count == 1:
    Y_train_categorical = keras.utils.to_categorical(Y_train, 2)
    #print(Y_train_categorical)

    # 学習
    main_epochs = 300
    main_batch_size = 128
    main_model.fit(X_train, Y_train_categorical, batch_size=main_batch_size, epochs=main_epochs) #学習

  # 仮ラベルの予測
  Y_predict = main_model.predict(X_unlabel)
  #print(Y_predict)

  # 仮ラベル付け
  Y_predict_label = np.zeros([Y_predict.shape[0], 1])
  #print(Y_predict_label)

  for i in range(Y_predict.shape[0]):
    if np.argmax(Y_predict[i]) == 0:
      Y_predict_label[i] = 0
    elif np.argmax(Y_predict[i]) == 1:
      Y_predict_label[i] = 1

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

  # 信頼度上位20点を抽出
  X_add = np.zeros([20, 2])
  Y_add = np.zeros([20, 1])

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
  Y_train_categorical = keras.utils.to_categorical(Y_train, 2)
  #print(Y_train_categorical)

  main_model.fit(X_train, Y_train_categorical, batch_size=main_batch_size, epochs=main_epochs)

  if Loop_count < Loop_final:
    # ループ毎の評価
    Y_test_categorical = keras.utils.to_categorical(Y_test, 2)
    classifier = main_model.evaluate(X_test, Y_test_categorical, verbose=0)
    print("%dループ目の精度" %Loop_count)
    print('Cross entropy:{0:.3f}, Accuracy:{1:.3f}'.format(classifier[0], classifier[1]))

  elif Loop_count == Loop_final:
    # 最終評価
    Y_test_categorical = keras.utils.to_categorical(Y_test, 2)
    classifier = main_model.evaluate(X_test, Y_test_categorical, verbose=0)
    print("最終精度")
    print('Cross entropy:{0:.3f}, Accuracy:{1:.3f}'.format(classifier[0], classifier[1]))

  Loop_count += 1

# 追加されたデータの描画
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
for i in range(X_train.shape[0]):
  if Y_train[i] == 0:
    ax.scatter(X_train[i, 0], X_train[i, 1], c='red', s=4)
  elif Y_train[i] == 1:
    ax.scatter(X_train[i, 0], X_train[i, 1], c='blue', s=4)

margin = 1.1
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
plt.title("X_train Data")
plt.show()


calculation_time = time.time() - startTime
print("Calculation time:{0:.3f}sec".format(calculation_time))

print("X_unlabelの要素数:", X_unlabel.shape[0])
print("X_trainの要素数:", X_train.shape[0])

winsound.Beep(2000, 500)
