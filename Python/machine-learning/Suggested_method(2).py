# ---------- 提案手法(信頼度による抽出あり) ---------- #
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer

# ファイルの読み込み
X_train = np.loadtxt("X_train.csv", delimiter=",", dtype="float")       # 入力データ
Y_train = np.loadtxt("Y_train.csv", delimiter=",", dtype="float")       # ラベルデータ
X_unlabel = np.loadtxt("X_unlabel.csv", delimiter=",", dtype="float")   # 未ラベルのデータ
X_test = np.loadtxt("X_test.csv", delimiter=",", dtype="float")         # テストデータ
Y_test = np.loadtxt("Y_test.csv", delimiter=",", dtype="float")         # テストデータ

# メインモデルの構築
main_model = Sequential()
main_model.add(InputLayer(input_shape=(2,)))
main_model.add(Dense(500, activation='relu'))
main_model.add(Dense(250, activation='relu'))
main_model.add(Dense(100, activation='relu'))
main_model.add(Dense(50, activation='relu'))
main_model.add(Dense(2, activation='softmax'))
main_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

Loop_count = 1
Loop_final = 150
startTime = time.time()

# ループ
for Loop in range(Loop_final):
  print("ループ回数: ", Loop_count)

  Y_train_categorical = keras.utils.to_categorical(Y_train-1, 2)
  #print(Y_train_categorical)

  # 学習
  main_epochs = 300
  main_batch_size = 256
  main_model.fit(X_train, Y_train_categorical, batch_size=main_batch_size, epochs=main_epochs)

  # 予測
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

  np.savetxt("Y_predict.csv", X=Y_predict, delimiter=",")
  np.savetxt("Y_predict_label.csv", X=Y_predict_label, delimiter=",")

  # 予測結果の描画
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

  # 野生データ(予測したデータ)の学習
  num = 200

  X_model1 = np.zeros([num, 2])
  Y_model1 = np.zeros([num, 1])

  X_model2 = np.zeros([num, 2])
  Y_model2 = np.zeros([num, 1])

  X_model3 = np.zeros([num, 2])
  Y_model3 = np.zeros([num, 1])

  X_model4 = np.zeros([num, 2])
  Y_model4 = np.zeros([num, 1])

  X_model5 = np.zeros([num, 2])
  Y_model5 = np.zeros([num, 1])

  shuffle_num1 = np.arange(X_unlabel.shape[0])
  np.random.shuffle(shuffle_num1)

  shuffle_num2 = np.arange(X_unlabel.shape[0])
  np.random.shuffle(shuffle_num2)

  shuffle_num3 = np.arange(X_unlabel.shape[0])
  np.random.shuffle(shuffle_num3)

  shuffle_num4 = np.arange(X_unlabel.shape[0])
  np.random.shuffle(shuffle_num4)

  shuffle_num5 = np.arange(X_unlabel.shape[0])
  np.random.shuffle(shuffle_num5)

  X_model1 = X_unlabel[shuffle_num1[0:num], :]
  Y_model1 = Y_predict_label[shuffle_num1[0:num]]
  Y_model1_reliability = Y_predict[shuffle_num1[0:num], :]    #信頼度

  X_model2 = X_unlabel[shuffle_num2[0:num], :]
  Y_model2 = Y_predict_label[shuffle_num2[0:num]]
  Y_model2_reliability = Y_predict[shuffle_num2[0:num], :]    #信頼度

  X_model3 = X_unlabel[shuffle_num3[0:num], :]
  Y_model3 = Y_predict_label[shuffle_num3[0:num]]
  Y_model3_reliability = Y_predict[shuffle_num3[0:num], :]    #信頼度

  X_model4 = X_unlabel[shuffle_num4[0:num], :]
  Y_model4 = Y_predict_label[shuffle_num4[0:num]]
  Y_model4_reliability = Y_predict[shuffle_num4[0:num], :]    #信頼度

  X_model5 = X_unlabel[shuffle_num5[0:num], :]
  Y_model5 = Y_predict_label[shuffle_num5[0:num]]
  Y_model5_reliability = Y_predict[shuffle_num5[0:num], :]    #信頼度

  # 野生データ学習用のモデル構築
  # モデル1
  model1 = Sequential()
  model1.add(InputLayer(input_shape=(2,)))
  model1.add(Dense(200, activation='relu'))
  model1.add(Dense(100, activation='relu'))
  model1.add(Dense(50, activation='relu'))
  model1.add(Dense(2, activation='softmax'))
  model1.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

  # モデル2
  model2 = Sequential()
  model2.add(InputLayer(input_shape=(2,)))
  model2.add(Dense(200, activation='relu'))
  model2.add(Dense(100, activation='relu'))
  model2.add(Dense(50, activation='relu'))
  model2.add(Dense(2, activation='softmax'))
  model2.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

  # モデル3
  model3 = Sequential()
  model3.add(InputLayer(input_shape=(2,)))
  model3.add(Dense(200, activation='relu'))
  model3.add(Dense(100, activation='relu'))
  model3.add(Dense(50, activation='relu'))
  model3.add(Dense(2, activation='softmax'))
  model3.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

  # モデル4
  model4 = Sequential()
  model4.add(InputLayer(input_shape=(2,)))
  model4.add(Dense(200, activation='relu'))
  model4.add(Dense(100, activation='relu'))
  model4.add(Dense(50, activation='relu'))
  model4.add(Dense(2, activation='softmax'))
  model4.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

  # モデル5
  model5 = Sequential()
  model5.add(InputLayer(input_shape=(2,)))
  model5.add(Dense(200, activation='relu'))
  model5.add(Dense(100, activation='relu'))
  model5.add(Dense(50, activation='relu'))
  model5.add(Dense(2, activation='softmax'))
  model5.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

  Y_model1_categorical = keras.utils.to_categorical(Y_model1-1, 2)
  Y_model2_categorical = keras.utils.to_categorical(Y_model2-1, 2)
  Y_model3_categorical = keras.utils.to_categorical(Y_model3-1, 2)
  Y_model4_categorical = keras.utils.to_categorical(Y_model4-1, 2)
  Y_model5_categorical = keras.utils.to_categorical(Y_model5-1, 2)

  # 学習
  epochs = 200
  batch_size = 64

  model1.fit(X_model1, Y_model1_categorical, batch_size=batch_size, epochs=epochs)
  model2.fit(X_model2, Y_model2_categorical, batch_size=batch_size, epochs=epochs)
  model3.fit(X_model3, Y_model3_categorical, batch_size=batch_size, epochs=epochs)
  model4.fit(X_model4, Y_model4_categorical, batch_size=batch_size, epochs=epochs)
  model5.fit(X_model5, Y_model5_categorical, batch_size=batch_size, epochs=epochs)

  # 正答率の比較
  accuracy_rank = [0]

  print("model1の正答率")
  classifier1 = model1.evaluate(X_train, Y_train_categorical, verbose=0)
  print('cross entropy1:{0:.3f}, accuracy1:{1:.3f}'.format(classifier1[0], classifier1[1]))
  accuracy_rank.append(classifier1[1])

  print("model2の正答率")
  classifier2 = model2.evaluate(X_train, Y_train_categorical, verbose=0)
  print('cross entropy2:{0:.3f}, accuracy2:{1:.3f}'.format(classifier2[0], classifier2[1]))
  accuracy_rank.append(classifier2[1])

  print("model3の正答率")
  classifier3 = model3.evaluate(X_train, Y_train_categorical, verbose=0)
  print('cross entropy3:{0:.3f}, accuracy3:{1:.3f}'.format(classifier3[0], classifier3[1]))
  accuracy_rank.append(classifier3[1])

  print("model4の正答率")
  classifier4 = model4.evaluate(X_train, Y_train_categorical, verbose=0)
  print('cross entropy4:{0:.3f}, accuracy4:{1:.3f}'.format(classifier4[0], classifier4[1]))
  accuracy_rank.append(classifier4[1])

  print("model5の正答率")
  classifier5 = model5.evaluate(X_train, Y_train_categorical, verbose=0)
  print('cross entropy5:{0:.3f}, accuracy5:{1:.3f}'.format(classifier5[0], classifier5[1]))
  accuracy_rank.append(classifier5[1])

  # 最も精度の良いモデルの番号
  top_accuracy = np.argmax(accuracy_rank)
  print("Top_accuracy_model:", top_accuracy)

# 信頼度によるデータの抽出
  X_add = np.zeros([10, 2])
  Y_add = np.zeros([10, 1])
  delete_data = []
  add_num = X_add.shape[0]

  if top_accuracy == 1:
    for i in range(X_add.shape[0]):
      max_data = 0
      max_data_index = 0

      for t in range(Y_model1_reliability.shape[0]):
        if max_data < np.amax(Y_model1_reliability[t]):
          max_data = np.amax(Y_model1_reliability[t], axis=0)
          max_data_index = t

      X_add[i] = X_model1[max_data_index]
      Y_add[i] = Y_model1[max_data_index]
      delete_data.append(max_data_index)

      Y_model1 = np.delete(Y_model1, max_data_index, axis=0)
      Y_model1_reliability = np.delete(Y_model1_reliability, max_data_index, axis=0)
      shuffle_num1 = np.delete(shuffle_num1, max_data_index, axis=0)

    X_train = np.concatenate((X_train, X_add), axis=0)
    Y_train = np.append(Y_train, Y_add)
    X_unlabel = np.delete(X_unlabel, delete_data[0:add_num], axis=0)

  elif top_accuracy == 2:
    for i in range(X_add.shape[0]):
      max_data = 0
      max_data_index = 0

      for t in range(Y_model2_reliability.shape[0]):
        if max_data < np.amax(Y_model2_reliability[t]):
          max_data = np.amax(Y_model2_reliability[t], axis=0)
          max_data_index = t

      X_add[i] = X_model2[max_data_index]
      Y_add[i] = Y_model2[max_data_index]
      delete_data.append(max_data_index)

      Y_model2 = np.delete(Y_model2, max_data_index, axis=0)
      Y_model2_reliability = np.delete(Y_model2_reliability, max_data_index, axis=0)
      shuffle_num2 = np.delete(shuffle_num2, max_data_index, axis=0)

    X_train = np.concatenate((X_train, X_add), axis=0)
    Y_train = np.append(Y_train, Y_add)
    X_unlabel = np.delete(X_unlabel, delete_data[0:add_num], axis=0)

  elif top_accuracy == 3:
    for i in range(X_add.shape[0]):
      max_data = 0
      max_data_index = 0

      for t in range(Y_model3_reliability.shape[0]):
        if max_data < np.amax(Y_model3_reliability[t]):
          max_data = np.amax(Y_model3_reliability[t], axis=0)
          max_data_index = t

      X_add[i] = X_model3[max_data_index]
      Y_add[i] = Y_model3[max_data_index]
      delete_data.append(max_data_index)

      Y_model3 = np.delete(Y_model3, max_data_index, axis=0)
      Y_model3_reliability = np.delete(Y_model3_reliability, max_data_index, axis=0)
      shuffle_num3 = np.delete(shuffle_num3, max_data_index, axis=0)

    X_train = np.concatenate((X_train, X_add), axis=0)
    Y_train = np.append(Y_train, Y_add)
    X_unlabel = np.delete(X_unlabel, delete_data[0:add_num], axis=0)

  elif top_accuracy == 4:
    for i in range(X_add.shape[0]):
      max_data = 0
      max_data_index = 0

      for t in range(Y_model4_reliability.shape[0]):
        if max_data < np.amax(Y_model4_reliability[t]):
          max_data = np.amax(Y_model4_reliability[t], axis=0)
          max_data_index = t

      X_add[i] = X_model4[max_data_index]
      Y_add[i] = Y_model4[max_data_index]
      delete_data.append(max_data_index)

      Y_model4 = np.delete(Y_model4, max_data_index, axis=0)
      Y_model4_reliability = np.delete(Y_model4_reliability, max_data_index, axis=0)
      shuffle_num4 = np.delete(shuffle_num4, max_data_index, axis=0)

    X_train = np.concatenate((X_train, X_add), axis=0)
    Y_train = np.append(Y_train, Y_add)
    X_unlabel = np.delete(X_unlabel, delete_data[0:add_num], axis=0)

  elif top_accuracy == 5:
    for i in range(X_add.shape[0]):
      max_data = 0
      max_data_index = 0

      for t in range(Y_model5_reliability.shape[0]):
        if max_data < np.amax(Y_model5_reliability[t]):
          max_data = np.amax(Y_model5_reliability[t], axis=0)
          max_data_index = t

      X_add[i] = X_model5[max_data_index]
      Y_add[i] = Y_model5[max_data_index]
      delete_data.append(max_data_index)

      Y_model5 = np.delete(Y_model5, max_data_index, axis=0)
      Y_model5_reliability = np.delete(Y_model5_reliability, max_data_index, axis=0)
      shuffle_num5 = np.delete(shuffle_num5, max_data_index, axis=0)

    X_train = np.concatenate((X_train, X_add), axis=0)
    Y_train = np.append(Y_train, Y_add)
    X_unlabel = np.delete(X_unlabel, delete_data[0:add_num], axis=0)

  # データ追加後の学習
  Y_train_categorical = keras.utils.to_categorical(Y_train-1, 2)
  print(Y_train_categorical)

  main_model.fit(X_train, Y_train_categorical, batch_size=main_batch_size, epochs=main_epochs)

  # 評価
  Y_test_categorical = keras.utils.to_categorical(Y_test-1, 2)
  classifier = main_model.evaluate(X_test, Y_test_categorical, verbose=0)
  if Loop_count < Loop_final:
    # ループ毎のmain_modelの評価
    print("%dループ目の精度"%Loop_count)
    print('main_model_Cross entropy:{0:.3f}, main_model_Accuracy:{1:.3f}'.format(classifier[0], classifier[1]))

  elif Loop_count == Loop_final:
    # 最終のmain_modelの評価
    print("最終精度")
    print('main_model_Cross entropy:{0:.3f}, main_model_Accuracy:{1:.3f}'.format(classifier[0], classifier[1]))

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
