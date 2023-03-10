# ---------- 教師あり学習 ---------- #

import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer

# ファイルの読み込み
X_train = np.loadtxt("X_train.csv", delimiter=",", dtype="float")  # 入力データ
Y_train = np.loadtxt("Y_train.csv", delimiter=",", dtype="float")  # 対応するラベルデータ
#X_unlabel = np.loadtxt("X_unlabel.csv", delimiter=",", dtype="float")  #未ラベルのデータ
X_test = np.loadtxt("X_test.csv", delimiter=",", dtype="float")  # テストデータ
Y_test = np.loadtxt("Y_test.csv", delimiter=",", dtype="float")  # テストデータ

# モデルの構築
Y__train_categorical = keras.utils.to_categorical(Y_train, 2)
print(Y__train_categorical)

main_model = Sequential()
main_model.add(InputLayer(input_shape=(2,)))
main_model.add(Dense(500, activation='relu'))
main_model.add(Dense(250, activation='relu'))
main_model.add(Dense(100, activation='relu'))
main_model.add(Dense(50, activation='relu'))
main_model.add(Dense(2, activation='sigmoid'))
main_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

startTime = time.time()

# 学習
main_epochs = 300
main_batch_size = 128
main_model.fit(X_train, Y__train_categorical, batch_size=main_batch_size, epochs=main_epochs)

# モデルの評価
Y_test_categorical = keras.utils.to_categorical(Y_test, 2)
classifier = main_model.evaluate(X_test, Y_test_categorical, verbose=0)
print('cross entropy{0:.3f}, accuracy{1:.3f}'.format(classifier[0], classifier[1]))

calculation_time = time.time() - startTime
print("Calculation time:{0:.3f}sec".format(calculation_time))   #時間
