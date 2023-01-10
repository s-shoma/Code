# ---------- svmを用いた提案手法 ---------- #
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
Loop_final = 10
startTime = time.time()

# ループ
for Loop in range(Loop_final):
  print("ループ回数:", Loop_count)

  if Loop_count == 1:
    # 学習
    main_model = svm.SVC(kernel="rbf", gamma="scale")
    main_model.fit(X_train, Y_train)

  # 仮ラベルの予測・付与
  Y_predict = main_model.decision_function(X_unlabel)
  Y_predict_label = main_model.predict(X_unlabel)
  np.savetxt("Y_predict.csv", X=Y_predict, delimiter=",")
  np.savetxt("Y_predict_label.csv", X=Y_predict_label, delimiter=",")

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

  X_model2 = X_unlabel[shuffle_num2[0:num], :]
  Y_model2 = Y_predict_label[shuffle_num2[0:num]]

  X_model3 = X_unlabel[shuffle_num3[0:num], :]
  Y_model3 = Y_predict_label[shuffle_num3[0:num]]

  X_model4 = X_unlabel[shuffle_num4[0:num], :]
  Y_model4 = Y_predict_label[shuffle_num4[0:num]]

  X_model5 = X_unlabel[shuffle_num5[0:num], :]
  Y_model5 = Y_predict_label[shuffle_num5[0:num]]

  model1 = svm.SVC(kernel="rbf", gamma="scale")
  model1.fit(X_model1, Y_model1)

  model2 = svm.SVC(kernel="rbf", gamma="scale")
  model2.fit(X_model2, Y_model2)

  model3 = svm.SVC(kernel="rbf", gamma="scale")
  model3.fit(X_model3, Y_model3)

  model4 = svm.SVC(kernel="rbf", gamma="scale")
  model4.fit(X_model4, Y_model4)

  model5 = svm.SVC(kernel="rbf", gamma="scale")
  model5.fit(X_model5, Y_model5)

  # 正答率の比較
  accuracy_rank = [0]

  print("model1の正答率")
  evaluation1 = model1.predict(X_train)
  model1_accuracy = np.count_nonzero(evaluation1 == Y_train) / Y_train.shape[0]
  print(model1_accuracy)
  accuracy_rank.append(model1_accuracy)

  print("model2の正答率")
  evaluation2 = model2.predict(X_train)
  model2_accuracy = np.count_nonzero(evaluation2 == Y_train) / Y_train.shape[0]
  print(model2_accuracy)
  accuracy_rank.append(model2_accuracy)

  print("model3の正答率")
  evaluation3 = model3.predict(X_train)
  model3_accuracy = np.count_nonzero(evaluation3 == Y_train) / Y_train.shape[0]
  print(model3_accuracy)
  accuracy_rank.append(model3_accuracy)

  print("model4の正答率")
  evaluation4 = model4.predict(X_train)
  model4_accuracy = np.count_nonzero(evaluation4 == Y_train) / Y_train.shape[0]
  print(model4_accuracy)
  accuracy_rank.append(model4_accuracy)

  print("model5の正答率")
  evaluation5 = model5.predict(X_train)
  model5_accuracy = np.count_nonzero(evaluation5 == Y_train) / Y_train.shape[0]
  print(model5_accuracy)
  accuracy_rank.append(model5_accuracy)

  # 最も精度の良いモデルの番号
  top_accuracy = np.argmax(accuracy_rank)
  print("Top_accuracy_model:", top_accuracy)

  # データの追加・削除
  if top_accuracy==1:
    X_train = np.concatenate((X_train, X_unlabel[shuffle_num1[0:num], :]), axis=0)
    Y_train = np.append(Y_train, Y_predict_label[shuffle_num1[0:num]])

    X_unlabel = np.delete(X_unlabel, shuffle_num1[0:num], axis=0)
    Y_predict_label = np.delete(Y_predict_label, shuffle_num1[0:num], axis=0)

  elif top_accuracy==2:
    X_train = np.concatenate((X_train, X_unlabel[shuffle_num2[0:num], :]), axis=0)
    Y_train = np.append(Y_train, Y_predict_label[shuffle_num2[0:num]])

    X_unlabel = np.delete(X_unlabel, shuffle_num2[0:num], axis=0)
    Y_predict_label = np.delete(Y_predict_label, shuffle_num2[0:num], axis=0)

  elif top_accuracy==3:
    X_train = np.concatenate((X_train, X_unlabel[shuffle_num3[0:num], :]), axis=0)
    Y_train = np.append(Y_train, Y_predict_label[shuffle_num3[0:num]])

    X_unlabel = np.delete(X_unlabel, shuffle_num3[0:num], axis=0)
    Y_predict_label = np.delete(Y_predict_label, shuffle_num3[0:num], axis=0)

  elif top_accuracy==4:
    X_train = np.concatenate((X_train, X_unlabel[shuffle_num4[0:num], :]), axis=0)
    Y_train = np.append(Y_train, Y_predict_label[shuffle_num4[0:num]])

    X_unlabel = np.delete(X_unlabel, shuffle_num4[0:num], axis=0)
    Y_predict_label = np.delete(Y_predict_label, shuffle_num4[0:num], axis=0)

  elif top_accuracy==5:
    X_train = np.concatenate((X_train, X_unlabel[shuffle_num5[0:num], :]), axis=0)
    Y_train = np.append(Y_train, Y_predict_label[shuffle_num5[0:num]])

    X_unlabel = np.delete(X_unlabel, shuffle_num5[0:num], axis=0)
    Y_predict_label = np.delete(Y_predict_label, shuffle_num5[0:num], axis=0)

  # データ追加後の学習
  main_model.fit(X_train, Y_train)

  # 評価
  evaluation = main_model.predict(X_test)
  main_model_accuracy = np.count_nonzero(evaluation == Y_test) / Y_test.shape[0]

  if Loop_count < Loop_final:
    #ループ毎の評価
    print("%dループ目の精度"%Loop_count)
    print("Accuracy:", main_model_accuracy)

  elif Loop_count == Loop_final:
    #最終評価
    print("最終精度")
    print("Accuracy:", main_model_accuracy)

  Loop_count += 1

# 追加されたデータの描画
#fig = plt.figure(figsize=(6, 6))
#ax = fig.add_subplot(1, 1, 1)
#for i in range(X_train.shape[0]):
  #if Y_train[i] == 0:
    #ax.scatter(X_train[i, 0], X_train[i, 1], c='red', s=4)
  #elif Y_train[i] == 1:
    #ax.scatter(X_train[i, 0], X_train[i, 1], c='blue', s=4)

#margin = 1.1
#ax.set_xlim(-2, 2)
#ax.set_ylim(-2, 2)
#plt.title("X_train Data")
#plt.show()

calculation_time = time.time() - startTime
print("Calculation time:{0:.3f}sec".format(calculation_time))

print("X_unlabelの要素数:", X_unlabel.shape[0])
print("X_trainの要素数:", X_train.shape[0])
