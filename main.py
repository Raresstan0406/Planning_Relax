from sklearn import neural_network
import numpy as np
import os

file_path=os.getcwd()
filedata=[]
data=[]
data2=[]
A1=[]
A2=[]
A3=[]
A4=[]
A5=[]
A6=[]
A7=[]
A8=[]
A9=[]
A10=[]
A11=[]
A12=[]
A13=[]
with open(file_path+ '\plrx.txt' , 'r') as file:
    filedata=file.read().splitlines()
    for i in filedata:
        data.append(i.split('\t '))
    for i in range(len(data)):
        data2 = data2 + data[i]
    for i in range(0, len(data2), 13):
        A1.append(float(data2[i]))

    for i in range(1, len(data2), 13):
        A2.append(float(data2[i]))

    for i in range(2, len(data2), 13):
        A3.append(float(data2[i]))

    for i in range(3, len(data2), 13):
        A4.append(float(data2[i]))

    for i in range(4, len(data2), 13):
        A5.append(float(data2[i]))

    for i in range(5, len(data2), 13):
        A6.append(float(data2[i]))

    for i in range(6, len(data2), 13):
        A7.append(float(data2[i]))

    for i in range(7, len(data2), 13):
        A8.append(float(data2[i]))

    for i in range(8, len(data2), 13):
        A9.append(float(data2[i]))

    for i in range(9, len(data2), 13):
        A10.append(float(data2[i]))

    for i in range(10, len(data2), 13):
        A11.append(float(data2[i]))

    for i in range(11, len(data2), 13):
        A12.append(float(data2[i]))

    for i in range(12, len(data2), 13):
        A13.append(float(data2[i]))

vectors=[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13]
matrix=np.matrix(vectors)
matrix=matrix.transpose()
def myMLP(x_train, y_train, x_test, y_test, lr, size1, size2=0):

    if (size2 == 0):
        clf = neural_network.MLPClassifier(hidden_layer_sizes=(size1), learning_rate_init=lr)
    else:
        clf = neural_network.MLPClassifier(hidden_layer_sizes=(size1, size2), learning_rate_init=lr)

    clf.fit(x_train, y_train)
    predictii = clf.predict(x_test)

    k = 0

    for i in range(len(y_test)):
        if (predictii[i] == y_test[i]):
            k += 1

    print(k / len(y_test))


dataset = matrix
dataset = np.asarray(dataset)

etichete = dataset[:, 12]

date_train = dataset[:136, :]
etichete_train = etichete[:136]

date_test = dataset[136:, :]
etichete_test = etichete[136:]

dim_intrare = 12

learning_rate = [0.1, 0.01]

for my_lr in learning_rate:

    for i in range(2):

        for j in range(2):
            nr_strat = i + 1

            if nr_strat == 1:
                size = dim_intrare // (j + 1)
                myMLP(date_train, etichete_train, date_test, etichete_test, my_lr, size1=size)

            elif nr_strat == 2:
                size1 = dim_intrare // (j + 1)
                size2 = size1 // (j + 1)
                myMLP(date_train, etichete_train, date_test, etichete_test, my_lr, size1=size1, size2=size2)
