import umap
import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from sklearn.neighbors import KNeighborsClassifier

file_list = ['conv1_1',
            'conv2_1',
            'conv3_1',
            'conv4_1',
            'conv5_1',
            'dense1',
            'dense2',
            'rand_conv3_1']

destf = open('knn_acc.txt', 'a')

for i in range(len(file_list)):
    if i < 7:
        continue

    f = file_list[i]
    print(f)
    data = pandas.read_csv('test/yeast_features_'+f+'.txt', sep = '\t', header=None)

    N = len(data.columns)
    K = 11
    num_rows = data.shape[0]

    for j in range(num_rows):
        data.at[j, 0] = data.at[j, 0].split('_')[0]
    
    X = data.iloc[:,1:96].to_numpy()
    Y = data.iloc[:,0].to_numpy()  

    print(X)
    print(Y)

    knn = KNeighborsClassifier(n_neighbors = K)
    knn.fit(X, Y)

    print('scoring')
    accuracy = knn.score(X, Y)
    print(accuracy)

    destf.write(f + '\t' + str(accuracy) + '\n')

destf.close()