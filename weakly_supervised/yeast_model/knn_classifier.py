"""
Perform a KNN Classification on the representations in each layer of the pretrained model.
"""

import umap
import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from sklearn.neighbors import KNeighborsClassifier

# All layer names

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
    f = file_list[i]
    print(f)
    data = pandas.read_csv('test/yeast_features_'+f+'.txt', sep = '\t', header=None)

    N = len(data.columns)
    K = 11
    num_rows = data.shape[0]

    # Split data frame by features (X) and labels (Y)
    for j in range(num_rows):
        data.at[j, 0] = data.at[j, 0].split('_')[0]
    
    X = data.iloc[:,1:96].to_numpy()
    Y = data.iloc[:,0].to_numpy()  

    # Get the sample size of each class
    class_counts = np.unique(Y, return_counts=True)

    # Fit a kNN classifier
    knn = KNeighborsClassifier(n_neighbors = K)
    knn.fit(X, Y)

    # print('scoring')
    # accuracy = knn.score(X, Y)
    # print(accuracy)

    # destf.write(f + '\t' + str(accuracy) + '\n')

    # For writing data
    model_class_accuracy = open('class_knn_'+f+'.txt', 'w')

    # Perform class-by-class KNN
    start = 0
    for j in range(17):
        curr_class = class_counts[0][j]
        print('scoring '+curr_class)
        end = start + class_counts[1][j]
        X_subset = X[start:end,:]
        Y_subset = Y[start:end]
        accuracy = knn.score(X_subset, Y_subset)
        print(accuracy)
        model_class_accuracy.write(curr_class + '\t' + str(accuracy) + '\n')
        start = end

destf.close()
model_class_accuracy.close()