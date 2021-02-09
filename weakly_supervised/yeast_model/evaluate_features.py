import umap
import numpy as np
import pandas
import matplotlib.pyplot as plt

file_list = ['conv1_1',
            'conv2_1',
            'conv3_1',
            'conv4_1',
            'conv5_1',
            'dense1',
            'dense2']


for i in range(len(file_list)):
    f = file_list[i]
    print(f)
    data = pandas.read_csv('test/yeast_features_'+f+'.txt', sep = '\t', )

    data = data.set_index(data.columns[0])

    data = data.drop(data.columns[len(data.columns)-1], axis=1, inplace=True)

    print(data)

    fit = umap.UMAP()
    u = fit.fit_transform(data)

    print(u)

    x = u[:,0]
    y = u[:,1]
    plt.figure(i)
    plt.scatter(x, y, marker='.')
    plt.savefig('test/yeast_features_'+f+'_plot.png')
