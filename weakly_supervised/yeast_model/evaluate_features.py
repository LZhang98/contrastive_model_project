import umap
import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib.cm import viridis

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

    data.drop(data.columns[len(data.columns)-1], axis=1, inplace=True)

    print(data)

    fit = umap.UMAP()
    u = fit.fit_transform(data)

    print(u)

    print("Getting classes:")
    index_values = data.index
    class_series = index_values.str.split("_").str[0]
    class_arr = class_series.to_numpy()
    class_list = list(set(class_arr))
    print("classes:", class_list)
    num_classes = len(class_list)

    numeric_class_arr = []
    for item in class_arr:
        numeric_class_arr.append(class_list.index(item))

    colors = [viridis(float(i)/num_classes) for i in numeric_class_arr]

    x = u[:,0]
    y = u[:,1]
    plt.figure(i)
    plt.scatter(x, y, color=colors, marker='.')
    plt.savefig('test/yeast_features_'+f+'_plot.png')

