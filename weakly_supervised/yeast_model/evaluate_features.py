import umap
import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib.cm import tab20
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.lines import Line2D

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
    data = pandas.read_csv('test/yeast_features_'+f+'.txt', sep = '\t', header=None)

    data = data.set_index(data.columns[0])

    N = len(data.columns)
    K = 21
    num_rows = data.shape[0]

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
    class_list.sort()
    print("classes:", class_list)
    num_classes = len(class_list)

    numeric_class_arr = []
    for item in class_arr:
        numeric_class_arr.append(class_list.index(item))

    colors = [tab20(float(i)/num_classes) for i in numeric_class_arr]

    legend_elements = []
    for j in range(num_classes):
        curr_class = class_list[j]
        curr_colour = tab20(float(j)/num_classes)
        legend_elements.append(Line2D([0], [0], color=curr_colour, lw=2, label=curr_class))

    x = u[:,0]
    y = u[:,1]
    # plt.figure(i)
    # plt.scatter(x, y, color=colors, marker=',', s=0.1)
    # plt.savefig('test/yeast_features_'+f+'_plot.png')

    fig, ax = plt.subplots(i)
    ax.scatter(x, y, color=colors, marker=',', s=0.1)
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
    plt.savefig('test/z_yeast_features_'+f+'_plot.png')
