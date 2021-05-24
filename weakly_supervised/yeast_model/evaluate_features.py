"""
Generate a UMAP visualization of the embeddings of each layer of the pretrained model.
"""

import umap
import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib.cm import tab20
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.lines import Line2D

# layer names
file_list = ['conv1_1',
            'conv2_1',
            'conv3_1',
            'conv4_1',
            'conv5_1',
            'dense1',
            'dense2',
            'rand_conv3_1']

rand = ['rand_conv3_1']

# Two for loop headers for if only the random layer is evaluated, or if every layer is.
# One of the two has been commented out.

# for i in range(len(file_list)):
    # f = file_list[i]
for i in range(1):
    f = rand[i]
    print(f)
    data = pandas.read_csv('test/yeast_features_'+f+'.txt', sep = '\t', header=None)

    data = data.set_index(data.columns[0])

    N = len(data.columns)
    K = 21
    num_rows = data.shape[0]

    data.drop(data.columns[len(data.columns)-1], axis=1, inplace=True)
    
    print(data)

    # Perform UMAP transformation
    fit = umap.UMAP()
    u = fit.fit_transform(data)

    print(u)

    # Generate class index for legend creation and colour coding
    print("Getting classes:")
    index_values = data.index
    class_series = index_values.str.split("_").str[0]
    class_arr = class_series.to_numpy()
    class_list = list(set(class_arr))
    class_list.sort()
    print("classes:", class_list)
    num_classes = len(class_list)

    # Create colour palette and legend
    numeric_class_arr = []
    for item in class_arr:
        numeric_class_arr.append(class_list.index(item))
    colors = [tab20(float(i)/num_classes) for i in numeric_class_arr]

    legend_elements = []
    for j in range(num_classes):
        curr_class = class_list[j]
        curr_colour = tab20(float(j)/num_classes)
        legend_elements.append(Line2D([0], [0], color=curr_colour, lw=2, label=curr_class))

    # Generate plot
    x = u[:,0]
    y = u[:,1]

    fig, ax = plt.subplots()
    ax.scatter(x, y, color=colors, marker=',', s=0.1)
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
    plt.savefig('test/z_yeast_features_'+f+'_plot.png', )
