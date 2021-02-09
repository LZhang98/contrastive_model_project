import umap
import numpy as np
import pandas
import umap.plot

file_list = ['conv1_1.txt',
            'conv2_1.txt',
            'conv3_1.txt',
            'conv4_1.txt',
            'conv5_1.txt',
            'dense1.txt',
            'dense2.txt']

data = pandas.read_csv('test/yeast_features/'+file_list[0], sep = '\t', )

print(data.shape)

print(data[0:5,:])