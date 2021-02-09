import umap
import numpy as np
import umap.plot

file_list = ['conv1_1.txt',
            'conv2_1.txt',
            'conv3_1.txt',
            'conv4_1.txt',
            'conv5_1.txt',
            'dense1.txt',
            'dense2.txt']

data = np.genfromtxt(fname="test/yeast_features_conv1_1.txt", delimiter="\t")

print(data.shape)

row_names = data[:,0]
print(row_names[0:5])