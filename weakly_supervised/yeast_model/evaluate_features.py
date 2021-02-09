import umap
import numpy as np
import umap.plot

data = np.genfromtxt(fname="test/yeast_features_conv1_1.txt", delimiter="\t")

print(data.shape)