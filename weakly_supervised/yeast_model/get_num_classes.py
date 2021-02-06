# TEMP SCRIPT TO GET NUM CLASSES

# INCORPORATE INTO TRAIN.PY LATER

import opts as opt
from dataset import Dataset

ds = Dataset()
num_classes = ds.add_dataset(opt.data_path)

print("Number of classes:", num_classes)

f = open("num_classes.txt", "w")

f.write(str(num_classes))

f.close()