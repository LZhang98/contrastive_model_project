
import os
import numpy as np
from PIL import Image
import skimage.exposure
import skimage.filters
import skimage.morphology

class Dataset(object):
    '''We store information about the dataset as a class.'''
    def __init__(self):
        self.image_ids = []
        self.image_info = []

    '''Add an individual image to the class. Called iteratively by add_dataset().
    For memory purposes, this class loads only the paths of each image, and returns the 
    actual images as needed.'''
    def add_image(self, image_id, path, name, class_index):
        image_info = {
            "id": image_id,            # Unique integer representation of the image
            "path": path,              # Path where the image is stored
            "name": name,               # Name of the image
            "class": class_index             # Index for one-hot label generation
        }
        self.image_info.append(image_info)

    """Function for adding a directory containing subdirectories of images, grouped by protein.
    
    Returns the number of folders in the directory. (Final folder_count)"""
    def add_dataset(self, root_dir):
        i = 0           # Used to assign a unique integer index to each image
        folder_count = 0 # Used for label

        # Iterate over every image in the dataset
        for currdir in os.listdir(root_dir):
            print ("Adding " + currdir)
            # Get all the GFP images only
            all_files = os.listdir(root_dir + currdir)
            image_names = []
            for file in all_files:
                if ("_green" in file):
                    image_names.append(file)

            # Add images
            for j in range(len(image_names)):
                self.add_image(
                    image_id=i,
                    path=root_dir + currdir + "/",
                    name=image_names[j],
                    class_index=folder_count
                    )
                i += 1
            
            folder_count += 1
        
        return folder_count

    '''Load and return the image indexed by the integer given by image_id.
       Also return its label
    '''
    def load_image(self, image_id):
        # Get the path of the image
        path = self.image_info[image_id]['path']
        
        # Get the one-hot label of the image
        label = self.image_info[image_id]['label']

        # Load and return all three channels for the Human Protein Atlas
        antibodyname = self.image_info[image_id]['name']
        nucleusname = antibodyname.replace("green", "blue")
        microtubulename = antibodyname.replace("green", "red")

        antibody = np.array(Image.open(path + antibodyname))
        nucleus = np.array(Image.open(path + nucleusname))
        microtubule = np.array(Image.open(path + microtubulename))

        return antibody, nucleus, microtubule, label

    '''Load and return the image indexed by the integer given by image_id; also returns the
    name of the image, just for debugging purposes'''
    def load_image_with_label(self, image_id):
        # Get the path of the image
        path = self.image_info[image_id]['path']

        # Load and return all three channels for the Human Protein Atlas, as well as the name of the image
        antibodyname = self.image_info[image_id]['name']
        nucleusname = antibodyname.replace("green", "blue")
        microtubulename = antibodyname.replace("green", "red")
        label = antibodyname.rsplit("_", -1)[0]

        antibody = np.array(Image.open(path + antibodyname))
        nucleus = np.array(Image.open(path + nucleusname))
        microtubule = np.array(Image.open(path + microtubulename))

        return antibody, nucleus, microtubule, label

    '''Sample a pair for the given image by drawing with equal probability from the folder'''
    def sample_pair_equally(self, image_id):
        path = self.image_info[image_id]['path']
        name = self.image_info[image_id]['name']

        # Get all other images in the same path as the image given by image_id
        all_files = os.listdir(path)
        image_names = []
        for file in all_files:
            if ("_green" in file and file != name):
                image_names.append(file)

        # If the directory has more than one image, sample a random image and return it
        if len(all_files) > 1:
            sampled_image = np.random.choice(image_names)

            antibodyname = sampled_image
            nucleusname = antibodyname.replace("green", "blue")
            microtubulename = antibodyname.replace("green", "red")

            antibody = np.array(Image.open(path + antibodyname))
            nucleus = np.array(Image.open(path + nucleusname))
            microtubule = np.array(Image.open(path + microtubulename))

            return antibody, nucleus, microtubule
        else:
            raise ValueError("Directory " + path + " has only one image.")

    '''Prepares the dataset file for use. Uses num_classes from add_dataset to generate one-hot vectors'''
    def prepare(self, num_classes):
        # Build (or rebuild) everything else from the info dicts.
        self.num_images = len(self.image_info)
        self.image_ids = np.arange(self.num_images)

        # Generate one hot vectors. Use num_classes for vector length
        for image in self.image_info:
            label = np.zeros(num_classes)
            label[image['class']] = 1
            image['label'] = label