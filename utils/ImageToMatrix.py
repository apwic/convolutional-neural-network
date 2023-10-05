import os
import random
from numpy import asarray
from PIL import Image

script_dir = os.path.dirname(__file__)

def ImageToMatrix(path):
    img = Image.open(path)
    img_array = asarray(img)
    return img_array

def getDataset(split_ratio = 0.8):
    # Define the base directory
    base_dir = 'dataset'

    # Define the categories and their respective paths
    categories = {
        'Train': ['Bears', 'Pandas'],
        'Test': ['Bears', 'Pandas']
    }

    # Assigning labels to each class
    labels = {
        'Bears': 0,
        'Pandas': 1
    }

    # This dictionary will hold our data
    data = {
        'Train': {},
        'Test': {}
    }

    # Navigate through each category and list files
    for dataset_type, classes in categories.items():
        for cls in classes:
            # Create the full path to the class directory
            class_dir = os.path.join(base_dir, dataset_type, cls)
            
            # List all files in the directory
            files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
            
            # Shuffle the files
            random.shuffle(files)
            
            # Pair each file with its label
            labeled_files = [(ImageToMatrix(file), labels[cls]) for file in files]
            
            # Store in the data dictionary
            data[dataset_type][cls] = labeled_files

    # Splitting data
    train_data = {}
    val_data = {}

    for cls, files in data['Train'].items():
        split_idx = int(len(files) * split_ratio)
        train_data[cls] = files[:split_idx]
        val_data[cls] = files[split_idx:]

    # The test data is already separated, so we can directly use it
    test_data = data['Test']

    return (train_data, val_data, test_data)