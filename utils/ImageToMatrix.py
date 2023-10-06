import os
import random
import numpy as np
from numpy import asarray
from PIL import Image

script_dir = os.path.dirname(__file__)
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

def ImageToMatrix(path):
    img = Image.open(path)
    img_array = asarray(img)
    return np.transpose(img_array, (2, 1, 0))

def getTrainDataset(split_ratio = 0.8):
    all_train_data = []

    # Collect training data
    for cls in categories['Train']:
        # Create the full path to the class directory
        class_dir = os.path.join(base_dir, 'Train', cls)
        
        # List all files in the directory
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        
        # Pair each file with its label
        labeled_files = [(ImageToMatrix(file), labels[cls]) for file in files]
        
        # Extend the all_train_data list
        all_train_data.extend(labeled_files)

    # Shuffle the combined training data
    random.shuffle(all_train_data)

    # Splitting data (let's say you want to split the training data into 80% train and 20% validation)
    split_ratio = 0.8
    split_idx = int(len(all_train_data) * split_ratio)
    
    train_data_X = np.array([item[0] for item in all_train_data[:split_idx]])
    train_data_y = np.array([item[1] for item in all_train_data[:split_idx]])
    val_data_X = np.array([item[0] for item in all_train_data[split_idx:]])
    val_data_y = np.array([item[1] for item in all_train_data[split_idx:]])

    return train_data_X, train_data_y, val_data_X, val_data_y

def getTestDataset():
    # Collect test data
    test_data = []
    for cls in categories['Test']:
        # Create the full path to the class directory
        class_dir = os.path.join(base_dir, 'Test', cls)
        
        # List all files in the directory
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        
        # Pair each file with its label
        labeled_files = [(ImageToMatrix(file), labels[cls]) for file in files]
        
        # Extend the test_data list
        test_data.extend(labeled_files)

    test_data_X = np.array([item[0] for item in test_data])
    test_data_y = np.array([item[1] for item in test_data])

    return test_data_X, test_data_y

if __name__ == '__main__':
    # TODO: gimana caranya ini masuk cuk ke sekwensial kt
    train_data_X, train_data_y, val_data_X, val_data_y = getTrainDataset()

    print(len(train_data_X[0]))