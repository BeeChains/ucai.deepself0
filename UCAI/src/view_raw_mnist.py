import os
import struct
import numpy as np
import matplotlib.pyplot as plt

def read_mnist_images(filename):
    """Read raw MNIST .ubyte image file."""
    with open(filename, 'rb') as f:
        # Read header
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, "Invalid magic number for images"
        
        # Read image data
        data = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
    return data

def read_mnist_labels(filename):
    """Read raw MNIST .ubyte label file."""
    with open(filename, 'rb') as f:
        # Read header
        magic, num_labels = struct.unpack(">II", f.read(8))
        assert magic == 2049, "Invalid magic number for labels"
        
        # Read label data
        data = np.fromfile(f, dtype=np.uint8)
    return data

def display_sample(image, label):
    """Display a single MNIST image with its label."""
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Paths to raw files (adjust if your data folder differs)
    data_dir = "../data/MNIST/raw/"
    train_images_file = os.path.join(data_dir, "train-images-idx3-ubyte")
    train_labels_file = os.path.join(data_dir, "train-labels-idx1-ubyte")
    
    # Read raw data
    images = read_mnist_images(train_images_file)
    labels = read_mnist_labels(train_labels_file)
    
    # Display first 5 samples
    for i in range(5):
        print(f"Image {i} - Raw shape: {images[i].shape}, Label: {labels[i]}")
        display_sample(images[i], labels[i])