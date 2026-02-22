"""MNIST dataset loader with download and preprocessing utilities."""

import os
import gzip
import pickle
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np

# Configuration
URL_BASE = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
FILES = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

DATASET_DIR = Path(__file__).parent
SAVE_FILE = DATASET_DIR / "mnist.pkl"
IMG_SIZE = 784


def _download_file(filename: str) -> None:
    """Download a single MNIST file if it doesn't exist."""
    filepath = DATASET_DIR / filename
    
    if filepath.exists():
        return
    
    print(f"Downloading {filename}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    request = urllib.request.Request(URL_BASE + filename, headers=headers)
    
    with urllib.request.urlopen(request) as response:
        filepath.write_bytes(response.read())
    
    print("Done")


def _load_labels(filename: str) -> np.ndarray:
    """Load labels from gzipped file."""
    filepath = DATASET_DIR / filename
    print(f"Loading {filename}...")
    
    with gzip.open(filepath, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    
    print("Done")
    return labels


def _load_images(filename: str) -> np.ndarray:
    """Load images from gzipped file."""
    filepath = DATASET_DIR / filename
    print(f"Loading {filename}...")
    
    with gzip.open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    
    data = data.reshape(-1, IMG_SIZE)
    print("Done")
    return data


def _to_one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert labels to one-hot encoding."""
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


def init_mnist() -> None:
    """Download and initialize MNIST dataset."""
    # Download all files
    for filename in FILES.values():
        _download_file(filename)
    
    # Load and save as pickle
    dataset = {
        'train_img': _load_images(FILES['train_img']),
        'train_label': _load_labels(FILES['train_label']),
        'test_img': _load_images(FILES['test_img']),
        'test_label': _load_labels(FILES['test_label'])
    }
    
    print("Creating pickle file...")
    with open(SAVE_FILE, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done!")


def load_mnist(
    normalize: bool = True,
    flatten: bool = True,
    one_hot_label: bool = False
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load MNIST dataset.
    
    Parameters
    ----------
    normalize : bool, default=True
        Normalize pixel values to [0.0, 1.0]
    flatten : bool, default=True
        Flatten images to 1D array
    one_hot_label : bool, default=False
        Convert labels to one-hot encoding
    
    Returns
    -------
    tuple
        ((train_images, train_labels), (test_images, test_labels))
    """
    if not SAVE_FILE.exists():
        init_mnist()
    
    with open(SAVE_FILE, 'rb') as f:
        dataset = pickle.load(f)
    
    # Normalize images
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32) / 255.0
    
    # Convert to one-hot labels
    if one_hot_label:
        dataset['train_label'] = _to_one_hot(dataset['train_label'])
        dataset['test_label'] = _to_one_hot(dataset['test_label'])
    
    # Reshape images
    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
    
    return (
        (dataset['train_img'], dataset['train_label']),
        (dataset['test_img'], dataset['test_label'])
    )


if __name__ == '__main__':
    init_mnist()