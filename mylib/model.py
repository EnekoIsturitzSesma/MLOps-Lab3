"""
Module providing basic image processing operations and a simple random classifier.
Includes functions for random class prediction, resizing, grayscale conversion,
and normalization.
"""

import random
import numpy as np


def predict_class(image):
    """
    Returns a random class label to simulate an image classification process.

    Parameters:
        image (PIL.Image): Input image (not used in the random prediction).

    Returns:
        str: A randomly selected label from ['dog', 'cat', 'horse', 'bear', 'pig'].
    """
    _ = image
    classes = ["dog", "cat", "horse", "bear", "pig"]
    return random.choice(classes)


def resize_image(image, size):
    """
    Resizes an image to the specified dimensions.

    Parameters:
        image (PIL.Image): Input image.
        size (tuple): Target size as (width, height).

    Returns:
        PIL.Image: The resized image.
    """
    return image.resize(size)


def convert_to_grayscale(image):
    """
    Converts an image to grayscale.

    Parameters:
        image (PIL.Image): Input image.

    Returns:
        PIL.Image: The grayscale image in mode 'L'.
    """
    return image.convert("L")


def normalize_image(image):
    """
    Normalizes image pixel values to the range [0, 1].

    Parameters:
        image (PIL.Image): Input image.

    Returns:
        numpy.ndarray: The normalized image as a NumPy array.
    """
    image_array = np.array(image) / 255.0
    return image_array
