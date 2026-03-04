import cv2

def resize(img, new_size):
    """
    Resize an image to a new size
    :param img: numpy.ndarray
                Input image
    :param new_size: tuple
                    (width, height)
    :return: numpy.ndarray
            Resized image
    """
    return cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

def normalize(img):
    """
    Normalize image value from 0 -> 255 to 0 -> 1
    :param img: numpy.ndarray
                Input image
    :return: numpy.ndarray
            New image
    """
    return img / 255.0
