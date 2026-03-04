import cv2
import xml.etree.ElementTree as ET

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

def read_xml(path: str):
    tree = ET.parse(path)
    root = tree.getroot()
    samples = []

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    canvas = [width, height]

    for obj in root.findall("object"):
        name = obj.find("name").text
        box = obj.find("bndbox")
        xmin = int(box.find("xmin").text)
        ymin = int(box.find("ymin").text)
        xmax = int(box.find("xmax").text)
        ymax = int(box.find("ymax").text)
        sample = [name, xmin, ymin, xmax, ymax]
        samples.append(sample)

    return canvas, samples
