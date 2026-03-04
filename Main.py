import os
import xml.etree.ElementTree as ET
import Helper

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sample_path = os.path.join(parent_path, "Sample")

# Get folder paths of images and their annotations
train_path = os.path.join(sample_path, "train")

xml_files = []
jpg_files = []
for file in os.listdir(train_path):
    if file.endswith(".xml"):
        xml_files.append(os.path.join(train_path, file))
    elif file.endswith(".jpg"):
        jpg_files.append(os.path.join(train_path, file))
    else:
        print("Invalid file")

classes = 2
grid_size = 7
boxes = 3


