import os
import xml.etree.ElementTree as ET

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sample_path = os.path.join(parent_path, "Sample")

# Get folder paths of images and their annotations
train_path = os.path.join(sample_path, "train")

xml_files = []
jpg_files = []
for file in os.listdir(train_path):
    if file.endswith(".xml"):
        xml_files.append(file)
    elif file.endswith(".jpg"):
        jpg_files.append(file)
    else:
        print("Invalid file")


