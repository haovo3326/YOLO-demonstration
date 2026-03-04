import xml.etree.ElementTree as ET

def read_xml(path: str):
    tree = ET.parse(path)
    root = tree.getroot()
    samples = []

    size = root.find("size")
    img_width = float(size.find("width").text)
    img_height = float(size.find("height").text)

    for obj in root.findall("object"):
        class_id = int(obj.find("name").text)
        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        sample = [class_id, x_center, y_center, width, height]
        samples.append(sample)

    return samples

