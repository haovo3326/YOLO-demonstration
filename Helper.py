import xml.etree.ElementTree as ET
import torch

def build_targets(batch_annotations, S, B, C):
    bs = len(batch_annotations)
    target = torch.zeros((bs, S, S, B, 5 + C), dtype=torch.float32)

    for b in range(bs):
        for class_id, xc, yc, w, h in batch_annotations[b]:

            cx = xc * S
            cy = yc * S
            cell_x = min(int(cx), S - 1)
            cell_y = min(int(cy), S - 1)

            x_cell = cx - cell_x
            y_cell = cy - cell_y

            k = 0  # simple: always assign box 0

            target[b, cell_y, cell_x, k, 0] = 1.0
            target[b, cell_y, cell_x, k, 1] = x_cell
            target[b, cell_y, cell_x, k, 2] = y_cell
            target[b, cell_y, cell_x, k, 3] = w
            target[b, cell_y, cell_x, k, 4] = h
            target[b, cell_y, cell_x, k, 5 + int(class_id)] = 1.0

    return target

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

