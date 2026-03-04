import math
import os
import cv2
import torch

import Helper
import Model
import random
import numpy as np

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sample_path = os.path.join(parent_path, "Sample")

# Get folder paths of images and their annotations
train_path = os.path.join(sample_path, "train")

xml_filepaths = []
jpg_filepaths = []
for file in os.listdir(train_path):
    if file.endswith(".xml"):
        xml_filepaths.append(os.path.join(train_path, file))
    elif file.endswith(".jpg"):
        jpg_filepaths.append(os.path.join(train_path, file))
    else:
        print("Invalid file")

classes = 2
grid_size = 7
boxes = 2
model = Model.YOLOModel(classes, grid_size, boxes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 100
batch_size = 16
input_size = 500
sample_size = len(xml_filepaths)
batch_num = math.ceil(sample_size / batch_size)

best_loss = float("inf")

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs} running...")
    pairs = list(zip(xml_filepaths, jpg_filepaths))
    random.shuffle(pairs)

    xml_filepaths, jpg_filepaths = zip(*pairs)
    xml_filepaths = list(xml_filepaths)
    jpg_filepaths = list(jpg_filepaths)

    epoch_loss = 0

    for i, batch_start in enumerate(range(0, sample_size, batch_size)):
        print(f"Batches {i + 1}/{batch_num} preparing...")
        batch_end = batch_start + batch_size
        if batch_end > sample_size: break

        xml_filepaths_batch = xml_filepaths[batch_start: batch_end]
        jpg_filepaths_batch = jpg_filepaths[batch_start: batch_end]

        # Batching data
        imgs = []
        annotations = []
        for xml_filepath in xml_filepaths_batch:
            samples = Helper.read_xml(xml_filepath)
            annotations.append(samples)

        for jpg_filepath in jpg_filepaths_batch:
            img = cv2.imread(jpg_filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (input_size, input_size),
                             interpolation=cv2.INTER_AREA)
            img = img.astype("float32") / 255.0
            imgs.append(img)

        imgs = np.stack(imgs, axis = 0)
        imgs = torch.from_numpy(imgs).float()
        imgs = imgs.permute(0, 3, 1, 2)
        imgs = imgs.to(device)

        # Fetching batch to model
        pred = model.forward(imgs)
        pred = pred.view(batch_size, grid_size, grid_size, boxes, 5 + classes)
        target = Helper.build_targets(annotations, grid_size, boxes, classes)
        target = target.to(device)
        loss = Helper.yolo_loss(pred, target)
        optimizer.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"Batch {i + 1}/{batch_num} loss = {loss.item():.4f}")

    epoch_loss /= batch_num
    if epoch_loss < best_loss:
        old_loss = best_loss
        best_loss = epoch_loss
        torch.save(model.state_dict(), "YOLO-vsth.pth")
        print(f"Best model saved: loss = {best_loss} < {old_loss}")












