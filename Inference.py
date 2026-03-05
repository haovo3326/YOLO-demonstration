import torch
import Model
import os
import Helper
import cv2
import numpy as np
import torch.nn.functional as F


def compute_iou(box1, box2):
    x1_i, y1_i, x2_i, y2_i = box1
    x1_j, y1_j, x2_j, y2_j = box2

    inter_x1 = max(x1_i, x1_j)
    inter_y1 = max(y1_i, y1_j)
    inter_x2 = min(x2_i, x2_j)
    inter_y2 = min(y2_i, y2_j)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_i = (x2_i - x1_i) * (y2_i - y1_i)
    area_j = (x2_j - x1_j) * (y2_j - y1_j)

    union = area_i + area_j - inter_area
    if union == 0:
        return 0

    return inter_area / union

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sample_path = os.path.join(parent_path, "Sample")
test_path = os.path.join(sample_path, "test")

xml_filepaths = []
jpg_filepaths = []
for file in os.listdir(test_path):
    if file.endswith(".xml"):
        xml_filepaths.append(os.path.join(test_path, file))
    elif file.endswith(".jpg"):
        jpg_filepaths.append(os.path.join(test_path, file))

xml_filepaths.sort()
jpg_filepaths.sort()

MAP = {1: "Dog", 0: "Cat"}

classes = 2
grid_size = 7
boxes = 2
input_size = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model.YOLOModel(classes, grid_size, boxes).to(device)
model.load_state_dict(torch.load("YOLO-vsth.pth", map_location=device))
model.eval()

jpg_filepath = jpg_filepaths[0]
xml_filepath = xml_filepaths[0]

# Read + resize (keep a display copy in BGR for drawing)
image0 = cv2.imread(jpg_filepath)  # BGR
disp = cv2.resize(image0, (input_size, input_size), interpolation=cv2.INTER_AREA)  # BGR for drawing

# Prepare model input (RGB + normalize)
img = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0
imgs = torch.from_numpy(img).unsqueeze(0)          # (1,H,W,C)
imgs = imgs.permute(0, 3, 1, 2).contiguous()       # (1,C,H,W)
imgs = imgs.to(device)

with torch.no_grad():
    preds = model(imgs)            # prefer model(imgs)
    pred = preds[0]                # (S,S,B*(5+C)) or similar

H, W = disp.shape[:2]
cell_w = W / grid_size
cell_h = H / grid_size
bbox_len = 5 + classes
bboxes = []
for i in range(grid_size):
    for j in range(grid_size):
        out = pred[i, j]

        for b in range(boxes):
            bbox_vector = out[b * bbox_len: (b + 1) * bbox_len]

            obj = float(bbox_vector[0])
            x_cell = float(bbox_vector[1])   # expected in [0,1] within cell
            y_cell = float(bbox_vector[2])   # expected in [0,1] within cell
            w = float(bbox_vector[3])        # expected in [0,1] relative to image
            h = float(bbox_vector[4])        # expected in [0,1] relative to image
            cls_logits = bbox_vector[5:5 + classes]
            cls_probs = F.softmax(cls_logits, dim=0)

            cls_id = int(torch.argmax(cls_probs).item())
            cls_prob = cls_probs[cls_id].item()
            label = MAP.get(cls_id, str(cls_id))

            # convert to pixel sizes
            bw = w * W
            bh = h * H

            # center in global pixels
            cx = (j + x_cell) * cell_w
            cy = (i + y_cell) * cell_h

            # xyxy box
            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            x2 = int(cx + bw / 2)
            y2 = int(cy + bh / 2)

            # clip
            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(0, min(W - 1, x2))
            y2 = max(0, min(H - 1, y2))

            bbox = [x1, y1, x2, y2, obj, cls_id, cls_prob]
            bboxes.append(bbox)

            # # draw
            # cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(
            #     disp, f"{label} {obj:.2f}",
            #     (x1, max(0, y1 - 6)),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            # )

# filter bounding boxes
confidence_threshold = 0.25
for bbox in bboxes:
    x1, y1, x2, y2, obj, cls_id, cls_prob = bbox
    score = obj * cls_prob
    if score < confidence_threshold:
        bboxes.remove(bbox)
        continue


dog_bboxes = []
cat_bboxes = []

for bbox in bboxes:
    _, _, _, _, _, cls_id, _ = bbox
    if cls_id == 1:
        dog_bboxes.append(bbox)
    elif cls_id == 0:
        cat_bboxes.append(bbox)


iou_threshold = 0.5

filtered_dog_bboxes = []
for i in range(len(dog_bboxes)):
    keep = True
    x1_i, y1_i, x2_i, y2_i, obj_i, _, cls_prob_i = dog_bboxes[i]
    score_i = obj_i * cls_prob_i

    for j in range(len(dog_bboxes)):
        if i == j:
            continue

        x1_j, y1_j, x2_j, y2_j, obj_j, _, cls_prob_j = dog_bboxes[j]
        score_j = obj_j * cls_prob_j

        iou = compute_iou(
            (x1_i, y1_i, x2_i, y2_i),
            (x1_j, y1_j, x2_j, y2_j)
        )

        if iou > iou_threshold and score_j > score_i:
            keep = False
            break

    if keep:
        filtered_dog_bboxes.append(dog_bboxes[i])

filtered_cat_bboxes = []

for i in range(len(cat_bboxes)):
    keep = True
    x1_i, y1_i, x2_i, y2_i, obj_i, _, cls_prob_i = cat_bboxes[i]
    score_i = obj_i * cls_prob_i

    for j in range(len(cat_bboxes)):
        if i == j:
            continue

        x1_j, y1_j, x2_j, y2_j, obj_j, _, cls_prob_j = cat_bboxes[j]
        score_j = obj_j * cls_prob_j

        iou = compute_iou(
            (x1_i, y1_i, x2_i, y2_i),
            (x1_j, y1_j, x2_j, y2_j)
        )

        if iou > iou_threshold and score_j > score_i:
            keep = False
            break

    if keep:
        filtered_cat_bboxes.append(cat_bboxes[i])

final_bboxes = filtered_dog_bboxes + filtered_cat_bboxes
for bbox in final_bboxes:
    x1, y1, x2, y2, obj, cls_id, cls_prob = bbox
    label = MAP[cls_id]
    score = obj * cls_prob

    cv2.rectangle(disp, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(
        disp,
        f"{label} {score:.2f}",
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,255,0),
        2
    )

cv2.imshow("Predictions", disp)
cv2.waitKey(0)
cv2.destroyAllWindows()

annotation = Helper.read_xml(xml_filepath)