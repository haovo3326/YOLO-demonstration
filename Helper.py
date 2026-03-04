import xml.etree.ElementTree as ET
import torch

def yolo_loss(pred, target, lambda_coord=5.0, lambda_noobj=0.5, eps=1e-9):
    """
    pred, target: (N, S, S, B, 5 + C)

    Layout (matches your build_targets):
      [..., 0]   = obj (0/1)
      [..., 1:5] = x, y, w, h
      [..., 5:]  = class one-hot
    """
    assert pred.dim() == 5 and target.dim() == 5, "pred/target must be (N,S,S,B,5+C)"
    N, S, _, B, D = pred.shape
    C = D - 5

    # Split tensors
    pred_obj  = pred[..., 0]
    pred_xywh = pred[..., 1:5]
    pred_cls  = pred[..., 5:]

    targ_obj  = target[..., 0]
    targ_xywh = target[..., 1:5]
    targ_cls  = target[..., 5:]

    # Masks
    obj_mask   = targ_obj == 1
    noobj_mask = targ_obj == 0

    # ---- Build absolute (image-normalized) xywh for IoU ----
    # pred/targ x,y are cell-relative in [0,1), convert to image-relative by adding cell offset / S
    # w,h already image-normalized in your read_xml/build_targets. :contentReference[oaicite:2]{index=2}
    gy = torch.arange(S, device=pred.device).view(1, S, 1, 1).expand(N, S, S, B)
    gx = torch.arange(S, device=pred.device).view(1, 1, S, 1).expand(N, S, S, B)

    pred_x = (gx + pred_xywh[..., 0]) / S
    pred_y = (gy + pred_xywh[..., 1]) / S
    pred_w = pred_xywh[..., 2].clamp(min=eps)
    pred_h = pred_xywh[..., 3].clamp(min=eps)

    targ_x = (gx + targ_xywh[..., 0]) / S
    targ_y = (gy + targ_xywh[..., 1]) / S
    targ_w = targ_xywh[..., 2].clamp(min=eps)
    targ_h = targ_xywh[..., 3].clamp(min=eps)

    def xywh_to_xyxy(x, y, w, h):
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return x1, y1, x2, y2

    px1, py1, px2, py2 = xywh_to_xyxy(pred_x, pred_y, pred_w, pred_h)
    tx1, ty1, tx2, ty2 = xywh_to_xyxy(targ_x, targ_y, targ_w, targ_h)

    inter_x1 = torch.maximum(px1, tx1)
    inter_y1 = torch.maximum(py1, ty1)
    inter_x2 = torch.minimum(px2, tx2)
    inter_y2 = torch.minimum(py2, ty2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    pred_area = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    targ_area = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union = pred_area + targ_area - inter_area
    iou = inter_area / (union + eps)

    # ---- Confidence (obj) target (YOLOv1 style) ----
    # for obj boxes: target confidence = IoU
    # for noobj boxes: target confidence = 0
    conf_target = torch.zeros_like(pred_obj)
    conf_target[obj_mask] = iou[obj_mask].detach()

    # ---- Losses (MSE like YOLOv1) ----
    # 1) Coordinate loss (only where obj=1)
    coord_loss = torch.tensor(0.0, device=pred.device)
    if obj_mask.any():
        # x,y
        xy_loss = (pred_xywh[..., 0:2][obj_mask] - targ_xywh[..., 0:2][obj_mask]).pow(2).sum()

        # sqrt(w), sqrt(h)
        pred_wh = pred_xywh[..., 2:4][obj_mask].clamp(min=eps)
        targ_wh = targ_xywh[..., 2:4][obj_mask].clamp(min=eps)
        wh_loss = (torch.sqrt(pred_wh) - torch.sqrt(targ_wh)).pow(2).sum()

        coord_loss = lambda_coord * (xy_loss + wh_loss)

    # 2) Objectness loss (obj boxes -> IoU)
    obj_loss = torch.tensor(0.0, device=pred.device)
    if obj_mask.any():
        obj_loss = (pred_obj[obj_mask] - conf_target[obj_mask]).pow(2).sum()

    # 3) No-object loss (noobj boxes -> 0)
    noobj_loss = torch.tensor(0.0, device=pred.device)
    if noobj_mask.any():
        noobj_loss = lambda_noobj * (pred_obj[noobj_mask] - 0.0).pow(2).sum()

    # 4) Class loss (only where obj=1)
    cls_loss = torch.tensor(0.0, device=pred.device)
    if obj_mask.any() and C > 0:
        cls_loss = (pred_cls[obj_mask] - targ_cls[obj_mask]).pow(2).sum()

    total = coord_loss + obj_loss + noobj_loss + cls_loss
    return total / max(N, 1)


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

