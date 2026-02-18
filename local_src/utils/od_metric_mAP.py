# ======================================================================
# METRICS IMPLEMENTATION (GENERIC mAP PER IMAGE)
# ======================================================================
import torch


def box_iou(box1, box2):
    """
    Computes IoU between two bounding boxes.

    Boxes must be in XYXY format.
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter
    if union == 0:
        return 0

    return inter / union


def compute_ap(recall, precision):
    """
    Computes Average Precision using VOC 11-point interpolation.
    """

    recall = torch.tensor(recall)
    precision = torch.tensor(precision)

    ap = 0.0

    for t in torch.arange(0, 1.1, 0.1):
        p = precision[recall >= t]
        ap += torch.max(p) if len(p) > 0 else 0

    return ap / 11


def evaluate_image_map(gt, pred, score_threshold=0.5, iou_threshold=0.5):
    """
    Computes mAP for a single image.

    Steps:
        1. Filter predictions by score
        2. Split GT / predictions by class
        3. Match predictions to GT via IoU
        4. Compute TP / FP
        5. Build Precision-Recall curve
        6. Compute AP per class
        7. Average across classes → mAP
    """

    gt_boxes = gt["boxes"]
    gt_labels = gt["labels"]

    pred_boxes = pred["boxes"]
    pred_labels = pred["labels"]
    pred_scores = pred["scores"]

    # --------------------------------------------------
    # Score filtering
    # --------------------------------------------------
    keep = pred_scores >= score_threshold
    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]

    aps = []

    # Classes appearing in either GT or prediction
    unique_labels = torch.unique(torch.cat([gt_labels, pred_labels]))

    for label in unique_labels:

        gt_mask = gt_labels == label
        pred_mask = pred_labels == label

        gt_l = gt_boxes[gt_mask]
        pred_l = pred_boxes[pred_mask]
        scores_l = pred_scores[pred_mask]

        if len(gt_l) == 0:
            continue

        # --------------------------------------------------
        # Sort predictions by descending confidence
        # --------------------------------------------------
        if len(scores_l) > 0:
            order = torch.argsort(scores_l, descending=True)
            pred_l = pred_l[order]
            scores_l = scores_l[order]

        tp = torch.zeros(len(pred_l))
        fp = torch.zeros(len(pred_l))

        matched = torch.zeros(len(gt_l))

        # --------------------------------------------------
        # Greedy matching prediction → GT
        # --------------------------------------------------
        for i, pbox in enumerate(pred_l):

            ious = torch.tensor([box_iou(pbox, gtbox) for gtbox in gt_l])

            if len(ious) == 0:
                fp[i] = 1
                continue

            best_iou, best_idx = torch.max(ious, dim=0)

            if best_iou >= iou_threshold and not matched[best_idx]:
                tp[i] = 1
                matched[best_idx] = 1
            else:
                fp[i] = 1

        if len(tp) == 0:
            continue

        # --------------------------------------------------
        # Build Precision / Recall curves
        # --------------------------------------------------
        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)

        recall = tp_cum / len(gt_l)
        precision = tp_cum / (tp_cum + fp_cum + 1e-6)

        ap = compute_ap(recall, precision)
        aps.append(ap)

    if len(aps) == 0:
        return 0.0

    return torch.mean(torch.tensor(aps)).item()