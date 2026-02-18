import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_image(img, label_names, gt=None, pred=None, title="", threshold=0.5, ):
    """
    Draws GT and predictions on image.

    Green  → Ground Truth
    Red    → Prediction
    """

    img = img.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    # --------------------------------------------------
    # Draw Ground Truth boxes
    # --------------------------------------------------
    if gt is not None:
        for box, label in zip(gt["boxes"], gt["labels"]):
            x1, y1, x2, y2 = box.tolist()
            name = label_names.get(int(label), str(int(label)))

            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

            ax.text(x1, y1 - 3, f"GT: {name}", color='lime', fontsize=9,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1))

    # --------------------------------------------------
    # Draw Predictions
    # --------------------------------------------------
    if pred is not None:
        for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):

            if score < threshold:
                continue

            x1, y1, x2, y2 = box.tolist()
            name = label_names.get(int(label), str(int(label)))

            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

            ax.text(x1, y2 + 10, f"Pred: {name} ({score:.2f})",
                    color='red', fontsize=9,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1))

    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.show()