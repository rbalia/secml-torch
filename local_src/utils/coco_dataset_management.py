# ======================================================================
# DATASET DOWNLOAD UTILITIES
# ======================================================================
# These functions guarantee that COCO is downloaded automatically.
# They are intentionally separated for clarity and reusability.
# ======================================================================
import os
import urllib
import zipfile

import torch
from torchvision.datasets import CocoDetection

DATA_ROOT = "./data/coco"

def download_file(url, dest):
    """
    Download file only if it does not exist already.

    Parameters
    ----------
    url : str
        Remote file URL
    dest : str
        Local destination path
    """
    if os.path.exists(dest):
        return

    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, dest)


def unzip_file(zip_path, extract_to):
    """
    Extract zip only if folder does not already exist.
    """
    if os.path.exists(extract_to):
        return

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(extract_to))


def load_coco_categories(dataset):
    """
    Builds mapping: category_id → textual label
    """
    cats = dataset.coco.loadCats(dataset.coco.getCatIds())
    return {cat['id']: cat['name'] for cat in cats}


def ensure_coco_downloaded():
    """
    Ensures COCO 2017 dataset is present locally.

    Structure produced:
        coco/
            train2017/
            val2017/
            annotations/
    """

    os.makedirs(DATA_ROOT, exist_ok=True)

    train_zip = os.path.join(DATA_ROOT, "train2017.zip")
    val_zip = os.path.join(DATA_ROOT, "val2017.zip")
    ann_zip = os.path.join(DATA_ROOT, "annotations_trainval2017.zip")

    download_file("http://images.cocodataset.org/zips/train2017.zip", train_zip)
    download_file("http://images.cocodataset.org/zips/val2017.zip", val_zip)
    download_file("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", ann_zip)

    unzip_file(train_zip, os.path.join(DATA_ROOT, "train2017"))
    unzip_file(val_zip, os.path.join(DATA_ROOT, "val2017"))
    unzip_file(ann_zip, os.path.join(DATA_ROOT, "annotations"))


# ======================================================================
# COLLATE FUNCTION
# ======================================================================
# Detection models expect a list of images and targets.
# Default PyTorch collate tries stacking tensors and breaks.
# ======================================================================
def collate_fn(batch):
    return tuple(zip(*batch))

"""def collate_fn_as_tensor(batch):
    samples, targets = zip(*batch)

    # Se tutte le immagini hanno stessa shape -> stack
    if all(s.shape == samples[0].shape for s in samples):
        samples = torch.stack(samples)
    else:
        samples = list(samples)

    return torch.FloatTensor(samples), targets
"""

# ======================================================================
# DATASET WRAPPER
# ======================================================================
# Torchvision CocoDetection returns annotations in raw COCO format.
# RetinaNet instead expects:
#   target = {
#       "boxes": Tensor[N,4] in XYXY format
#       "labels": Tensor[N]
#   }
# This wrapper performs the conversion.
# ======================================================================
class CocoDetectionWrapper(CocoDetection):

    def __init__(self, root, annFile, transform=None):
        super().__init__(root=root, annFile=annFile)
        self.transform = transform

    def __getitem__(self, idx):
        """
        Converts COCO bbox from XYWH → XYXY and packs target dict.
        """

        img, annotations = super().__getitem__(idx)

        boxes = []
        labels = []

        # --------------------------------------------------
        # Convert annotations
        # --------------------------------------------------
        for obj in annotations:
            x, y, w, h = obj["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(obj["category_id"])

        # Handle images without annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([self.ids[idx]])
        }

        if self.transform:
            img = self.transform(img)

        return img, target