import os
import time
import warnings
from collections import OrderedDict

from torch import Tensor
from torch.utils.data import Subset

from adv.evasion import LpPerturbationModels
from local_src.utils.ObjectDetectorWrapper import ObjectDetectorWrapper
from local_src.utils.coco_dataset_management import DATA_ROOT, ensure_coco_downloaded, CocoDetectionWrapper, \
    load_coco_categories, collate_fn
from local_src.utils.custom_PGD import wrappedPGD
from local_src.utils.od_metric_mAP import evaluate_image_map
from local_src.utils.visualization import visualize_image

# ======================================================================
# WORKAROUND WINDOWS + MKL
# ======================================================================
# Some PyTorch / MKL installations on Windows load OpenMP twice and crash.
# This environment variable disables the safety check.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.datasets import CocoDetection
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import urllib.request
import zipfile
from tqdm import tqdm
from secmlt.adv.evasion.pgd import PGD
from secmlt.adv.backends import Backends



# ======================================================================
# GLOBAL CONFIGURATION
# ======================================================================
# DATA_ROOT: where COCO will be downloaded / extracted
# DEVICE: GPU if available, otherwise CPU
# NUM_SAMPLES_VIS: number of images to visualize & evaluate
# SCORE_THRESHOLD: minimum confidence to consider predictions valid
# IOU_THRESHOLD: IoU used for TP/FP assignment in metric
# ======================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES_VIS = 6
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
BATCH_COUNT = 2
BATCH_SIZE = 8
DATASET_SIZE = BATCH_SIZE * BATCH_COUNT
# --------------------------------------------------
# SHUFFLE CONTROL (Reproducible and optional)
# --------------------------------------------------
ENABLE_SHUFFLE = True      # Enable / disable dataset shuffle
SHUFFLE_SEED = 42           # Seed used when shuffle is enabled



# ======================================================================
# MAIN PIPELINE
# ======================================================================

def set_seed(seed):
    """
    Sets seed for full reproducibility.

    Controls:
        - Python random
        - PyTorch CPU RNG
        - PyTorch CUDA RNG

    NOTE:
    This does NOT force deterministic GPU kernels, which can be
    enabled separately if strict determinism is required.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def main():

    global COCO_CATEGORIES

    ensure_coco_downloaded()

    transform = T.ToTensor()

    # Dataset
    val_dataset = CocoDetectionWrapper(
        root=os.path.join(DATA_ROOT, "val2017"),
        annFile=os.path.join(DATA_ROOT, "annotations/instances_val2017.json"),
        transform=transform
    )

    COCO_CATEGORIES = load_coco_categories(val_dataset)

    # --------------------------------------------------
    # Optional reproducible shuffle
    # --------------------------------------------------
    generator = None

    if ENABLE_SHUFFLE:
        set_seed(SHUFFLE_SEED)
        generator = torch.Generator()
        generator.manual_seed(SHUFFLE_SEED)

    # --------------------------------------------------
    # Reduce Loader size for debugging
    # --------------------------------------------------
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=ENABLE_SHUFFLE,
        generator=generator,
        collate_fn=collate_fn
    )

    subset_size = DATASET_SIZE
    indices = list(range(subset_size))

    val_subset = Subset(val_loader.dataset, indices)

    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=val_loader.num_workers,
        collate_fn=val_loader.collate_fn
    )

    # --------------------------------------------------
    # Load pretrained RetinaNet and setup attack
    # --------------------------------------------------

    # RetinaNet:    prediction return -> boxes [x1, y1, x2, y2], labels, scores
    weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
    model = retinanet_resnet50_fpn(weights=weights)

    # FasterRCNN:   prediction return -> boxes [x1, y1, x2, y2], labels, scores
    #weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    #model = fasterrcnn_resnet50_fpn(weights=weights)

    # MaskRCNN:     prediction return -> boxes [x1, y1, x2, y2], labels, scores, masks
    #weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    #model = maskrcnn_resnet50_fpn(weights=weights)

    # TODO: WRAP model
    model.to(DEVICE)
    model.eval()
    model = ObjectDetectorWrapper(model)

    """
        per gestire il ridimensionamento delle immagini internamente al modello viene usata:
        self.transform(images, targets)
        che a sua volta è un alias per: GeneralizedRCNNTransform(...)
    """


    # TODO: Implement attack
    # Create and run attack
    epsilon = 0.5
    num_steps = 10
    step_size = 0.005
    perturbation_model = LpPerturbationModels.LINF
    y_target = None
    native_attack = PGD(
        perturbation_model=perturbation_model,
        epsilon=epsilon,
        num_steps=num_steps,
        step_size=step_size,
        random_start=False,
        y_target=None,
        backend=Backends.NATIVE,
    )

    native_adv_ds = native_attack(model, val_loader)

    """# NON PUO' STARE SU TORCH.NO_GRAD
    for images, targets in val_loader:
        # TODO: gradients va rifinito internamente, capito qual è il fine dei gradienti, e quali servono,
        #  perchè l'output di un object detector non è unico

        gradients = model.gradient(images, targets)

        for im, pr in zip(images, gradients):
            pr /= pr.max()
            visualize_image(im, COCO_CATEGORIES, gt=None, pred=None, title=f"Image {0}", threshold=0)
            visualize_image(pr, COCO_CATEGORIES, gt=None, pred=None, title=f"Image {0}", threshold=0)


        # TODO: a che serviva questa parte che segue? solo informativa?

         ### MODELLO IN EVAL, LA CALL RITORNA LA PREDIZIONE ################
        images = [xi.to(DEVICE) for xi in images]
        loss_dict1 = model.model(images)

        ### MODELLO IN TRAIN, LA CALL RITORNA LA LOSS ######################
        images = [xi.to(model._get_device()) for xi in images]  # RIC EDIT
        targets = [{k: v.to(model._get_device()) for k, v in t.items()} for t in targets]
        model.model.train(True)# = True
        loss_dict2 = model.model(images, targets)

        model.model.eval()
        break"""


    # --------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------
    print("Evaluating per-image mAP...")
    idx = 0
    with torch.no_grad():
        for images, targets in native_adv_ds:
            # Forward pass (image list required)
            start = time.time()
            output_batch = model(images)
            print(time.time() - start)

            # Move predictions in cpu
            output_batch_cpu = [
                {k: v.cpu() for k, v in output.items()}
                for output in output_batch
            ]

            # Plot predicted image
            for img, gt, pred in zip(images, targets, output_batch_cpu):

                # TODO: bug su GT
                #gt = gt[0]

                visualize_image(img, COCO_CATEGORIES, gt=gt, pred=pred, title=f"Image {idx}", threshold=SCORE_THRESHOLD)

                map_val = evaluate_image_map(gt, pred)
                print(f"Image {idx} -> mAP: {map_val:.4f}")

                idx+=1
                if idx >= NUM_SAMPLES_VIS:
                    break
            if idx >= NUM_SAMPLES_VIS:
                break

    # --------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------
    print("Evaluating per-image mAP...")
    idx = 0
    with torch.no_grad():
        for images, targets in val_loader:

            output_batch = model(images)

            # Move predictions in cpu
            output_batch_cpu = [
                {k: v.cpu() for k, v in output.items()}
                for output in output_batch
            ]

            # Plot predicted image
            for img, gt, pred in zip(images, targets, output_batch_cpu):

                visualize_image(img, COCO_CATEGORIES, gt=gt, pred=pred, title=f"Image {idx}",
                                threshold=SCORE_THRESHOLD)

                map_val = evaluate_image_map(gt, pred)
                print(f"Image {idx} -> mAP: {map_val:.4f}")

                idx += 1
                if idx >= NUM_SAMPLES_VIS:
                    break
            if idx >= NUM_SAMPLES_VIS:
                break

if __name__ == "__main__":
    main()
