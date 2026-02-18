"""Wrappers for PyTorch models."""
from collections import OrderedDict
from typing import List

import torch
from secmlt.models.base_model import BaseModel
from secmlt.models.data_processing.data_processing import DataProcessing
from secmlt.models.pytorch.base_pytorch_trainer import BasePyTorchTrainer
from torch.utils.data import DataLoader


class ObjectDetectorWrapper(BaseModel):
    """Wrapper for PyTorch Object Detector."""

    def __init__(
        self,
        model: torch.nn.Module,
        preprocessing: DataProcessing = None,
        postprocessing: DataProcessing = None,
        trainer: BasePyTorchTrainer = None,
    ) -> None:
        """
        Create wrapped PyTorch classifier.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model.
        preprocessing : DataProcessing, optional
            Preprocessing to apply before the forward., by default None.
        postprocessing : DataProcessing, optional
            Postprocessing to apply after the forward, by default None.
        trainer : BasePyTorchTrainer, optional
            Trainer object to train the model, by default None.
        """
        super().__init__(preprocessing=preprocessing, postprocessing=postprocessing)
        self._model: torch.nn.Module = model
        self._trainer = trainer

    @property
    def model(self) -> torch.nn.Module:
        """
        Get the wrapped instance of PyTorch model.

        Returns
        -------
        torch.nn.Module
            Wrapped PyTorch model.
        """
        return self._model

    def _get_device(self) -> torch.device:
        return next(self._model.parameters()).device

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the predicted class for the given samples.

        Parameters
        ----------
        x : torch.Tensor
            Input samples.

        Returns
        -------
        # TODO: define output
        torch.Tensor
            Predicted class for the samples.
        """
        scores = self.decision_function(x)
        return torch.argmax(scores, dim=-1)

    def _decision_function(self, x: List[torch.Tensor]) -> List[dict]:
        """
        Compute decision function of the model for a list of images.

        Parameters
        ----------
        x : List[torch.Tensor]
            List of input images. Each tensor must have shape [C, H, W].
            This format follows the torchvision object detection API,
            where images may have different spatial resolutions and
            therefore cannot be stacked into a single batch tensor.

        Returns
        -------
        List[dict]
            Model predictions, one dictionary per input image.
            Each dictionary typically contains:
                - "boxes": Tensor[N, 4]
                - "labels": Tensor[N]
                - "scores": Tensor[N]
        """

        # Move each image tensor to the model device
        x = [xi.to(self._get_device()) for xi in x]

        # Forward pass through detection model
        return self._model(x)


    def gradient(self, x: torch.Tensor, y: int) -> torch.Tensor:
        """
        Compute batch gradients of class y w.r.t. x.

        Parameters
        ----------
        x : torch.Tensor
            Input samples.
        y : int
            Class label.

        Returns
        -------
        torch.Tensor
            Gradient of class y w.r.t. input x.
        """
        # TODO: Restore gradient evaluation for classification
        """x = x.clone().requires_grad_()
        if x.grad is not None:
            x.grad.zero_()
        output = self.decision_function(x)
        output = output[:, y].sum()
        output.backward()
        return x.grad"""

        # Move samples to device
        x = [
            xi.to(self._get_device())
            for xi in x
        ]

        y = [{k: v.to(self._get_device()) for k, v in t.items()} for t in y]

        # Create differentiable inputs
        x = [
            xi.clone().detach().requires_grad_(True)
            for xi in x
        ]
        # FROM LOSS =========================

        # TODO: Potrebbe aver più senso prendere uno step intermedio della forward, che si ferma prima del POST processing
        # Forward pass (detection inference)
        self.model.train()
        detections = self.model.forward(x, y)
        self.model.eval()
        # TODO: Semanticamente potrebbe non avere senso, verifica solo la restituzione di un gradiente

        detection_sum = sum(detections.values())
        detection_sum.backward()

        # FROM PREDICTIONS =========================

        """detections = self.model.forward(x)  # , y)
        detection_score_sum = sum(
            detection["scores"].sum()
            for detection in detections
        )
        detection_boxes_sum = sum(
            detection["boxes"].sum()
            for detection in detections
        )
        detection_sum = detection_boxes_sum + detection_score_sum
        # Backward pass
        detection_sum.backward()"""

        # Debug autograd connectivity
        print(detection_sum.requires_grad)
        print(detection_sum.grad_fn)

        # Retrieve gradient w.r.t. first image
        return [xi.grad for xi in x]


    def train(self, dataloader: DataLoader) -> torch.nn.Module:
        """
        Train the model with given dataloader, if trainer is set.

        Parameters
        ----------
        dataloader : DataLoader
            Training PyTorch dataloader to use for training.

        Returns
        -------
        torch.nn.Module
            Trained PyTorch model.

        Raises
        ------
        ValueError
            Raises ValueError if the trainer is not set.
        """
        if self._trainer is None:
            msg = "Cannot train without a trainer."
            raise ValueError(msg)
        return self._trainer.train(self._model, dataloader)
