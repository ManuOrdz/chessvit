"""Compute batch statistics
"""

import torch
import numpy as np


def _fraction(a: float, b: float) -> float:
    return a / b if b != 0 else 0


class StatsAggregator:
    """Simple class for aggregating statistics between batches."""

    def __init__(self, classes: list):
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.reset()
        self.mistakes = []

    def reset(self):
        """Reset the aggregator."""
        self.confusion_matrix = np.zeros(
            (len(self.classes), len(self.classes)), dtype=np.uint32
        )
        self.mistakes = []

    def add_batch(
        self,
        one_hot_outputs: torch.Tensor,
        labels: torch.Tensor,
        inputs: torch.Tensor = None,
    ):
        """Add a batch to compute statistics over.

        Args:
            one_hot_outputs (torch.Tensor): the one hot outputs of the model
            labels (torch.Tensor): the groundtruth labels
            inputs (torch.Tensor, optional): the inputs (if supplied, will be used to keep track of mistakes)
        """
        outputs = one_hot_outputs.cpu().argmax(axis=-1).numpy()
        labels = labels.cpu().numpy()
        for predicted_class, _ in enumerate(self.classes):
            predicted_mask = outputs == predicted_class
            for actual_class, _ in enumerate(self.classes):
                actual_mask = labels == actual_class
                self.confusion_matrix[predicted_class, actual_class] += (
                    actual_mask & predicted_mask
                ).sum()
        if inputs is not None:
            mistakes_mask = outputs != labels
            mistakes = inputs[mistakes_mask].cpu().numpy()
            groundtruth = map(self.classes.__getitem__, labels[mistakes_mask])
            self.mistakes.extend(zip(groundtruth, mistakes))

    def add_batch_segmentation(
        self, outputs: torch.Tensor, labels: torch.Tensor, inputs: torch.Tensor = None
    ):
        """Accumulate statistics for a segmentation task (binary or multi-class)."""
        preds = outputs.cpu().numpy()

        # --- Calcular matriz de confusión ---
        for predicted_class, _ in enumerate(self.classes):
            pred_mask = preds == predicted_class
            for actual_class, _ in enumerate(self.classes):
                gt_mask = labels == actual_class
                self.confusion_matrix[predicted_class, actual_class] += (
                    gt_mask & pred_mask
                ).sum()

        # --- Guardar errores opcionalmente ---
        if inputs is not None:
            mistakes_mask = preds != labels
            mistakes = inputs[mistakes_mask].cpu().numpy()
            groundtruth = map(self.classes.__getitem__, labels[mistakes_mask])
            self.mistakes.extend(zip(groundtruth, mistakes))

    def accuracy(self) -> float:
        """Obtain the overall accuracy.

        Returns:
            float: the accuracy
        """
        correct = np.trace(self.confusion_matrix)  # sum along diagonal
        total = np.sum(self.confusion_matrix)
        return _fraction(correct, total)

    def precision(self, cls: str) -> float:
        """Obtain the precision for a specific class label.

        Args:
            cls (str): the class

        Returns:
            float: the precision
        """
        idx = self.class_to_idx[cls]
        tp = self.confusion_matrix[idx, idx]
        tp_plus_fp = self.confusion_matrix[idx].sum()
        return _fraction(tp, tp_plus_fp)

    def recall(self, cls: str) -> float:
        """Obtain the recall for a specific class label.

        Args:
            cls (str): the class

        Returns:
            float: the recall
        """
        idx = self.class_to_idx[cls]
        tp = self.confusion_matrix[idx, idx]
        p = self.confusion_matrix[:, idx].sum()
        return _fraction(tp, p)

    def f1_score(self, cls: str) -> float:
        """Obtain the F1-score for a specific class label.

        Args:
            cls (str): the class

        Returns:
            float: the F1-score
        """
        precision = self.precision(cls)
        recall = self.recall(cls)
        return _fraction(2 * precision * recall, precision + recall)

    # ============================
    # SEMANTIC SEGMENTATION METRICS
    # ============================

    def iou(self, cls: str) -> float:
        """Compute the Intersection over Union (IoU) for a specific class."""
        idx = self.class_to_idx[cls]
        tp = self.confusion_matrix[idx, idx]
        fp = self.confusion_matrix[idx, :].sum() - tp
        fn = self.confusion_matrix[:, idx].sum() - tp
        return _fraction(tp, tp + fp + fn)

    def mean_iou(self) -> float:
        """Compute the mean Intersection over Union (mIoU)."""
        ious = [self.iou(cls) for cls in self.classes]
        return np.mean(ious)

    def dice_score(self, cls: str) -> float:
        """Compute the Dice coefficient (F1 for segmentation) for a class."""
        idx = self.class_to_idx[cls]
        tp = self.confusion_matrix[idx, idx]
        fp = self.confusion_matrix[idx, :].sum() - tp
        fn = self.confusion_matrix[:, idx].sum() - tp
        return _fraction(2 * tp, 2 * tp + fp + fn)

    def mean_dice(self) -> float:
        """Compute the mean Dice coefficient across all classes."""
        dices = [self.dice_score(cls) for cls in self.classes]
        return np.mean(dices)

    def pixel_accuracy(self) -> float:
        """Compute the pixel accuracy (ratio of correctly classified pixels)."""
        correct = np.trace(self.confusion_matrix)
        total = np.sum(self.confusion_matrix)
        return _fraction(correct, total)

    def balanced_accuracy(self) -> float:
        """Compute the balanced accuracy (mean of per-class recalls)."""
        recalls = [self.recall(cls) for cls in self.classes]
        return np.mean(recalls)
