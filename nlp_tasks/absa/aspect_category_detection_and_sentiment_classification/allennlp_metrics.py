from typing import Dict, Optional, Tuple, Union, List

from allennlp.training import metrics
import torch
from overrides import overrides
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


class BinaryF1(metrics.Metric):
    """

    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.values = None

    @overrides
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        gold_labels_np = gold_labels.numpy()
        predictions_np = predictions.numpy()
        predictions_np = predictions_np >= self.threshold
        predictions_np = predictions_np.astype(np.int)

        precision = precision_score(gold_labels_np, predictions_np)
        recall = recall_score(gold_labels_np, predictions_np)
        f1 = f1_score(gold_labels_np, predictions_np)

        self.values = {
            "precision": precision,
            "recall": recall,
            "fscore": f1
        }

    def get_metric(self, reset: bool = False) -> Union[float, Tuple[float, ...], Dict[str, float], Dict[str, List[float]]]:
        """
        Compute and return the metric. Optionally also call :func:`self.reset`.
        """
        result = self.values
        if reset:
            self.reset()
        return result

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        self.threshold = 0.5
        self.values = None