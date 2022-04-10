# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch
import torchvision.transforms.functional as F

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from sklearn.metrics import confusion_matrix
from detectron2.evaluation.evaluator import DatasetEvaluator
from .metrics import Accuracy, BinaryF1, IoUMetric


class SemSegEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
            self,
            dataset_name,
            distributed=False,
            output_dir=None,
            *,
            num_classes=None,
            ignore_label=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)
        if num_classes is not None:
            self._logger.warn(
                "SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata."
            )
        if ignore_label is not None:
            self._logger.warn(
                "SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self.Acc = Accuracy()
        self.F1 = BinaryF1()
        self.IoU = IoUMetric(2, 'cpu')

        self._cpu_device = torch.device("cpu")

        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

    def reset(self):
        self.Acc.reset()
        self.F1.reset()
        self.IoU.reset()

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            pred = output["sem_seg"][0].unsqueeze(0).cpu()

            with PathManager.open(self.input_file_to_gt_file[input["file_name"]], "rb") as f:
                gt = F.to_tensor(Image.open(f))

            self.Acc.add(pred, gt)
            self.F1.add(pred, gt)
            self.IoU.add(pred, gt)

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):
        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        acc = self.Acc.get()
        f1 = self.F1.get()
        prec = self.F1.precision
        rec = self.F1.recall
        iou = self.IoU.get()
        res = {}

        res["mIoU"] = 100 * iou[1].item()
        res['F1'] = 100 * f1.item()
        res["Acc"] = 100 * acc.item()
        res['Prec'] = 100 * prec.item()
        res['Rec'] = 100 * rec.item()

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        print(results)
        self._logger.info(results)
        return results