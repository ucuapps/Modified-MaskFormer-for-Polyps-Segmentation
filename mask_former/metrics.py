import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools


class Metric:
    def add(self, y_pred, y_true):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def write_to_tensorboard(self, writer, epoch):
        raise NotImplementedError


class Accuracy(Metric):
    NAME = "accuracy"
    THRESHOLD = 0.5

    def __init__(self):
        self.reset()

    @torch.no_grad()
    def add(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred) > self.THRESHOLD
        y_true = y_true.type(torch.bool)
        self.correct += torch.sum(y_pred == y_true)
        self.total += y_pred.numel()

    def get(self):
        self.score = self.correct / self.total
        return self.score

    def reset(self):
        self.correct = 0.0
        self.total = 0.0

    def write_to_tensorboard(self, writer, epoch):
        writer.add_scalar(f"{self.NAME}", self.score, epoch)


class BinaryF1(Metric):
    NAME = "f1"
    THRESHOLD = 0.5

    def __init__(self):
        self.reset()

    @torch.no_grad()
    def add(self, y_pred, y_true):
        y_pred = (torch.sigmoid(y_pred) > self.THRESHOLD).view(y_pred.shape[0], -1)
        y_true = y_true.type(torch.bool).view(y_true.shape[0], -1)
        self.true_positives += torch.sum(y_pred & y_true)
        self.false_positives += torch.sum(y_pred & ~y_true)
        self.false_negatives += torch.sum(~y_pred & y_true)
        self.true_negatives += torch.sum(~y_pred & ~y_true)

    def get(self):
        self.positive_score = (
                2
                * self.true_positives
                / (2 * self.true_positives + self.false_negatives + self.false_positives)
        )
        self.negative_score = (
                2
                * self.true_negatives
                / (2 * self.true_negatives + self.false_negatives + self.false_positives)
        )
        self.precision = self.true_positives / (
                self.true_positives + self.false_positives
        )
        self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        self.jaccard = self.true_positives / (
                self.true_positives + self.false_positives + self.false_negatives
        )
        return self.positive_score

    def reset(self):
        self.true_positives = 0.0
        self.false_positives = 0.0
        self.false_negatives = 0.0
        self.true_negatives = 0.0

    def write_to_tensorboard(self, writer, epoch):
        writer.add_scalar(f"{self.NAME}-positive", self.positive_score, epoch)
        writer.add_scalar(f"{self.NAME}-negative", self.negative_score, epoch)
        writer.add_scalar(f"precision", self.precision, epoch)
        writer.add_scalar(f"jaccard", self.jaccard, epoch)
        writer.add_scalar(f"recall", self.recall, epoch)


class IoUMetric(Metric):
    NAME = "meanIoU"

    def __init__(self, classes, device, ignore_value=255):
        self.classes = classes
        self.device = device
        self.ignore_value = ignore_value
        self.reset()

    def add(self, output, target):
        output = (output.sigmoid() > 0.5).view(-1).int()
        target = (target.view(-1) > 0).int()

        for i, j in itertools.product(torch.unique(target), torch.unique(output)):
            # print(i, j)
            self.conf_matrix[i, j] += torch.sum((target == i) & (output == j))

    def get(self):
        conf_matrix = self.conf_matrix.float()
        true_positives = torch.diagonal(conf_matrix)
        false_positives = torch.sum(conf_matrix, 0) - true_positives
        false_negatives = torch.sum(conf_matrix, 1) - true_positives

        iou_per_class = true_positives / (
                true_positives + false_negatives + false_positives
        )
        self.score = iou_per_class
        return self.score

    def reset(self):
        self.conf_matrix = torch.zeros(
            (self.classes, self.classes), dtype=torch.int64
        ).to(self.device)

    def write_to_tensorboard(self, writer, epoch):
        writer.add_scalar(self.NAME, self.score, epoch)
