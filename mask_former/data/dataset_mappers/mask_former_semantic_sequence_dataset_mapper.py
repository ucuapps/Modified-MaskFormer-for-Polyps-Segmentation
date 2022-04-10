# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import glob
import os

import numpy as np
import torch
from torch.nn import functional as F

from PIL import Image

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

from mask_former.data.transforms import ResizeShortestEdge

__all__ = ["MaskFormerSemanticSequenceDatasetMapper"]


class MaskFormerSemanticSequenceDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        # TODO: add custom augmentations - to control the random seed, look at mask_former/data/transformes.py
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerSemanticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        #  '/datasets/extra_space2/dobko/cut_images_ALL_dir/seq21_images_seq6_C6_99.jpg'

        # TODO: remove this path, to work for a general case
        root_path = '/datasets/EndoCV2022_ChallengeDataset/PolypGen2.0'
        seq = dataset_dict["file_name"].split('/')[-1].split('_')[0]
        base_dir = os.path.join(root_path, seq, 'images')

        all_ids_sequence = np.sort([x.split('_')[-1][:-4] for x in glob.glob(base_dir + '/**')])[:-1]

        cur_id = dataset_dict["file_name"].split('/')[-1].split('_')[-1][:-4]
        previous_ids = [x for x in all_ids_sequence if int(x) < int(cur_id)]

        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # If we have the first two frames, then we just copy the same image

        if cur_id == all_ids_sequence[0]:
            images = [image, image, image]
        elif cur_id == all_ids_sequence[1]:
            image1 = utils.read_image(dataset_dict["file_name"].replace(cur_id, previous_ids[-1]), format=self.img_format)
            print(dataset_dict["file_name"].replace(cur_id, previous_ids[-1]))
            images = [image1, image, image]
        else:
            image1 = utils.read_image(dataset_dict["file_name"].replace(cur_id, previous_ids[-1]), format=self.img_format)
            print(dataset_dict["file_name"].replace(cur_id, previous_ids[-1]))
            image2 = utils.read_image(dataset_dict["file_name"].replace(cur_id, previous_ids[-2]), format=self.img_format)
            images = [image2, image1, image]

        def binary_loader(path):
            with open(path, 'rb') as f:
                try:
                    img = Image.open(f)
                except:
                    img = Image.fromarray(tifffile.imread(path).astype('uint8'))
                # return img.convert('1')
                return np.asarray(img.convert('L'))

        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            # sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
            # TODO: Change the behaviour for Multiclass segmentation
            sem_seg_gt = binary_loader(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        # TODO: add custom augmentations - look above
        images_aug = []
        for img in images:
            print(img.shape, sem_seg_gt.shape, '----------------------------')

            aug_input = T.AugInput(img, sem_seg=sem_seg_gt)
            aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
            img = torch.as_tensor(np.ascontiguousarray(aug_input.image.transpose(2, 0, 1)))
            sem_seg_gt = aug_input.sem_seg

            # Pad image and segmentation label here!
            if sem_seg_gt is not None:
                sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

            if self.size_divisibility > 0:
                image_size = (img.shape[-2], img.shape[-1])
                padding_size = [
                    0,
                    self.size_divisibility - image_size[1],
                    0,
                    self.size_divisibility - image_size[0],
                ]
                img = F.pad(img, padding_size, value=128).contiguous()
                if sem_seg_gt is not None:
                    sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()
            images_aug.append(img)

        image_shape = (images[-1].shape[-2], imageimages[-1].shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.

        dataset_dict["image"] = torch.cat(images, 0)

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        return dataset_dict
