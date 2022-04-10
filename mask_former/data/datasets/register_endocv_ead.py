import os
import json
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from PIL import Image
import numpy as np


def get_endocv_ead_dicts(root_dir, json_name):
    train_json_path = os.path.join(root_dir, json_name)

    with open(train_json_path, 'r') as f:
        json_file = json.load(f)

    dataset_dicts = []
    for idx, file_name in enumerate(list(json_file.keys())):
        record = {}
        image_info = json_file[file_name]

        record['file_name'] = image_info['file_name']
        record['image_id'] = idx
        record['height'] = image_info['height']
        record['width'] = image_info['width']
        record['sem_seg_file_name'] = image_info['mask']
        dataset_dicts.append(record)
    return dataset_dicts


def register_all_endocv_ead():
    root_dir = '/datasets/EndoCV2022_ChallengeDataset/EAD2.0'
    categorynames = ['nonmucosa', 'artefact', 'saturation', 'specularity', 'bubbles']
    for name, d in [('endocv_ead_train', 'train.json'), ('endocv_ead_val', 'val.json')]:
        DatasetCatalog.register(name, lambda d=d: get_endocv_ead_dicts(root_dir, d))
        MetadataCatalog.get(name).set(
            thing_classes=categorynames,
            evaluator_type="sem_seg",
            ignore_label=0,
            stuff_classes=categorynames,
        )


register_all_endocv_ead()