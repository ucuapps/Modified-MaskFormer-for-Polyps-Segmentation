import os
import json
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from PIL import Image
import numpy as np


def get_hyperkvasir_dicts(root_path, json_name, segmentation_masks_dir):
    images_dir = os.path.join(root_path, 'images')
    masks_dir = os.path.join(root_path, 'masks')

    json_path = os.path.join(root_path, json_name)

    with open(json_path, 'r') as file:
        json_file = json.load(file)

    segmentation_masks_dir = os.path.join(root_path, segmentation_masks_dir)
    if not os.path.exists(segmentation_masks_dir):
        os.mkdir(segmentation_masks_dir)

    dataset_dicts = []
    for idx, file_name in enumerate(list(json_file.keys())):
        record = {}
        image_info = json_file[file_name]
        image_path = os.path.join(images_dir, file_name + '.jpg')
        mask_path = os.path.join(masks_dir, file_name + '.jpg')

        mask = np.asarray(Image.open(mask_path).convert('L'))
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = (mask / 255).astype(np.int32)
        mask_path = os.path.join(segmentation_masks_dir, f'{file_name}.png')
        if not os.path.exists(mask_path):
            im = Image.fromarray(mask).convert('L')
            im.save(mask_path)

        record['file_name'] = image_path
        record['image_id'] = idx
        record['height'] = image_info['height']
        record['width'] = image_info['width']
        record['sem_seg_file_name'] = mask_path
        dataset_dicts.append(record)
    return dataset_dicts


def register_all_hyperkvasir():
    root_path = '/home/mariiak/endoscopy_datasets/hyper-kvasir-segmented-images'
    for name, d in [('hyperkvasir_train', 'train.json'), ('hyperkvasir_val', 'val.json')]:
        DatasetCatalog.register(name, lambda d=d: get_hyperkvasir_dicts(root_path, d, 'pre_masks'))
        MetadataCatalog.get(name).set(
            thing_classes=["pathology"],
            evaluator_type="bin_seg",
            ignore_label=0,
            stuff_classes=[''],
        )


register_all_hyperkvasir()