import os
import json
import torch
import numpy as np
import cv2
from tifffile import tifffile
from skimage.measure import label, regionprops

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from mask_former import add_mask_former_config
from inference_polypgen_segmentation import tta_model_predict, parse_args


def get_bboxes_from_mask(mask):
    lbl_0 = label(mask)
    props = regionprops(lbl_0)
    return props


coco_format = {
    "images": [
        {
        }
    ],
    "categories": [

    ],
    "annotations": [
        {
        }
    ]
}


def create_image_annotation(file_name, width, height, image_id):
    images = {
        'file_name': file_name,
        'height': height,
        'width': width,
        'id': image_id
    }
    return images


def create_annotation_coco_format(min_x, min_y, width, height, score, image_id, category_id, annotation_id):
    bbox = (min_x, min_y, width, height)
    area = width * height
    annotation = {
        'id': annotation_id,
        'image_id': image_id,
        'bbox': bbox,
        'area': area,
        'iscrowd': 0,
        'category_id': category_id,
        'segmentation': [],
        'score': float(score)
    }

    return annotation


if __name__ == "__main__":
    args = parse_args()

    json_filename = os.path.join(args.output_path, f'round1.json')

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)

    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.checkpoint

    predictor = DefaultPredictor(cfg)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    dataset = {'categories': [], 'annotations': [], 'images': []}

    classes = ['polyp']

    for i, cls_ in enumerate(classes, 1):
        dataset['categories'].append({'id': i, 'name': cls_, 'supercategory': 'mark'})

    count = 0
    annot_count = 1

    images = []
    annotations = []
    score = []

    for i, im_name in enumerate(os.listdir(args.input_path)):
        mask_name = im_name.replace('.jpg', '.tif')

        im_name_full = os.path.join(args.input_path, im_name)
        mask_name_full = os.path.join(args.output_path, mask_name)
        im = cv2.imread(im_name_full)

        # TTA Code
        t_im = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0)
        outputs = tta_model_predict(t_im, predictor)[0][0].sigmoid() > 0.5
        pred = outputs.to('cpu').numpy()

        pred_numpy = np.where(pred, 255, 0).astype('uint8')
        tifffile.imsave(mask_name_full, pred_numpy)  # uncomment this line to save segmentation masks

        file_id = im_name
        mask = pred_numpy

        height, width = mask.shape
        images.append(create_image_annotation(file_id, width, height, count))

        props = get_bboxes_from_mask(mask)

        for prop in props:
            annot_count += 1
            cls_id, score, x1, y1, x2, y2 = ('polyp', 1.0, prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2])
            width_box = max(0, float(x2) - float(x1))
            height_box = max(0, float(y2) - float(y1))

            annotations.append(
                create_annotation_coco_format(float(x1), float(y1), width_box, height_box, score, count, 1,
                                              annot_count))

        count = count + 1

    classes = ['polyp']

    coco_format['images'], coco_format['annotations'] = images, annotations

    for index, label in enumerate(classes):
        ann = {
            "supercategory": "none",
            "id": index + 1,  # Index starts with '1' .
            "name": label
        }
        coco_format['categories'].append(ann)

    with open(json_filename, 'w') as f:
        json.dump(coco_format, f)
