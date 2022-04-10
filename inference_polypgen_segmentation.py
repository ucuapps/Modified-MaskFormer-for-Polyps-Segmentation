import os
import torch
import argparse
import numpy as np
import cv2
from tifffile import tifffile
import ttach as tta

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from mask_former import add_mask_former_config


def parse_args():
    parser = argparse.ArgumentParser("Generation self-attention video")
    parser.add_argument(
        "--config",
        default="config.yaml",
        type=str,
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        default="model_0014999.pth",
        type=str,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--input_path",
        default='/datasets/testDataRound-I/testImagesRound-I',
        type=str,
        help="Path to input test images",
    )
    parser.add_argument(
        "--output_path",
        default='test_inference',
        type=str,
        help="Path to output files",
    )
    return parser.parse_args()


def tta_model_predict(X, predictor):
    tta_transforms = tta.Compose(
        [tta.HorizontalFlip(), tta.VerticalFlip(), tta.Rotate90(angles=[0, 180]), tta.Scale(scales=[0.5, 1, 2])])
    masks = []
    for transformer in tta_transforms:
        augmented_image = transformer.augment_image(X)
        model_output = predictor(augmented_image.squeeze(0).permute(1, 2, 0).numpy())['sem_seg'].unsqueeze(0)
        deaug_mask = transformer.deaugment_mask(model_output)
        masks.append(deaug_mask)

    mask = torch.sum(torch.stack(masks), dim=0) / len(masks)
    return mask


if __name__ == "__main__":
    args = parse_args()

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)

    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.checkpoint

    predictor = DefaultPredictor(cfg)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

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
        tifffile.imsave(mask_name_full, pred_numpy)
