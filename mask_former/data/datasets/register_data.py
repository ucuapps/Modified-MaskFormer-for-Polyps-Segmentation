import os
import json

import numpy as np

from PIL import Image
from cv2 import cv2

from detectron2.data import DatasetCatalog, MetadataCatalog


# def get_hyperkvasir_dicts(root_path, json_name, segmentation_masks_dir):
#     images_dir = os.path.join(root_path, 'images')
#     masks_dir = os.path.join(root_path, 'masks')
#
#     json_path = os.path.join(root_path, json_name)
#
#     with open(json_path, 'r') as file:
#         json_file = json.load(file)
#
#     segmentation_masks_dir = os.path.join(root_path, segmentation_masks_dir)
#     if not os.path.exists(segmentation_masks_dir):
#         os.mkdir(segmentation_masks_dir)
#
#     dataset_dicts = []
#     for idx, file_name in enumerate(list(json_file.keys())):
#         record = {}
#         image_info = json_file[file_name]
#         image_path = os.path.join(images_dir, file_name + '.jpg')
#         mask_path = os.path.join(masks_dir, file_name + '.jpg')
#
#         mask = np.asarray(Image.open(mask_path).convert('L'))
#         _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
#         mask = (mask / 255).astype(np.int32)
#         mask_path = os.path.join(segmentation_masks_dir, f'{file_name}.png')
#         if not os.path.exists(mask_path):
#             im = Image.fromarray(mask)
#             print('Image size: ', im.size)
#             im.save(mask_path)
#
#         record['file_name'] = image_path
#         record['image_id'] = idx
#         record['height'] = image_info['height']
#         record['width'] = image_info['width']
#         record['sem_seg_file_name'] = mask_path
#         dataset_dicts.append(record)
#     return dataset_dicts


def get_EndoCV_dicts(root_path, json_name, segmentation_masks_dir, cut_imgs_dir):

    def get_cut_img(im):
        """
        Returns cut image, amount of pixels cut from the left side, amount of pixels cut from the right side
        """
        np_im = np.array(im)
        shape = np_im.shape
        h = shape[0] // 2
        w = shape[1]
        row = np_im[h, :, :]
        indices = np.argwhere(row.sum(axis=1) > 16 * 3).flatten()
        left, right = indices[0], indices[-1]
        return np_im[:, left:right + 1, :], left - 1, w - right

    # root_path = DIR
    # json_name = 'val.json'
    # segmentation_masks_dir = '/datasets/EndoCV2022_ChallengeDataset/segmentation_masks_dir'

    json_path = os.path.join(root_path, json_name)

    with open(json_path, 'r') as file:
        json_file = json.load(file)

    if not os.path.exists(segmentation_masks_dir):
        os.mkdir(segmentation_masks_dir)

    dataset_dicts = []
    for idx, file_name in enumerate(list(json_file.keys())):
        record = {}
        image_info = json_file[file_name]
        image_path = file_name
        mask_path = file_name.replace('images', 'masks')
        mask_path = mask_path.split('.')[0] + '.' + mask_path.split('.')[1] + '_mask.' + mask_path.split('.')[2]

        new_mask_path = mask_path.replace(root_path + '/', '').replace('/', '_').replace('.jpg', '.png')
        new_mask_path = os.path.join(segmentation_masks_dir, new_mask_path)

        new_img_path = image_path.replace(root_path + '/', '').replace('/', '_')  # .replace('.jpg', '.png')
        new_img_path = os.path.join(cut_imgs_dir, new_img_path)

        if not os.path.exists(new_img_path):

            original_im = Image.open(image_path)
            cut_im, cut_left, cut_right = get_cut_img(original_im)
            im = Image.fromarray(cut_im)
            im.save(new_img_path)

            try:
                mask = np.asarray(Image.open(mask_path).convert('L'))
            except FileNotFoundError:
                try:
                    mask_path = mask_path.replace('.jpg', '.tif')
                    mask = np.asarray(Image.open(mask_path).convert('L'))
                except FileNotFoundError:
                    mask_path = mask_path.replace('_mask', '')
                    mask = np.asarray(Image.open(mask_path).convert('L'))

            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = (mask / 255).astype(np.int32)

            mask = mask[:, cut_left + 1: mask.shape[1] - cut_right + 1]

            im = Image.fromarray(mask)
            im.save(new_mask_path)

            # print(cut_im.shape[0], mask.shape[0], cut_im.shape[1], mask.shape[1] )
            assert cut_im.shape[0] == mask.shape[0] and cut_im.shape[1] == mask.shape[1]

            # if idx % 200 == 0:
            #     plt.figure()
            #     plt.subplot(1, 4, 1)
            #     plt.imshow(Image.open(image_path))
            #     plt.axis('off')
            #     plt.subplot(1, 4, 2)
            #     plt.imshow(Image.open(new_img_path))
            #     plt.axis('off')
            #     plt.subplot(1, 4, 3)
            #     plt.imshow(Image.open(mask_path).convert('L'))
            #     plt.axis('off')
            #     plt.subplot(1, 4, 4)
            #     plt.imshow(Image.open(new_mask_path).convert('L'))
            #     plt.axis('off')
            #     plt.show()
        record['file_name'] = new_img_path
        record['image_id'] = idx
        # record['height'] = image_info['height']
        # record['width'] = image_info['width']
        record['sem_seg_file_name'] = new_mask_path
        dataset_dicts.append(record)

    return dataset_dicts


# def register_hyperkvasir():
#
#     root_path = '/home/mariiak/endoscopy_datasets/hyper-kvasir-segmented-images'
#     segmentation_masks_dir = '/home/mariiak/endoscopy_datasets/hyper-kvasir-segmented-images/pre_masks'
#
#     for name, d in [
#         ('train', 'train.json'),
#         ('test', 'val.json'),
#     ]:
#         # dataset_dicts = get_hyperkvasir_dicts(root_path, json_name)
#         # image_dir = os.path.join(root, image_dirname)
#         # gt_dir = os.path.join(root, sem_seg_dirname)
#         name = f"hyperkvasir_{name}"
#         DatasetCatalog.register(
#             name, lambda d=d: get_hyperkvasir_dicts(root_path, d, segmentation_masks_dir)
#         )
#         # images_dir = os.path.join(root_path, 'images')
#         # masks_dir = os.path.join(root_path, 'masks')
#         MetadataCatalog.get(name).set(
#             # image_root=images_dir,
#             # sem_seg_root=masks_dir,
#             evaluator_type="bin_seg",
#             ignore_label=0,
#             # stuff_classes=['helthy_tissue', 'pathology'],
#             # **meta,
#         )


def register_EndoCV():

    root_path = '/datasets/EndoCV2022_ChallengeDataset/PolypGen2.0'
    # segmentation_masks_dir = '/datasets/EndoCV2022_ChallengeDataset/segmentation_masks_dir'
    # segmentation_masks_dir = '/home/mariiak/segmentation_masks_dir2'
    # cut_imgs_dir = '/home/mariiak/cut_images_dir'
    segmentation_masks_dir = '/datasets/extra_space2/dobko/segmentation_masks_dir'
    cut_imgs_dir = '/datasets/extra_space2/dobko/cut_images_ALL_dir'


    for name, d in [
        ('train', 'train_non_empty.json'),
        ('test', 'val.json'),
    ]:
        # dataset_dicts = get_hyperkvasir_dicts(root_path, json_name)
        # image_dir = os.path.join(root, image_dirname)
        # gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"EndoCV_{name}"
        DatasetCatalog.register(
            name, lambda d=d: get_EndoCV_dicts(root_path, d, segmentation_masks_dir, cut_imgs_dir)
        )
        # images_dir = os.path.join(root_path, 'images')
        # masks_dir = os.path.join(root_path, 'masks')
        MetadataCatalog.get(name).set(
            # image_root=images_dir,
            # sem_seg_root=masks_dir,
            evaluator_type="bin_seg",
            ignore_label=0,
            # stuff_classes=['helthy_tissue', 'pathology'],
            # **meta,
        )

# register_hyperkvasir()
register_EndoCV()