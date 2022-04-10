from detectron2.data import transforms as T


__all__ = ['ResizeShortestEdge']

# EXAMPLES --------------------------------------------------------


class MyColorAugmentation(T.Augmentation):
    def get_transform(self, image):
        r = np.random.rand(2)
        return T.ColorTransform(lambda x: x * r[0] + r[1] * 10)


class MyCustomResize(T.Augmentation):
    def get_transform(self, image):
        old_h, old_w = image.shape[:2]
        new_h, new_w = int(old_h * np.random.rand()), int(old_w * 1.5)
        return T.ResizeTransform(old_h, old_w, new_h, new_w)


class MyCustomCrop(T.Augmentation):
    def get_transform(self, image, sem_seg):
        # decide where to crop using both image and sem_seg
        return T.CropTransform(...)

# OUR AUGMENTATIONS -----------------------------------------------


class ResizeShortestEdge(T.Augmentation):
    pass