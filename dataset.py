import os
from glob import glob
import cv2

def sortfunc(name):
    base_name = os.path.basename(name)
    return int(base_name.split('_')[0])

class Dataset:
    def __init__(self, image_dir, mask_dir, is_reverse=False) -> None:
        images_path = glob(image_dir + "/*.jpg")
        masks_path = glob(mask_dir + "/*.png")
        assert len(images_path) == len(masks_path)
        self.images_path = images_path
        self.masks_path = masks_path

        self.images_path = sorted(self.images_path, key=sortfunc, reverse=is_reverse)
        self.masks_path = sorted(self.masks_path, key=sortfunc, reverse=is_reverse)
    
    def __getitem__(self, key):
        image = cv2.imread(self.images_path[key])
        mask = cv2.imread(self.masks_path[key])
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]))
        return image, mask, self.images_path[key]
    
    def __len__(self):
        return len(self.images_path)