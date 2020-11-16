from albumentations import Compose, ShiftScaleRotate,Resize, Rotate, Normalize, OneOf, ElasticTransform, GridDistortion, OpticalDistortion, CoarseDropout
import numpy as np
from skimage.transform import AffineTransform, warp
import albumentations as A
import cv2
from dataset.AutoAugment import *
from PIL import Image, ImageEnhance, ImageOps

def train_aug(image_size):

    return Compose([Resize(*image_size),
                    ShiftScaleRotate(rotate_limit=10),
                    Normalize()
                    ], 
                    p = 1)

def valid_aug(image_size):
    
    return Compose([Resize(*image_size),
                    Normalize()
                    ], p = 1)

def cutout_aug(h = 1, w = 1):

    return Compose([CoarseDropout(max_holes=2, min_holes = 1, 
                                    max_height=h, min_height = h,
                                    max_width=w, min_width = w, p = 0.7)], p = 1)

def apply_aug(aug, image):
    return aug(image=image)['image']    



class Transform:
    def __init__(self,
                train=False,
                mode = 'train',
                args = None,
                ):

        self.args = args

        self.train = train
        self.mode = mode

        if self.args.auto_aug:
            self.trans = AutoAugment()

    def __call__(self, example):
        
        if self.train:
            path, y = example
        else:
            path = example


        if self.mode == 'train':

            img = cv2.imread(self.args.path + path , 1)

            if self.args.use_cutout:
                h, w = img.shape
                img = apply_aug(cutout_aug(h = h//3, w = w//3), img)

            if self.args.auto_aug:
                # auto_aug
                img = Image.fromarray(img).convert("RGB")
                img = self.trans(img)
                img = np.array(img)

            img = apply_aug(train_aug(self.args.image_size), img)
            # cv2.imwrite('test2/' + path + '.jpg', img)
        else:

            img = cv2.imread(self.args.path + path , 1)

            img = apply_aug(valid_aug(self.args.image_size), img)

            # cv2.imwrite('test/' + path + '.jpg', img)

        img = np.moveaxis(img, -1, 0)  # conver to channel first, pytorch suck
        img = img.astype(np.float32)

        if self.train:
            y = y.astype(np.int64)
            return img, y
        else:
            return img


