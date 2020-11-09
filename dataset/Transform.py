from albumentations import Compose, ShiftScaleRotate,Resize, Rotate, Normalize, OneOf, ElasticTransform, GridDistortion, OpticalDistortion, CoarseDropout
import numpy as np
from skimage.transform import AffineTransform, warp
import albumentations as A
import cv2
from dataset.AutoAugment import *
from PIL import Image, ImageEnhance, ImageOps

def train_aug(image_size, use_cutout = False):
    if use_cutout:
        return Compose([Resize(*image_size),
                        ShiftScaleRotate(rotate_limit=10),
                        Normalize()
                        ], 
                        p = 1)

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

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def cut_out(img0):
    HEIGHT = 137
    WIDTH = 236
    SIZE = 128
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    # xmin = xmin - 13 if (xmin > 13) else 0
    # ymin = ymin - 10 if (ymin > 10) else 0
    # xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    # ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    # img = 255 - img
    h, w  = img.shape
    img0[ymin:ymax,xmin:xmax] = apply_aug(cutout_aug(h = h//3, w = w//3), img)
    return img0
    # return img

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

            if self.args.use_cutout:
                img = cv2.imread(self.args.path + path + '.png', 0)

                img = (img*(255.0/img.max())).astype(np.uint8)
                img = cut_out(img)
                img = cv2.merge((img, img, img))
            else:
                img = cv2.imread(self.args.path + path + '.png', 1)


            if self.args.auto_aug:
                # auto_aug
                img = Image.fromarray(img).convert("RGB")
                img = self.trans(img)
                img = np.array(img)

            img = apply_aug(train_aug(self.args.image_size, self.args.use_cutout), img)
            cv2.imwrite('test2/' + path + '.jpg', img)
        else:

            img = cv2.imread(self.args.path + path + '.png', 1)

            img = apply_aug(valid_aug(self.args.image_size), img)

            cv2.imwrite('test/' + path + '.jpg', img)

        img = np.moveaxis(img, -1, 0)  # conver to channel first, pytorch suck
        img = img.astype(np.float32)

        if self.train:
            y = y.astype(np.int64)
            return img, y
        else:
            return img


