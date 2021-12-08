import numpy as np
import torch
import math
import copy
from PIL import Image
import cv2
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import Sequence

class Gridmask(object):
    def __init__(self,
                 use_h=True,
                 use_w=True,
                 rotate=1,
                 offset=False,
                 ratio=0.5,
                 mode=1,
                 prob=0.7,
                 upper_iter=360000):
        super(Gridmask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.prob = prob
        self.st_prob = prob
        self.upper_iter = upper_iter

    def __call__(self, x, curr_iter):
        self.prob = self.st_prob * min(1, 1.0 * curr_iter / self.upper_iter)
        if np.random.rand() > self.prob:
            return x
        h, w, _ = x.shape
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2
                    + w].astype(np.float32)

        if self.mode == 1:
            mask = 1 - mask
        mask = np.expand_dims(mask, axis=-1)
        if self.offset:
            offset = (2 * (np.random.rand(h, w) - 0.5)).astype(np.float32)
            x = (x * mask + offset * (1 - mask)).astype(x.dtype)
        else:
            x = (x * mask).astype(x.dtype)

        return x


class Cutmix(object):
    def __init__(self, alpha=1.5, beta=1.5):
        """
        CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features, see https://arxiv.org/abs/1905.04899
        Cutmix image and gt_bbbox/gt_score
        Args:
             alpha (float): alpha parameter of beta distribute
             beta (float): beta parameter of beta distribute
        """
        super(Cutmix, self).__init__()
        self.alpha = alpha
        self.beta = beta
        if self.alpha <= 0.0:
            raise ValueError("alpha shold be positive in {}".format(self))
        if self.beta <= 0.0:
            raise ValueError("beta shold be positive in {}".format(self))

    @staticmethod
    def apply_image(img1, img2, factor):
        """ _rand_bbox """
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        cut_rat = np.sqrt(1. - factor)

        cut_w = np.int(w * cut_rat)
        cut_h = np.int(h * cut_rat)

        # uniform
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w - 1)
        bby1 = np.clip(cy - cut_h // 2, 0, h - 1)
        bbx2 = np.clip(cx + cut_w // 2, 0, w - 1)
        bby2 = np.clip(cy + cut_h // 2, 0, h - 1)

        img_1_pad = np.zeros((h, w, img1.shape[2]), 'float32')
        img_1_pad[:img1.shape[0], :img1.shape[1], :] = \
            img1.astype('float32')
        img_2_pad = np.zeros((h, w, img2.shape[2]), 'float32')
        img_2_pad[:img2.shape[0], :img2.shape[1], :] = \
            img2.astype('float32')
        img_1_pad[bby1:bby2, bbx1:bbx2, :] = img_2_pad[bby1:bby2, bbx1:bbx2, :]
        return img_1_pad

    def __call__(self, sample, context=None):
        if not isinstance(sample, Sequence):
            return sample

        assert len(sample) == 2, 'cutmix need two samples'

        factor = np.random.beta(self.alpha, self.beta)
        factor = max(0.0, min(1.0, factor))
        if factor >= 1.0:
            return sample[0]
        if factor <= 0.0:
            return sample[1]
        img1 = sample[0]['image']
        img2 = sample[1]['image']
        img = self.apply_image(img1, img2, factor)
        gt_bbox1 = sample[0]['gt_bbox']
        gt_bbox2 = sample[1]['gt_bbox']
        gt_bbox = np.concatenate((gt_bbox1, gt_bbox2), axis=0)
        gt_class1 = sample[0]['gt_class']
        gt_class2 = sample[1]['gt_class']
        gt_class = np.concatenate((gt_class1, gt_class2), axis=0)
        gt_score1 = np.ones_like(sample[0]['gt_class'])
        gt_score2 = np.ones_like(sample[1]['gt_class'])
        gt_score = np.concatenate(
            (gt_score1 * factor, gt_score2 * (1. - factor)), axis=0)
        result = copy.deepcopy(sample[0])
        result['image'] = img
        result['gt_bbox'] = gt_bbox
        result['gt_score'] = gt_score
        result['gt_class'] = gt_class
        if 'is_crowd' in sample[0]:
            is_crowd1 = sample[0]['is_crowd']
            is_crowd2 = sample[1]['is_crowd']
            is_crowd = np.concatenate((is_crowd1, is_crowd2), axis=0)
            result['is_crowd'] = is_crowd
        if 'difficult' in sample[0]:
            is_difficult1 = sample[0]['difficult']
            is_difficult2 = sample[1]['difficult']
            is_difficult = np.concatenate(
                (is_difficult1, is_difficult2), axis=0)
            result['difficult'] = is_difficult
        return result




class AGSModule(nn.Module):
    """
    Special Attention Module
    """
    def __init__(self):
        super(AGSModule, self).__init__()



    def forward(self, x):
        pass    



# class EMA():
#     def __init__(self):
#         pass






if __name__ == '__main__':

    img = cv2.imread('/home/ubuntu/Documents/pycharm/PaddleOCR/samples/1.jpg')
    img = img / 255
    gridmask = Gridmask()
    out = gridmask(img, 200000)
    plt.imshow(out)
    plt.show()



    print(out.shape)


