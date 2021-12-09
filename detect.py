import torch
import yaml
import cv2
import numpy as np

from model.arch import build_model
from model.postprocess import build_postprocess
from dataset.augment.operators import ClsResizeImg

import matplotlib.pyplot as plt
from torchvision import transforms



class BaseDetector(object):
    def __init__(self, cfg):
        self.device = cfg['Global']['device']
        ckpt = torch.load(cfg['Global']['checkpoints'], map_location=lambda storage, loc: storage)
        model = build_model(cfg['Architecture'])
        model.load_state_dict(ckpt)
        print('loaded pretrained model from path: {}'.format(cfg['Global']['checkpoints']))
        self.model = model.to(self.device).eval()

        post_cfg = cfg['PostProcess']
        self.postprocess = build_postprocess(post_cfg)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    def inference(self, path):
        pass


class OCRTextDetctor(BaseDetector):
    def __init__(self, cfg):
        super(OCRTextDetctor, self).__init__(cfg)
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        # ])


    def inference(self, path):
        raw_image = cv2.imread(path)
        src_h, src_w, _ = raw_image.shape
        ratio_h  = float(1024) / src_h
        ratio_w = float(480) / src_w

        shape_list = np.array([src_h, src_w, ratio_h, ratio_w])

        image = cv2.resize(raw_image, (480, 1024))
        image = self.transform(image).unsqueeze(0)

        image = image.to(self.device)
        with torch.no_grad():
            pred = self.model(image)

        map = pred['maps'].cpu().squeeze(0).squeeze(0).numpy()
        print(map.shape, map.max(), map.min())
        plt.imshow(map)
        plt.show()

        shape_list = np.expand_dims(shape_list, 0)
        det_bboxes = self.postprocess(pred, shape_list)
        points = det_bboxes[0]['points']
        points = np.reshape(points, (points.shape[0], -1))[:, [0, 1, 4, 5]]
        for dets in points:
            cv2.rectangle(raw_image, (dets[0], dets[1]), (dets[2], dets[3]), (255, 0, 0), cv2.LINE_4)
        plt.imshow(raw_image)
        plt.show()



class OCRTextClassifier(BaseDetector):
    def __init__(self, cfg):
        super(OCRTextClassifier, self).__init__(cfg)
        self.img_shape = cfg['Global']['image_shape']
        self.resize_fn = ClsResizeImg(self.img_shape)

    def inference(self, path):
        img = cv2.imread(path)
        img = self.resize_fn({'image': img})
        img['image'] = np.transpose(img['image'], (1, 2, 0))
        image = self.transform(img['image']).unsqueeze(0)
        image = image.to(self.device)
        with torch.no_grad():
            pred = self.model(image)

        label_idx = torch.argmax(pred, dim=-1).cpu().item()
        print(label_idx)


class OCRTextRecognizer(BaseDetector):
    def __init__(self, cfg):
        super(OCRTextRecognizer, self).__init__(cfg)
        print(self.model)



if __name__ == '__main__':
    file_path = './config/ppocr_rec.yaml'
    cfgs = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    d = OCRTextRecognizer(cfgs)
    # d.inference('/home/ubuntu/Documents/pycharm/PaddleOCR/samples/word_201.png')
    # d.inference('/home/ubuntu/Documents/pycharm/PaddleOCR/samples/1.jpg')






