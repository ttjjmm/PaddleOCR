import torch
import yaml
import cv2
import numpy as np
from model.arch import build_model

class OCRDetector(object):
    def __init__(self, cfg):
        self.device = cfg['Global']['device']
        ckpt = torch.load(cfg['Global']['checkpoints'], map_location=lambda storage, loc: storage)
        model = build_model(cfg['Architecture'])
        model.load_state_dict(ckpt, strict=True)
        self.model = model.to(self.device).eval()

    def inference(self, path):
        image = cv2.imread(path)















if __name__ == '__main__':
    file_path = '/home/tjm/Documents/python/pycharmProjects/PaddleOCR/config/ppocr_mb.yaml'
    cfg = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    d = OCRDetector(cfg)







