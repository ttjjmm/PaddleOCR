import torch
import yaml
import cv2
import numpy as np
from model.arch import build_model
from model.postprocess.db_postprocess import DBPostProcess
import matplotlib.pyplot as plt
from torchvision import transforms



class OCRDetector(object):
    def __init__(self, cfg):
        self.device = cfg['Global']['device']
        ckpt = torch.load(cfg['Global']['checkpoints'], map_location=lambda storage, loc: storage)
        model = build_model(cfg['Architecture'])
        model.load_state_dict(ckpt, strict=True)
        self.model = model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.postprocess = DBPostProcess(cfg['PostProcess'])
        # for k, v in model.state_dict().items():
        #     print(k, v.shape)


    def inference(self, path):
        image = cv2.imread(path)
        image = cv2.resize(image, (480, 1024))
        image = self.transform(image).unsqueeze(0)

        image = image.to(self.device)
        with torch.no_grad():
            pred = self.model(image)

        map = pred['maps'].cpu().squeeze(0).squeeze(0).numpy()
        print(map.shape, map.max(), map.min())
        plt.imshow(map)
        plt.show()



if __name__ == '__main__':
    file_path = '/config/ppocr_det.yaml'
    cfg = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    d = OCRDetector(cfg)
    d.inference('/home/tjm/Documents/python/pycharmProjects/PaddleOCR/samples/1.jpg')






