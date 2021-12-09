import torch
import yaml
import cv2
import numpy as np
from model.arch import build_model
# from model.postprocess.db_postprocess import DBPostProcess
import matplotlib.pyplot as plt
from torchvision import transforms

import paddle.fluid as fluid
from collections import OrderedDict



class OCRDetector(object):
    def __init__(self, cfg):
        self.device = cfg['Global']['device']
        # ckpt = torch.load(cfg['Global']['checkpoints'], map_location=lambda storage, loc: storage)
        model = build_model(cfg['Architecture'])
        model.eval()

        new_state_dict = OrderedDict()

        weights_path = '/home/ubuntu/Documents/pycharm/PaddleOCR/weights/ch_ppocr_mobile_v2.0_det_train/best_accuracy.pdparams'
        with fluid.dygraph.guard():
            para_state_dict, opti_state_dict = fluid.load_dygraph(weights_path)


        for k, v in model.state_dict().items():
            if k.endswith('num_batches_tracked'): continue
            elif k.endswith('running_mean'):
                ppname = k.replace('running_mean', '_mean')
            elif k.endswith('running_var'):
                ppname = k.replace('running_var', '_variance')
            elif k.endswith('weight') or k.endswith('bias'):
                ppname = k
            else:
                print('Redundance: {}'.format(k))
                raise RuntimeError

            if ppname.endswith('fc.weight'):
                new_state_dict[k] = torch.FloatTensor(para_state_dict[ppname]).T
            else:
                new_state_dict[k] = torch.FloatTensor(para_state_dict[ppname])

            print(k, v.shape)

        model.load_state_dict(new_state_dict, strict=True)
        torch.save(model.state_dict(), './ch_ppocr_mobile_v2.0_det_train.pt')



        self.model = model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # self.postprocess = DBPostProcess(cfg['PostProcess'])
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
    file_path = './config/ppocr_det.yaml'
    cfg = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    d = OCRDetector(cfg)
    # d.inference('/home/tjm/Documents/python/pycharmProjects/PaddleOCR/samples/1.jpg')






