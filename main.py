import torch
import yaml
from model.arch import build_model
from model.backbones.rec_mbnetv3 import MobileNetV3
from collections import OrderedDict

if __name__ == '__main__':

    ckpt = torch.load('/home/tjm/Documents/python/pycharmProjects/PaddleOCR/ch_ppocr_mobile_v2.0_cls_train.pth')
    for idx, (k, v) in enumerate(ckpt.items()):
        print(k, v.shape)
    exit(11)
    # print(len(ckpt))
    # exit(11)

    file_path = '/config/ppocr_det.yaml'
    cfg = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    model = MobileNetV3(scale=0.35) #build_model(cfg['Architecture'])
    new_stat = OrderedDict()
    for idx, (k, v) in enumerate(model.state_dict().items()):
        key_points = k.split('.')
        if key_points[-1] == 'num_batches_tracked':
            continue
        if key_points[-1] == 'running_mean':
            new_key = k.replace('running_mean', '_mean')
            # tail = '_mean'
        elif key_points[-1] == 'running_var':
            new_key = k.replace('running_var', '_variance')
            # tail = '_variance'
        else:
            new_key = k
        new_stat[k] = ckpt[new_key]
        print(idx, new_key, v.shape)
    model.load_state_dict(new_stat, strict=True)
    # torch.save(new_stat, './ch_ppocr_mobile_v2.0_det_train.pt')