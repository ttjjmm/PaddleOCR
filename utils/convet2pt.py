import paddle.fluid as fluid
from collections import OrderedDict
import torch

import yaml
import os

from model.arch import build_model

def ocr_converter(yaml_path, src_weight, dst_weight=None):
        # super(OCRConverter, self).__init__()
        cfg = yaml.load(open(yaml_path, 'rb'), Loader=yaml.Loader)
        model = build_model(cfg['Architecture'])
        model.eval()
        # exit(1)
        new_state_dict = OrderedDict()

        # weights_path = '/home/ubuntu/Documents/pycharm/PaddleOCR/weights/ch_ppocr_mobile_v2.0_det_train/best_accuracy.pdparams'
        with fluid.dygraph.guard():
            para_state_dict, opti_state_dict = fluid.load_dygraph(src_weight)

        # for k, v in para_state_dict.items():
        #     print(k, v.shape)
        # print(len(model.state_dict()))

        for k, v in model.state_dict().items():
            if k.endswith('num_batches_tracked'):
                continue
            elif k.endswith('running_mean'):
                ppname = k.replace('running_mean', '_mean')
            elif k.endswith('running_var'):
                ppname = k.replace('running_var', '_variance')
            elif k.endswith('weight') or k.endswith('bias'):
                ppname = k
            elif 'lstm' in k:
                ppname = k
            else:
                print('Redundance: {}'.format(k))
                raise RuntimeError

            if ppname.endswith('fc1.weight') or ppname.endswith('fc2.weight'):
                new_state_dict[k] = torch.FloatTensor(para_state_dict['student2_model.' + ppname]).T
            else:
                new_state_dict[k] = torch.FloatTensor(para_state_dict['student2_model.'+ ppname])
            print(k, v.shape)

        model.load_state_dict(new_state_dict, strict=True)

        if dst_weight is None:
            save_name = src_weight.split('/')[-2]
            save_path = os.path.join('../weights', '{}.pt'.format(save_name))
        else:
            save_path = dst_weight

        if os.path.exists(save_path):
            os.remove(save_path)

        torch.save(model.state_dict(), save_path)




if __name__ == '__main__':
    ocr_converter('/home/ubuntu/Documents/pycharm/PaddleOCR/config/ppocr_det.yaml',
                  '/home/ubuntu/Documents/pycharm/PaddleOCR/weights/ch_PP-OCRv2_det_distill_train/ch_PP-OCRv2_det_distill_train/best_accuracy.pdparams')