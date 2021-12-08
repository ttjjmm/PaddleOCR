import paddle.fluid as fluid
from collections import OrderedDict
import torch

if __name__ == '__main__':
    state = fluid.io.load_program_state('/home/ubuntu/Documents/pycharm/PaddleOCR/weights/ch_ppocr_mobile_v2.0_cls_train/best_accuracy.pdparams')
    new_state = OrderedDict()
    for k, v in state.items():
        # if
        if not isinstance(v, dict):
            new_state[k] = torch.FloatTensor(v)
            print(k, v.shape)
    # torch.save(new_state, "../weights/ch_ppocr_mobile_v2.0_cls_train.pth")