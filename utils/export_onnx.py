import os
import yaml
import torch
import onnx
from pathlib import Path
from model.arch import build_model
import argparse

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=ROOT / 'config/ppocr_cls.yaml', help='configuration file path')
    parser.add_argument('--save_path', type=str, default=ROOT / 'onnx', help='onnx file save fold path')
    parser.add_argument('--simplify', action='store_true', default=True, help='onnx model simplify')
    opt = parser.parse_args()
    return opt


def export_onnx(args):
    model_type_name = args.cfg.name.split('.')[0] + '.onnx'
    save_onnx_path = Path(args.save_path) / model_type_name

    cfg = yaml.load(open(args.cfg, 'rb'), Loader=yaml.Loader)
    global_cfg = cfg['Global']
    device = global_cfg['device']
    ckpt = torch.load(ROOT / global_cfg['checkpoints'], map_location=lambda storage, loc: storage)
    model = build_model(cfg['Architecture'])
    model.load_state_dict(ckpt, strict=True)
    print('step1.Loaded pretrained model from path: {}'.format(global_cfg['checkpoints']))
    model = model.to(device).eval()
    in_h, in_w = global_cfg['image_shape'][1:]
    dummy = torch.autograd.Variable(torch.randn(1, 3, in_h, in_w)).to(device)

    torch.onnx.export(model, dummy, save_onnx_path,
                      verbose=True,
                      output_names=['preds'],
                      keep_initializers_as_inputs=True,
                      opset_version=11)
    print('step2. Converted pt to ONNX model: {}'.format(save_onnx_path))

    if args.simplify:
        try:
            from onnxsim import simplify
        except ImportError:
            raise ImportError('Please run "pip install onnx-simplifier" to install onnxsim pakage!')
        onnx_model = onnx.load(save_onnx_path)
        model, check = simplify(onnx_model)
        assert check, 'Simplified ONNX model could not be validated'
        # par_dir, file_name = os.path.split(args.output)
        # sim_path = os.path.join(par_dir, '{}_sim.onnx'.format(file_name.split('.')[0]))
        onnx.save(model, save_onnx_path)
        print('step3: Successfully simplified ONNX model: {}'.format(save_onnx_path))


if __name__ == '__main__':
    args = parse_args()
    export_onnx(args)


