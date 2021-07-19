import os
import torch
import argparse
import numpy as np
import yaml

from model.arch import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/home/tjm/Documents/python/pycharmProjects/PaddleOCR/config/ppocr_mb.yaml',
                        help='train config file path')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')
    args = parser.parse_args()
    return args


def load_config(cfg_path):
    config = yaml.load(open(cfg_path, 'rb'), Loader=yaml.Loader)
    m = build_model(config['Architecture'])
    inp = torch.randn((1, 3, 320, 320))
    print(m(inp)['maps'].shape)

    return config












if __name__ == '__main__':
    args = parse_args()
    print(load_config(args.config))


