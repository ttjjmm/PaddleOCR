import os
import torch
import argparse
import numpy as np
import yaml
from tqdm import tqdm
from model.arch import build_model
from model.losses import build_loss
from dataset import build_dataloader


import logging
logger = logging.getLogger(__name__)

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
    # m = build_model(config['Architecture'])
    # inp = torch.randn((8, 3, 320, 320))
    # out = m(inp)['maps']
    # print(out[:, 1, :, :].shape)

    # loader = build_dataloader(config, "Train", logger)
    return config


def main():
    args = parse_args()
    config = load_config(args.config)

    train_dataloader = build_dataloader(config, 'Train', logger)
    model = build_model(config['Architecture'])
    print(model)
    # optimizer =
    loss_class = build_loss(config['Loss'])
    model = model.to('cuda:0')



    for batch in tqdm(train_dataloader):
        imgs = batch[0].to('cuda:0')
        preds = model(imgs)
        loss = loss_class(preds, batch)['loss']
        loss.backward()

if __name__ == '__main__':
    main()
#     args = parse_args()
#     print(load_config(args.config))


