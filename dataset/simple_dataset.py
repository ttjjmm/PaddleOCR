import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
from dataset.augment import transform, create_operators
from icecream import ic

class SimpleDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None):
        super(SimpleDataSet, self).__init__()
        self.logger = logger
        # mode -> [train, eval]
        self.mode = mode.lower()

        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']

        self.delimiter = dataset_config.get('delimiter', '\t')

        label_file_list = dataset_config.pop('label_file_list')
        data_source_num = len(label_file_list)
        ratio_list = dataset_config.get("ratio_list", [1.0])
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)

        assert len(ratio_list) == data_source_num, "The length of ratio_list should be the same as the file_list."

        self.data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']

        self.seed = seed
        logger.info("Initialize indexs of datasets:%s" % label_file_list)
        self.data_lines = self.get_image_info_list(label_file_list, ratio_list)

        self.data_idx_order_list = list(range(len(self.data_lines)))
        if self.mode == "train" and self.do_shuffle:
            self.shuffle_data_random()

        self.ops = create_operators(dataset_config['transforms'], global_config)

    def get_image_info_list(self, file_list, ratio_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for idx, file in enumerate(file_list):
            with open(file, "rb") as f:
                lines = f.readlines()
                if self.mode == "train" or ratio_list[idx] < 1.0:
                    random.seed(self.seed)
                    lines = random.sample(lines, round(len(lines) * ratio_list[idx]))
                data_lines.extend(lines)
        return data_lines

    def shuffle_data_random(self):
        random.seed(self.seed)
        random.shuffle(self.data_lines)
        return

    def __getitem__(self, idx):
        file_idx = self.data_idx_order_list[idx]
        data_line = self.data_lines[file_idx]
        try:
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            label = substr[1]
            # print(label, len(label))
            img_path = os.path.join(self.data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            # print(data['img_path'])
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            outs = transform(data, self.ops)
            # print(outs['image'].shape)
        except Exception as e:
            self.logger.error("When parsing line {}, error happened with msg: {}".format(data_line, e))
            outs = None
        if outs is None:
            # during evaluation, we should fix the idx to get same results for many times of evaluation.
            rnd_idx = np.random.randint(self.__len__()) if self.mode == "train" else (idx + 1) % self.__len__()
            return self.__getitem__(rnd_idx)

        return outs

    def __len__(self):
        return len(self.data_idx_order_list)

    @staticmethod
    def collate_fn(batch):
        data_list = []
        data = []
        for idx, each in enumerate(batch):
            for idx2, item in enumerate(each):
                if idx == 0:
                    data_list.append([item])
                else:
                    data_list[idx2].append(item)
        for each in data_list:
            data.append(torch.stack(each, dim=0))
        return data


if __name__ == '__main__':
    import logging
    import matplotlib.pyplot as plt
    logger = logging.getLogger(__name__)
    import yaml
    cfg = yaml.load(open('/config/ppocr_det.yaml', 'rb'), Loader=yaml.Loader)
    print(cfg)
    ds = SimpleDataSet(cfg, 'Train', logger)
    print(len(ds))
    z = ds[10]
    print(z[1].shape)
    for i in z:
        print(i.shape)
    img = np.expand_dims(z[1].numpy(), axis=-1)
    plt.imshow(img)
    plt.show()












