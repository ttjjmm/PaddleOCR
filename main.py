import yaml



if __name__ == '__main__':
    file_path = '/home/tjm/Documents/python/pycharmProjects/PaddleOCR/config/ppocr_mb.yaml'
    cfg = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    print(cfg)