import torch


class ClsPostProcess(object):
    def __init__(self, label_list, **kwargs):
        self.label_list = label_list

    def __call__(self, preds, label=None,  **kwargs):
        # if isinstance(preds, torch.tensor):
        preds = preds.cpu().numpy()
        pred_idxs = preds.argmax(axis=1)
        decod_out = [
            (self.label_list[idx], preds[i, idx]) for i, idx in enumerate(pred_idxs)
        ]
        if label is None:
            return decod_out
        label = [(self.label_list[idx], 1.0) for idx in label]
        return  decod_out, label









