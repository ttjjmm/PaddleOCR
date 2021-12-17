import torch
import torch.nn as nn



class CTCLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')

    def forward(self, predicts, batch):
        predicts = predicts.transpose((1, 0, 2))
        N, B, _ = predicts.shape
        preds_lengths = torch.tensor([N] * B, dtype=torch.int64, device=predicts.device)
        labels = batch[1].to(torch.int32)
        label_lengths = batch[2].to(torch.int64)
        loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)

        loss = loss.mean()
        return {'loss': loss}


if __name__ == '__main__':
    print(__file__)




