import sys
import shutil
import os
import os.path as osp
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils import TrainDataset,ValTestDataset, validate
from models import TwoHeadedNet, LossNet, IoU, accuracy, precision_at_k



def main():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'

    loss_net = LossNet()
    loss_net = loss_net.to(device)

    iou_fn = IoU()
    mse_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-3)

    train = TrainDataset
    train_loader = DataLoader(train, batch_size=250, shuffle=True, num_workers=10)

    num_epochs = 3:
    for epoch in range(num_epochs):
        print('epoch:', epoch)

        for batch in train_loader:

            optimizer

            images, labels, bbs = batch
            images, labels, bbs = images.to(device), labels.to(device), bbs.to(device)

            preds = torch.tensor([10.0,10.0,50.0,50.0])
            noise = torch.randn_like(bbs) * 25
            preds = (preds + noise).to(device)

            iou = IoU(bbs, preds)
            pred_iou = loss_net(bbs, preds)

            loss = mse_loss_fn(iou, pred_iou)
            print('mse:', float(loss))

            loss.backward()
            optimizer.step()

    torch.save(model, 'pretrained_loss_net')


if __name__ == '__main__':
    main()