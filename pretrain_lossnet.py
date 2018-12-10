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

    mse_loss_fn = nn.MSELoss()

    for i in range(500):










if __name__ == '__main__':
    main()