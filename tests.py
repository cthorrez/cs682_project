import numpy as np
import torch
from models import IoU
from utils import show_img_bbs, TrainDataset
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_pil_image




def main():
    # pred = torch.FloatTensor([[2,2,4,5],[1,1,3,6],[2,2,4,5]])
    # gt = torch.FloatTensor([[3,4,6,6],[2,2,5,5],[3,4,6,6]])

    # iou = IoU()
    # print(iou(pred,gt))

    train = TrainDataset(normalize=False)

    img, label, bbox = train[45000]

    pred = torch.tensor([5,6,55,47])

    show_img_bbs(img, pred, bbox)





if __name__ == '__main__':
    main()