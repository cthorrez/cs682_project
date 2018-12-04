import torch
from models import IoU




def main():
    pred = torch.FloatTensor([[2,2,4,5],[1,1,3,6],[2,2,4,5]])
    gt = torch.FloatTensor([[3,4,6,6],[2,2,5,5],[3,4,6,6]])



    iou = IoU()

    print(iou(pred,gt))



if __name__ == '__main__':
    main()