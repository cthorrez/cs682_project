import torch
from torch import nn


def accuracy(scores, labels):
    preds = torch.argmax(scores, dim=1)
    correct = float(torch.sum(preds==labels))
    total = float(len(preds))
    return correct/total

def precision_at_k(scores, labels, k=5):
    _, indices = torch.topk(scores, dim=1, k=k)
    labels = labels.unsqueeze(1)
    correct = float(torch.sum(indices==labels))
    total = float(len(scores))
    return correct/total



# takes list of upper lefts and bottom rights, each is Bx2, B is batch size
# returns Bx1 tensor of areas
def area(uls, brs):
    diffs = (brs-uls)
    diffs[diffs<0] = 0
    return diffs[:,0] * diffs[:,1]

class IoU(nn.Module):
    def __init__(self):
        super(IoU, self).__init__()

    def forward(self, pred, gt, eps=1e-6):
        bs = pred.shape[0]

        gt1 = gt.view(bs,2,-1)
        pred1 = pred.view(bs,2,-1)

        ul,_ = torch.max( torch.cat([gt1[:,0,:,None], pred1[:,0,:,None]],dim=2) ,dim=2)
        br,_ = torch.min( torch.cat([gt1[:,1,:,None], pred1[:,1,:,None]],dim=2), dim=2)

        inter = area(ul, br)
        
        area_pred = area(pred[:,:2], pred[:,2:])
        area_gt = area(gt[:,:2], gt[:,2:])
        union = area_pred + area_gt - inter

        iou = (torch.sum(inter)+eps)/(torch.sum(union)+eps)
        return iou    






class TwoHeadedNet(nn.Module):
    def __init__(self):
        super(TwoHeadedNet, self).__init__()
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(3,32,3, padding=1)
        self.conv2 = nn.Conv2d(32,64,3, padding=1)
        self.conv3 = nn.Conv2d(64,128,3, padding=1)
        self.fc_clf = nn.Linear(8192,200)
        self.fc_bb = nn.Linear(8192,4)

    def forward(self, x):
        bs = x.shape[0]
        x = self.conv1(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.relu(x)
        x = x.view(bs,-1)
        scores = self.fc_clf(x)
        bb = self.fc_bb(x)
        bb = self.sigmoid(bb)*64
        return scores, bb


class LossNet(nn.Module):
    def __init__(self):
        super(LossNet, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64,1)

    def forward(self,bbs, bb_preds):
        x = torch.cat([bbs, bb_preds], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
