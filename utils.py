import numpy as np
import torch
import torch.nn.functional as F
import os
import os.path as osp
from itertools import chain
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tf
from models import IoU, accuracy, precision_at_k
import cv2
import matplotlib
import matplotlib.pyplot as plt


def show_img_bbs(img, pred, actual):

    img = tf.to_pil_image(img)

    fig,ax = plt.subplots(1)
    ax.imshow(img)

    ll = pred[:2]
    w = pred[2]- ll[0]
    h = pred[3] - ll[1]
    rect = matplotlib.patches.Rectangle(ll,w,h,linewidth=3,edgecolor='r',facecolor='none')
    ax.add_patch(rect)

    ll = actual[:2]
    w = actual[2] - ll[0]
    h = actual[3] - ll[1]
    rect = matplotlib.patches.Rectangle(ll,w,h,linewidth=3,edgecolor='b',facecolor='none')
    ax.add_patch(rect)

    plt.show()



def my_mse(gt, preds):
    return torch.mean(torch.pow(gt-preds,2))

def validate(data_loader, model):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'
    mse, ce, iou, acc, pk = [], [], [], [], []
    iou_fn = IoU()
    # mse_loss_fn = torch.nn.MSELoss(reduction='elementwise_mean')
    mse_loss_fn = my_mse

    for batch in data_loader:
        imgs, labels, bbs = batch
        imgs, labels, bbs = imgs.to(device), labels.to(device), bbs.to(device)

        scores, bb_preds = model(imgs)
    
        tmp_acc = float(accuracy(scores,labels))
        # print(bbs)
        # print(bb_preds)

        tmp_iou= float(torch.mean(iou_fn(bbs, bb_preds)))
        tmp_mse = float(mse_loss_fn(bbs,bb_preds))
        tmp_ce = float(F.cross_entropy(scores,labels))
        tmp_pk = float(precision_at_k(scores, labels))

        mse.append(tmp_mse)
        acc.append(tmp_acc)
        ce.append(tmp_ce)
        iou.append(tmp_iou)
        pk.append(tmp_pk)

        #print('val:', 'acc:', tmp_acc, 'iou:', tmp_iou, 'mse:', tmp_mse, 'ce:', tmp_ce)
    mse = np.mean(mse)
    ce = np.mean(ce)
    iou = np.mean(iou)
    acc = np.mean(acc)
    pk = np.mean(pk)

    return acc, iou, mse, ce ,pk

class TrainDataset(Dataset):
    def __init__(self, seed=0, normalize=True):
        self.data_dir = 'data/train'
        self.categories = sorted(os.listdir(self.data_dir))
        self.cat2idx = {self.categories[i]:i for i in range(len(self.categories))}
        self.normalize = normalize
        
        self.fname2bbox = {}
        for cat in self.categories:
            bbox_fname = osp.join(self.data_dir,cat,cat+'_boxes.txt')
            bboxes = np.genfromtxt(bbox_fname, delimiter='\t', dtype=str)
            for bbox in bboxes:
                pic_fname, y_min, x_min, y_max, x_max = bbox[0], int(bbox[1]), int(bbox[2]), int(bbox[3]), int(bbox[4])
                self.fname2bbox.update({osp.join(self.data_dir,cat,'images',pic_fname) : torch.FloatTensor([y_min, x_min, y_max, x_max])})
        self.numel = len(self.fname2bbox)
            
    def __len__(self):
        return self.numel

    def __getitem__(self, idx):
        fname = list(self.fname2bbox.keys())[idx]
        category = fname.split('/')[2]

        image = cv2.imread(fname, cv2.IMREAD_COLOR)
        image = tf.to_tensor(image)

        if self.normalize:
            image = tf.normalize(image, mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        label = self.cat2idx[category]
        bbox = self.fname2bbox[fname]
        bbox = torch.FloatTensor(bbox)

        return image, label, bbox

class ValTestDataset(Dataset):
    def __init__(self, mode, seed=0, normalize=True):
        if mode == 'val':
            low, high = 0,5000
        elif mode == 'test':
            low,high = 5000,10000
        else:
            print('mode must be val or test')
            exit(1)

        self.normalize = normalize

        matrix = np.genfromtxt('data/val/val_annotations.txt', delimiter='\t', dtype=str)
        self.strings = matrix[:,:2]
        self.bbs = matrix[:,2:].astype(float)
        self.categories = sorted(np.unique(self.strings[:,1]))
        self.cat2idx = {self.categories[i]:i for i in range(len(self.categories))}

        self.strings = self.strings[low:high]
        self.bbs = self.bbs[low:high]
        self.numel = len(self.bbs)
                    
    def __len__(self):
        return self.numel

    def __getitem__(self, idx):
        fname = 'data/val/images/'+self.strings[idx,0]
        label = self.cat2idx[self.strings[idx,1]]

        image = cv2.imread(fname, cv2.IMREAD_COLOR)
        image = tf.to_tensor(image)
        if self.normalize:
            image = tf.normalize(image, mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        bbox = torch.FloatTensor(self.bbs[idx,:])
        return image, label, bbox