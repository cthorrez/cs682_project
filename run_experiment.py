import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils import TrainDataset,ValTestDataset, validate
from models import TwoHeadedNet, IoU, accuracy





def main():
    torch.manual_seed(2)

    model = TwoHeadedNet()
    #optimizer = optim.RMSprop(model.parameters())
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-3)
    cross_ent_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()
    iou_fn = IoU()

    val = ValTestDataset(mode='val')
    val_loader = DataLoader(val, batch_size=10, shuffle=False)

    train = TrainDataset()
    train_loader = DataLoader(train, batch_size=25, shuffle=True)


    num_epochs = 3
    results = []

    for epoch in range(num_epochs):

        for i,batch in enumerate(train_loader):
            images, labels, bbs = batch
            
            optimizer.zero_grad()
            scores, bb_preds = model(images)

            print(torch.mean(bb_preds, dim=0).data)

            cross_ent_loss = cross_ent_loss_fn(scores, labels)
            mse_loss = mse_loss_fn(bb_preds, bbs)
            iou = iou_fn(bb_preds, bbs)
            acc = accuracy(scores, labels)

            print('cross ent loss:', float(cross_ent_loss))
            print('mse loss:', float(mse_loss))
            print('iou:', float(iou))
            print('accuracy:', float(acc))

            loss = cross_ent_loss - iou
            loss.backward()
            optimizer.step()

        acc, iou, mse, ce = validate(val_loader, model)
        results.append([epoch, acc, iou, mse, ce])


    name = 'model_iou_wd0.001'

    results = np.array(results)
    np.savetxt(osp.join('log',name+'.csv'),results, delimiter=',')
    torch.save(model, osp.join('models',name+'.csv'))



if __name__ == '__main__':
    main()
