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


def main(batch_size = 100, weight_decay=1e-4, num_epochs=1, name='default', loss_idx=0):
    if os.path.exists('log/'+name):
        shutil.rmtree('log/'+name)
    os.makedirs(osp.join('log',name,'models'))

    print('batch_size:', batch_size)
    print('num_epochs:', num_epochs)

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'


    batches_per_epoch = 50000//batch_size


    torch.manual_seed(2)

    model = TwoHeadedNet()
    model = model.to(device)

    loss_net = LossNet()
    loss_net = loss_net.to(device)

    #optimizer = optim.RMSprop(model.parameters())
    optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    loss_optimizer = optim.Adam(loss_net.parameters(), weight_decay=weight_decay)
    cross_ent_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()
    iou_fn = IoU()

    val = ValTestDataset(mode='val')
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=5)

    train = TrainDataset()
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=5)


    val_results = []
    train_results = []

    for epoch in range(num_epochs):
        model = model.to(device)
        loss_net = loss_net.to(device)
        print('epoch:', epoch)
        epoch_results = []

        for i,batch in enumerate(train_loader):
            images, labels, bbs = batch

            images, labels, bbs = images.to(device), labels.to(device), bbs.to(device)
            
            optimizer.zero_grad()
            scores, bb_preds = model(images)

            # print(torch.mean(bb_preds, dim=0).data)

            cross_ent_loss = cross_ent_loss_fn(scores, labels)
            mse_loss = mse_loss_fn(bb_preds, bbs)
            iou = iou_fn(bb_preds, bbs)
            acc = accuracy(scores, labels)
            pk = precision_at_k(scores, labels)


            train_results.append([acc, pk, float(iou), float(mse_loss), float(cross_ent_loss)])

            # print('cross ent loss:', float(cross_ent_loss))
            # print('mse loss:', float(mse_loss))
            # print('iou:', float(iou))
            # print('accuracy:', acc)
            # print('precision at 5:', pk)

            if loss_idx < 5:
                loss1 = cross_ent_loss
                loss2 = -iou
                loss3 = mse_loss
                loss4 = cross_ent_loss - iou
                loss5 = cross_ent_loss + mse_loss/200

                losses = [loss1, loss2, loss3, loss4, loss5]

                loss = losses[loss_idx]


            # if using the learned loss
            if loss_idx == 5:
                pred_iou = loss_net(bbs, bb_preds)
                print('iou:', iou)
                print('pred_iou:', pred_iou)
                loss_net_loss = mse_loss_fn(iou, pred_iou)
                print('loss net loss:', float(loss_net_loss))
                loss_net_loss.backward(retain_graph=True)
                loss_optimizer.step()
                loss = -torch.sum(pred_iou)

            loss.backward()
            optimizer.step()

        epoch_results = train_results[epoch*batches_per_epoch:(epoch+1)*batches_per_epoch]
        epoch_results = np.mean(epoch_results, axis=0)
        acc, pk, iou, mse, ce = epoch_results
        print('train:', 'acc:', round(acc,3), 'p5:', round(pk,3), 'iou:', round(iou,3), 'mse:', round(mse,3), 'ce:', round(ce,3))



        acc, iou, mse, ce, pk = validate(val_loader, model)
        print('val:  ', 'acc:', round(acc,3), 'p5:', round(pk,3), 'iou:', round(iou,3), 'mse:', round(mse,3), 'ce:', round(ce,3))
        val_results.append([acc, pk, iou, mse, ce])

        torch.save(model.cpu(), osp.join('log', name, 'models','model_'+str(epoch)))


    train_results = np.array(train_results)
    val_loader = np.array(val_results)

    np.savetxt(osp.join('log',name,'train.csv'),train_results, delimiter=',')
    np.savetxt(osp.join('log',name,'val.csv'),val_results, delimiter=',')


if __name__ == '__main__':
    cfg_fname = sys.argv[1]
    with open(cfg_fname) as cfg_file:
        args = json.load(cfg_file)
    main(**args)
