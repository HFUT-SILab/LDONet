import os
import argparse
import time
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from models.LDONet_T import LDONet_T

plt.switch_backend('agg')

from torch.optim import lr_scheduler
from models.dataset import MyDataset


def fit(epoch, model, data_loader, phase='training', optimizer=None, criterion=None):
    if phase != 'training' and phase != 'testing':
        raise TypeError('input error!')

    if phase == 'training':
        model.train()
    else:
        model.eval()

    running_loss = 0
    running_correct = 0

    for batch_id, (datas, target) in enumerate(data_loader):
        datas = datas.cuda()
        target = target.cuda()
        batch_size = datas.size(0)

        if phase == 'training':
            optimizer.zero_grad()
            output, _, _ = model(datas, target)
        else:
            with torch.no_grad():
                output, _, _ = model(datas, None)

        ce = criterion(output, target)
        loss = ce
        running_loss += loss.item() * batch_size

        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum().item()

        if phase == 'training':
            loss.backward(retain_graph=None)
            optimizer.step()

    total = len(data_loader.dataset)
    loss = running_loss / total
    accuracy = (100.0 * running_correct) / total

    if epoch % 10 == 0:
        print('epoch %d: \t%s loss is \t%7.5f ;\t%s accuracy is \t%d/%d \t%7.3f%%' % (
            epoch, phase, loss, phase, running_correct, total, accuracy))

    return loss, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LDONet-T for Palmprint Recognition"
    )

    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epoch_num", type=int, default=200)
    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument("--id_num", type=int, default=250,
                        help="IITD: 460 KTU: 145 Tongji: 600 REST: 358 XJTU: 200 POLYU 378 Multi-Spec 500 IITD_Right 230 Tongji_LR 300")
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--save_interval", type=int, default=2)

    parser.add_argument("--train_set_file", type=str, default='../dataset/train_Blue_linux.txt')

    parser.add_argument("--des_path", type=str, default='../results/Blue/LDONet_T/checkpoint/')
    parser.add_argument("--path_rst", type=str, default='../results/Blue/LDONet_T/rst_test/')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    batch_size = args.batch_size
    epoch_num = args.epoch_num
    num_classes = args.id_num

    print('tempture:', args.temp)

    des_path = args.des_path
    path_rst = args.path_rst

    if not os.path.exists(des_path):
        os.makedirs(des_path)
    if not os.path.exists(path_rst):
        os.makedirs(path_rst)

    train_set_file = args.train_set_file

    trainset = MyDataset(txt=train_set_file, transforms=None, train=True, imside=128, outchannels=1)
    data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=4, shuffle=True)

    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
    print('------Init Model------')

    net = LDONet_T(label_num=num_classes)
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(current_epoch):
        # Linear warmup + cosine decay improves stability and generalization.
        if current_epoch < args.warmup_epochs:
            return float(current_epoch + 1) / float(max(1, args.warmup_epochs))

        progress = (current_epoch - args.warmup_epochs) / float(max(1, epoch_num - args.warmup_epochs))
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        min_factor = args.min_lr / args.lr
        return min_factor + (1.0 - min_factor) * cosine

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    train_losses, train_accuracy = [], []
    bestacc = 0

    total_time = 0
    for _ in range(1):
        for epoch in range(epoch_num):
            start_time = time.time()
            epoch_loss, epoch_accuracy = fit(
                epoch,
                net,
                data_loader_train,
                phase='training',
                optimizer=optimizer,
                criterion=criterion
            )
            end_time = time.time()
            epoch_time = end_time - start_time
            total_time = total_time + epoch_time
            print(f"one epoch completed in : {epoch_time:.2f} seconds")
            print(f"{epoch} epochs completed in {total_time:.2f} seconds")

            scheduler.step()

            train_losses.append(epoch_loss)
            train_accuracy.append(epoch_accuracy)

            if epoch_accuracy >= bestacc:
                bestacc = epoch_accuracy
                torch.save(net.state_dict(), des_path + 'net_params_best.pth')

            if epoch % args.save_interval == 0 and epoch >= 300:
                torch.save(net.state_dict(), des_path + 'epoch_' + str(epoch) + '_net_params.pth')

            with open(os.path.join(path_rst, 'train_losses.txt'), 'w') as f:
                for v in train_losses:
                    f.write(str(v) + '\n')

            with open(os.path.join(path_rst, 'train_accuracy.txt'), 'w') as f:
                for v in train_accuracy:
                    f.write(str(v) + '\n')
