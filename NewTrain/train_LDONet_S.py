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
from torch.optim import lr_scheduler

from models.dataset import MyDataset
from models.LDONet_S import LDONet_S


def fit(epoch, model, data_loader, phase='training', optimizer=None, criterion=None):
    if phase != 'training' and phase != 'testing':
        raise TypeError('input error!')

    if phase == 'training':
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_correct = 0

    for datas, target in data_loader:
        datas = datas.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
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
        description="LDONet-S direct training"
    )

    parser.add_argument("--dataset", type=str, default="Blue", choices=["Blue", "Green", "NIR", "Red", "HFUT", "PolyU", "Tongji"])
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epoch_num", type=int, default=200)
    parser.add_argument("--id_num", type=int, default=250)
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--save_interval", type=int, default=2)
    parser.add_argument("--student_weight", type=float, default=0.7)
    parser.add_argument("--student_path", type=str, default=None)

    parser.add_argument("--train_set_file", type=str, default='../dataset/train_Blue_linux.txt')
    parser.add_argument("--des_path", type=str, default='../results/Blue/LDONet_S/checkpoint/')
    parser.add_argument("--path_rst", type=str, default='../results/Blue/LDONet_S/rst_test/')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    batch_size = args.batch_size
    epoch_num = args.epoch_num
    num_classes = args.id_num

    des_path = args.des_path
    path_rst = args.path_rst

    os.makedirs(des_path, exist_ok=True)
    os.makedirs(path_rst, exist_ok=True)

    train_set_file = args.train_set_file

    trainset = MyDataset(txt=train_set_file, transforms=None, train=True, imside=128, outchannels=1)
    data_loader_train = DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
    )

    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
    print('------Init LDONet-S Model------')
    print(f"Dataset: {args.dataset}")
    print(f"Classes: {num_classes}")
    print(f"LR: {args.lr}")
    print(f"Weight Decay: {args.weight_decay}")

    net = LDONet_S(
        label_num=num_classes,
        weight=args.student_weight,
    )
    net.cuda()

    if args.student_path:
        net.load_state_dict(torch.load(args.student_path, map_location='cuda'))
        print(f"Resume from: {args.student_path}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(current_epoch):
        if current_epoch < args.warmup_epochs:
            return float(current_epoch + 1) / float(max(1, args.warmup_epochs))

        progress = (current_epoch - args.warmup_epochs) / float(max(1, epoch_num - args.warmup_epochs))
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        min_factor = args.min_lr / args.lr
        return min_factor + (1.0 - min_factor) * cosine

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    train_losses, train_accuracy = [], []
    bestacc = 0.0
    total_time = 0.0

    for epoch in range(epoch_num):
        start_time = time.time()
        epoch_loss, epoch_accuracy = fit(
            epoch,
            net,
            data_loader_train,
            phase='training',
            optimizer=optimizer,
            criterion=criterion,
        )
        epoch_time = time.time() - start_time
        total_time += epoch_time

        scheduler.step()

        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)

        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"epoch {epoch}: lr={current_lr:.8f}, epoch_time={epoch_time:.2f}s")

        if epoch_accuracy >= bestacc:
            bestacc = epoch_accuracy
            torch.save(net.state_dict(), os.path.join(des_path, 'net_params_best.pth'))

        if epoch % args.save_interval == 0 and epoch >= 200:
            torch.save(net.state_dict(), os.path.join(des_path, f'epoch_{epoch}_net_params.pth'))

        with open(os.path.join(path_rst, 'train_losses.txt'), 'w') as f:
            for v in train_losses:
                f.write(str(v) + '\n')

        with open(os.path.join(path_rst, 'train_accuracy.txt'), 'w') as f:
            for v in train_accuracy:
                f.write(str(v) + '\n')

    print("=" * 70)
    print("LDONet-S training complete")
    print(f"Best accuracy: {bestacc:.3f}%")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Model saved in: {des_path}")
    print("=" * 70)
