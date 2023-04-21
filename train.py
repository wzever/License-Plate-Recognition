import torch
from torch import nn
import argparse
import os
from utils import print_args
from dataset import split_set
from model import *
import time
import matplotlib.pyplot as plt
import tensorboardX

import json
labels = os.listdir('VehicleLicense/Data')
with open('VehicleLicense/label_match.json', encoding='utf-8') as f:
    chinese_label = json.load(fp=f)
mapping = {k: chinese_label[v] for k, v in enumerate(labels)}

def train(args):
    # load training dataset 
    batch_size = args.batch_size
    train_ratio = args.train_ratio
    train_iter, test_iter = split_set(train_ratio, batch_size)

    # set hyper-parameters
    lr = args.lr
    wd = args.wd
    lr_period = args.lr_period
    lr_decay = args.lr_decay
    num_epochs = args.num_epochs

    # get current device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get training net
    net = CNN().to(device)

    # set optimizer & scheduler
    optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_period, lr_decay)

    # set loss function
    loss_func = nn.CrossEntropyLoss()

    # log training process
    if not os.path.exists("log"):
        os.mkdir("log")  
    writer = tensorboardX.SummaryWriter("log")

    step_n = 0
    best_acc = 0
    accs = []

    for epoch in range(num_epochs):
        start_time = time.time()
        train_corr, test_corr = 0, 0
        for i, (img, labels) in enumerate(train_iter): 
            net.train()
            img, labels = img.to(device), labels.to(device)
            pred = net(img)
            loss = loss_func(pred, labels).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            _, pred = torch.max(pred.data, dim=1)
            train_corr += pred.eq(labels.data).cpu().sum().item()

            step_n += 1
            # 记录训练过程
            writer.add_scalar("train loss", loss.item(), global_step=step_n)

        end_time = time.time()
        train_acc = 100.0 * train_corr / batch_size / len(train_iter)
        print(f'epoch {epoch}, train time = {end_time - start_time}')

        scheduler.step()
        end_time = time.time()

        for i, (img, labels) in enumerate(test_iter): 
            net.eval() # test
            img, labels = img.to(device), labels.to(device)

            pred = net(img)
            loss = loss_func(pred, labels).mean()

            _, pred = torch.max(pred.data, dim=1)
            batch_corr = pred.eq(labels.data).cpu().sum().item()
            test_corr += batch_corr

        test_acc = 100.0 * test_corr / batch_size / len(test_iter)
        print(f'train_acc = {train_acc}, test_acc = {test_acc}')
        accs.append(test_acc)

        if not os.path.exists("pretrained"):
            os.mkdir("pretrained")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), f"pretrained/{epoch}-{test_acc:.4f}.pt")
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train and evaluate classification net.')
    parser.add_argument('--train_ratio', default=0.85)
    parser.add_argument('--num_epochs', default=100)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--lr', default=2e-4)
    parser.add_argument('--wd', default=5e-4)
    parser.add_argument('--lr_period', default=10)
    parser.add_argument('--lr_decay', default=0.95)

    args = parser.parse_args()
    print_args(args)
    return args


if __name__ == '__main__':
    train(parse_arguments())