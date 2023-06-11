#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from util import pretrain_dataset
from unet import unet

# %%
def get_parser():
    parser = argparse.ArgumentParser(description='Baseline for ADE20K', add_help=False)
    
    #hardware
    parser.add_argument('--device', default='cuda',
                        help='default(cuda)')
    
    #dataloader
    parser.add_argument('--data', default='/home/tnt/Downloads/NIH_images',
                        help='Data root ')
    parser.add_argument('--batch_size', default=5, type=int,
                        help='batch size per gpu')
    parser.add_argument('--num_workers', default=4, type=int)

    #optimizer
    parser.add_argument('--lr', default=0.01, type=float,
                        help='absloute learning rate ')
    parser.add_argument('--weight_decay', default=0.0005, type=float)

    #run
    parser.add_argument('--epochs', default=64, type=int)
    
    #print
    parser.add_argument('--print_freq', default=500, type=int,
                        help='print loss at iter % preint_freq == 0')
    
    #save
    parser.add_argument('--output_dir', default='checkpoint', type=str,
                        help='path for saving checkpoint')

    parser.add_argument('--log_dir', default='log', type=str,)

    return parser

# %%
def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

    #Dataset
    train_transform = transforms.Compose([
        transforms.Resize(512),
        #transforms.ToTensor(),
    ])
    
    train_dataset = pretrain_dataset.Pretrain_Dataset(
        root=args.data, split='train', transform=train_transform
    )
    for x, y in train_dataset:
        print(x.shape, x.dtype, y.shape, y.dtype)
        break

    #Loader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )

    for x, y in train_loader:
        print(x.shape, x.dtype, y.shape, y.dtype)
        break

    model = unet.UNet2D()
    model = model.to(device)
    #print(model)

    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    #Loss 
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        
        #train one epoch
        model.train()
        total_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            x = torch.reshape(x, (5, 1, 512, 512)).float()
            y = torch.reshape(y, (5, 1, 512, 512)).float()
            logit = model(x)
            loss = criterion(logit, y)
            
            #backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 1 ==0:
                print(f"Loss: {loss:.6f}")

            total_loss += loss * args.batch_size 
        total_loss /= len(train_dataset)
        print(f"avg per epoch loss: {total_loss:.3f}")
    print("Done!")
            
if __name__ == '__main__':
    import sys
    sys.argv = ['']
    parser = get_parser()
    args = parser.parse_args()
   
    main(args)
