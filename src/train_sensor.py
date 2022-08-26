import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader.sensor_loader2 import CustomDataset

from tqdm import tqdm
import os
import pathlib
import wandb
import json

BATCH_SIZE = 1
NUM_CLASSES = 6
LEARNING_RATE = 1e-4
EPOCHS = 20


def train_sensor(args, experiment, model, train_dataloader, loss_fn, optimizer, device):
    size = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)

    model.train()
    
    train_loss, correct = 0, 0

    for batch, (x, y) in enumerate(train_dataloader):
        # print('xshape', x.shape)
        # print('y', y)
        
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)

        pred = model(x)
        # print('pred', pred)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()

        loss = loss.item()
        train_loss += loss

        correct += (pred.argmax(1)==y).type(torch.float).mean().item()

    train_loss /= num_batches
    correct /= size

    print(f'Train loss: {train_loss:>7f} Train Accuracy : {(100*correct):>8f}%')
    wandb.log({
        'train_acc' : correct, 
        'train_loss' : train_loss
        })

def test_sensor(args, experiment, model, test_dataloader, loss_fn, optimizer, device):
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)

    model.eval()

    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            test_loss += loss_fn(pred, y).item()

            correct += (pred.argmax(1)==y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f'Test Accuracy : {(100*correct):>0.1f}%, Avg loss : {test_loss:>8f} \n')

    wandb.log({
        'test_acc' : correct, 
        'test_loss' : test_loss
        })

