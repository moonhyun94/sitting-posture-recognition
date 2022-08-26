import argparse
import torch
import torch.nn as nn
<<<<<<< HEAD
from torch.utils.data import DataLoader
from dataloader.sensor_loader2 import CustomDataset
=======
>>>>>>> 60bab893a28460f24cc22e968cc37b850cbcdef8

from tqdm import tqdm
import os
import pathlib
import wandb
import json


<<<<<<< HEAD
def train_sensor(args, experiment, model, train_dataloader, loss_fn, optimizer, device):
    size = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)

=======
    
def train(args, experiment, dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
>>>>>>> 60bab893a28460f24cc22e968cc37b850cbcdef8
    model.train()
    
    train_loss, correct = 0, 0

<<<<<<< HEAD
    for batch, (x, y) in enumerate(train_dataloader):
        # print('xshape', x.shape)
        # print('y', y)
=======
    for batch, (x, y) in tqdm(enumerate(dataloader)):
>>>>>>> 60bab893a28460f24cc22e968cc37b850cbcdef8
        
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

<<<<<<< HEAD
    print(f'Train loss: {train_loss:>7f} Train Accuracy : {(100*correct):>8f}%')
    wandb.log({
        'train_acc' : correct, 
        'train_loss' : train_loss
        })

def test_sensor(args, experiment, model, test_dataloader, loss_fn, optimizer, device):
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
=======
    print(f'loss: {train_loss:>7f}\n Train Accuracy : {(100*correct):>8f}')
    experiment.log({
        'Train Accuracy' : 100 * correct, 
        'Train Loss' : train_loss
        })

def test(args, experiment, dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
>>>>>>> 60bab893a28460f24cc22e968cc37b850cbcdef8

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

<<<<<<< HEAD
    wandb.log({
        'test_acc' : correct, 
        'test_loss' : test_loss
        })

=======
    experiment.log({
        'Test Accuracy' : 100 * correct, 
        'Test Loss' : test_loss
        })
>>>>>>> 60bab893a28460f24cc22e968cc37b850cbcdef8
