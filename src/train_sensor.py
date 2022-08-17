import argparse
from pickletools import optimize
import torch
import torch.nn as nn

from tqdm import tqdm
import os
import pathlib
import wandb
import json


    
def train(args, experiment, dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    model.train()
    
    train_loss, correct = 0, 0

    for batch, (x, y) in tqdm(enumerate(dataloader)):
        

        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()

        loss = loss.item()
        train_loss += loss

        correct += (pred.argmax(1)==y).type(torch.float).mean().item()

    train_loss /= num_batches
    correct /= size

    print(f'loss: {train_loss:>7f}\n Train Accuracy : {(100*correct):>8f}')
    experiment.log({
        'Train Accuracy' : 100 * correct, 
        'Train Loss' : train_loss
        })

def test(args, experiment, dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()

    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            test_loss += loss_fn(pred, y).item()

            correct += (pred.argmax(1)==y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f'Test Accuracy : {(100*correct):>0.1f}%, Avg loss : {test_loss:>8f} \n')

    experiment.log({
        'Test Accuracy' : 100 * correct, 
        'Test Loss' : test_loss
        })