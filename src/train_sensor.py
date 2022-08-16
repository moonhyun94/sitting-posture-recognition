from pickletools import optimize
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Sensor.sensor_loader import CustomDataset
from Sensor_C3D.sensor_C3D import C3D

from tqdm import tqdm
import os
import pathlib

BATCH_SIZE = 1
NUM_CLASSES = 6
LEARNING_RATE = 1e-4
EPOCHS = 20


def train(dataloader, model, loss, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.train()
    
    train_loss, correct = 0, 0

    for batch, (x, y) in enumerate(dataloader):
        

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

    print(f'loss: {train_loss:>4f}\n Train Accuracy : {(100*correct):>4f}')

def test(dataloader, model, loss):
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

    print(f'Test Accuracy : {(100*correct):>0.1f}%, Avg loss : {test_loss:>5f} \n')

if __name__=="__main__":


    device = torch.device('cuda:1')
    my_model = C3D(NUM_CLASSES, pretrained=True).to(device)

    train_dataset = CustomDataset(mode='Train')
    test_dataset = CustomDataset(mode='Test')

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(my_model.parameters(), lr=LEARNING_RATE)

    for epoch in tqdm(range(EPOCHS)):
        print(f'\n Epoch : {epoch+1} \n--------------------------------')

        train(train_dataloader, my_model, loss_fn, optimizer)
        test(test_dataloader, my_model, loss_fn)

    print('Done')