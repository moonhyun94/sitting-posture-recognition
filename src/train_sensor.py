import argparse
from pickletools import optimize
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader.sensor_loader import CustomDataset
from models.sensor_C3D import C3D

from tqdm import tqdm
import os
import pathlib
import wandb
import json

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

    print(f'loss: {train_loss:>7f}\n Train Accuracy : {(100*correct):>8f}')

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

    print(f'Test Accuracy : {(100*correct):>0.1f}%, Avg loss : {test_loss:>8f} \n')

    wandb.log({
        'Test Accuracy' : 100 * correct, 
        'Test Loss' : test_loss
        })
        

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-verbose', default=0, type=int,
                        help="Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. default: 0")
    parser.add_argument('-model', default='C3D', help='Choose model type Conv_5_C3D, Conv_3_C3D')
    parser.add_argument('-epochs', default=EPOCHS, type=int, help="Number of epoch to train. default: 5")
    parser.add_argument('-batch_size', default=BATCH_SIZE, type=int, help="Number of samples per gradient update. default: 1")
    parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE, help='Set learning rate')
    parser.add_argument('-chkt_filename', default='./c3d-pretrained.pth', help="Model Checkpoint filename to save.")
    # parser.add_argument('-t', '--fine_tunning_interval', default=1, type=int, help="Fine-tuning interval. default: 1")
    parser.add_argument('-gn','--gpu_number', default=0, type=int,
                        help='Number of GPU. default: 0')
    # parser.add_argument('-vdp', '--video_data_path', default='./Large_Captcha_Dataset',
    #                     help="Location of video dataset. default: \'./Large_Captcha_Dataset\'")
    parser.add_argument('-sdp', '--sensor_data_path', default='./data/Train',
                        help="Location of sensor dataset. default: \'./Large_Captcha_Dataset\'")
    parser.add_argument('-wandb', default=False, action="store_true",
                        help="Do you wanna use wandb? just give it True! default:False")
    parser.add_argument('-pn', '--project_name', default='Sensor_C3D', required=True,
                        help="Set wandb project name")
    args = parser.parse_args()
    
    return args

if __name__=="__main__":

    args = get_args()
    experiment=None

    if args.wandb:
        experiment = wandb.init(
            project=args.project_name, entity='captcha-active-learning-jinro', config={
                'learning_rate' : args.learning_rate,
                'epochs' : args.epochs,
                'batch_size' : args.batch_size,
            }

        )

    print(json.dumps({
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }, indent=4))

    device = torch.device('cuda')

    my_model = C3D(NUM_CLASSES, pretrained=True).to(device)

    train_dataset = CustomDataset(mode='Train')
    test_dataset = CustomDataset(mode='Test')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(my_model.parameters(), lr=args.learning_rate)

    for epoch in tqdm(range(EPOCHS)):
        print(f'Epoch : {epoch+1} \n--------------------------------')

        train(train_dataloader, my_model, loss_fn, optimizer)
        test(test_dataloader, my_model, loss_fn)

    print('Done')