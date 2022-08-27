import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.data import CustomDataset
from models.sensor_predictor import SensorPredictor


def train_step(context, data):
    sensor_predictor = context["sensor_predictor"]
    optimizer = context["optimizer"]

    optimizer.zero_grad()

    frames, sensors = data

    sensor_preds =  sensor_predictor(frames)

    loss = torch.mean((sensor_preds - sensors)**2)
    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def eval_step(context, data):
    sensor_predictor = context["sensor_predictor"]

    frames, sensors = data

    sensor_preds =  sensor_predictor(frames)
    loss = torch.mean((sensor_preds - sensors)**2)

    return loss


def save(context, path):
    state_dict = dict()

    for key in context.keys():
        if hasattr(context[key], "state_dict"):
            state_dict[key] = context[key].state_dict()

    torch.save(state_dict, path)


def load(context, path):
    state_dict = torch.load(path, map_location="cpu")

    for key in context.keys():
        if hasattr(context[key], "load_state_dict") and key in state_dict.keys():
            context[key].load_state_dict(state_dict[key])


def train(args):
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    trainset = CustomDataset("train")
    testset = CustomDataset("test")

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=4)

    sensor_predictor = SensorPredictor().to(device)
    optimizer = optim.AdamW(sensor_predictor.parameters(), lr=args.lr)

    context = {
        "sensor_predictor": sensor_predictor,
        "optimizer": optimizer
    }

    min_test_loss = np.inf

    for e in range(args.epochs):
        train_loss = 0.0
        test_loss = 0.0

        sensor_predictor.train()

        for frames, sensors, _ in tqdm(train_loader, desc="Training"):
            frames = frames.to(device)
            sensors = sensors.to(device)

            loss = train_step(context, [frames, sensors])
            train_loss += loss / len(train_loader)

        sensor_predictor.eval()

        for frames, sensors, _ in tqdm(test_loader, desc="Evaluating"):
            frames = frames.to(device)
            sensors = sensors.to(device)

            loss = eval_step(context, [frames, sensors])
            test_loss += loss / len(test_loader)

        print(f"Epochs {e + 1}/{args.epochs}")
        print(f"Train loss: {train_loss:.8f}")
        print(f"Test loss: {test_loss:.8f}")

        if min_test_loss > test_loss:
            min_test_loss = test_loss
            save(context, "ckpts/sensor_predictor1.pt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
